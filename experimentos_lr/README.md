# Investigação: por que o QAM colapsa, e o que de fato o destrava

Experimentos de 19-20/08/2026 sobre o colapso de inicialização do QAM 4L, dando
continuidade a `relatorios/relatorio_colapso_inicializacao.md`. Todos em T4, dado
regenerado do Kaggle com o pipeline determinístico (`prep_qam.py`).

**Os JSONs de resultado se perderam junto com as VMs do Colab. Os números abaixo são o
registro.** Os scripts reproduzem tudo.

---

## Resumo

O único regime que produziu rede viva foi **o LR nascer baixo e permanecer baixo**. Tocar
em `1e-3` mata, nas quatro formas testadas:

| Forma de tocar em 1e-3 | Resultado |
|---|---|
| **começar** nele | morto — 15/15 folds |
| **rampar até** ele (warmup) | morto — 3/3 |
| **voltar** a ele após escapar | escape destruído — 2/2 |
| **passar brevemente** e reduzir | morto — 19/19 |

Consistente com o mecanismo do relatório: os primeiros passos com LR alto matam ReLUs de
forma irreversível, o gradiente vai a zero, e a partir daí o LR é irrelevante.

Isso explica por que o warmup falha aqui apesar de ser a ferramenta certa para esse
problema — ele *começa* baixo, mas termina em `1e-3`, que é exatamente onde a rede morre.

---

## Exp 1 — warmup isolado (`exp1_warmup_vs_baseline.py`)

QAM 4L, `lr=1e-3`, 3 seeds. A ablação original nunca isolou o warmup.

| Variante | Mortos | best_vla |
|---|---|---|
| A_baseline | 3/3 | 20,10 / 20,05 / 20,00% |
| W_warmonly | 3/3 | 20,00 / 20,02 / 20,00% |

Chance = 20,00%. **O warmup estava mesmo sendo aplicado** — verificado por duas vias: o LR
lido do otimizador segue a rampa (1,00% do alvo no passo 0 → 100% no passo 300), e a loss
da época 1 separa os grupos sem sobreposição:

| Variante | seed 0 | seed 1 | seed 2 | dispersão |
|---|---|---|---|---|
| A_baseline | 1,9364 | 1,8859 | 1,8513 | 0,085 |
| W_warmonly | 1,6254 | 1,6236 | 1,6244 | **0,0018** |

Com LR minúsculo no início a rede fica perto da inicialização e a loss cola em
`ln(5)=1,6094`. O baseline, com LR cheio, é chutado para um estado confiantemente errado
(loss *acima* de ln(C)) antes de colapsar. **O warmup mudou o caminho, não o destino.**

## Exp 2 — decomposição da prevenção (`exp2_decomposicao.py`)

| Variante | norm | init | warm | clip | Mortos |
|---|:--:|:--:|:--:|:--:|:--:|
| A_baseline | – | – | – | – | 3/3 |
| W_warmonly | – | – | ✓ | – | 3/3 |
| I_initonly | – | ✓ | – | – | 3/3 |
| K_cliponly | – | – | – | ✓ | 3/3 |
| C_init_warm_clip | – | ✓ | ✓ | ✓ | 3/3 |

**15/15 mortos, incluindo o controle positivo.** Sem sobrevivente, o bloco não consegue
ordenar os componentes por mérito — só estabelece um teto de eficácia a `lr=1e-3`.

Nota: `C_init_warm_clip` é apenas a **camada 1** (prevenção) do `train_one_fold_v2`. As
camadas 2 e 3 (detectar e reinicializar com nova seed) não estão aqui. Como foram os
restarts — que acabaram chegando a `9e-5` — que salvaram o QAM 4L no reteste da seção 6.2,
este resultado é consistente com o relatório, não contraditório.

## Exp 3 — estratégias de LR (`exp3_estrategias_lr.py`)

Os dois braços são bit a bit idênticos até o escape e divergem **só** pela ação sobre o LR,
então a diferença é causalmente atribuível.

| Seed | Escape | **A** (eleva p/ 1e-3) | **B** (só reduz) | Δ |
|---|---|---|---|---|
| 0 | nunca | 20,05% @ep13 · morto | 20,05% @ep13 · morto | — |
| 1 | ep 16,2 | 21,45% @ep16 · morto | **24,77% @ep34** · vivo | **+3,32 p.p.** |
| 2 | ep 10,8 | 20,12% @ep4 · morto | **21,43% @ep30** · morto | +1,31 p.p. |

**Elevar o LR de volta desfaz o escape.** No seed 1 é a diferença entre morto e vivo. B é
mais lento (pico em ep34 vs ep16) e chega mais alto — o pico do A não é convergência, é o
instante do escape logo antes de ser destruído.

Efeito colateral observado: **o scheduler de platô trabalha contra o escape.** Enquanto
colapsada não há melhora, a paciência estoura e o LR cai — no seed 0, para `1,4e-6` até a
ep15. Ele é enterrado pelo próprio scheduler antes de ter chance.

## Exp 4 e 5 — sonda intra-época (`exp4_sonda_intraepoca.py`, `exp5_sonda_continuacao.py`)

Reduzir o LR a cada `x` batches na 1ª época se não houver aprendizado. Busca binária em `x`:

| x | fração da época | LR fim ep1 | Escapes | best_vla |
|---|---|---|---|---|
| 125 | 1/2 | 2,5e-04 | 0/3 | 20,00 / 20,98 / 20,00% |
| 62 | 1/4 | 6,3e-05 | 0/3 | 20,02 / 21,43 / 20,00% |
| 31 | 1/8 | 3,9e-06 | 0/3 | 19,90 / 20,60 / 20,00% |
| 15 | 1/16 | 1,0e-06 | 0/3 | 20,52 / 19,95 / 20,95% |
| 8 | 1/31 | 1,0e-06 | 0/3 | 20,62 / 20,85 / 21,23% |
| 4 | 1/62 | 1,0e-06 | 1/3 | 20,65 / 21,30 / 20,25% |
| 2 | 1/125 | 1,0e-06 | 0/1 | 20,90% |

**19 execuções, 0 sobreviventes.** Nenhum `x` funcionou. `x=1` não chegou a rodar.

Por quê: a janela de dano é **anterior** à primeira sonda. Nos primeiros batches a `1e-3` a
loss explode muito acima de `ln(5)=1,6094`:

```
x=15, seed 0:  batch 15 → loss 6,6253   (4× ln(5))
               batch 30 → 2,0732
               batch 45 → 1,7172
               batch 150 → 1,6141   ← assintota em ln(C) por cima, nunca desce
```

Cada redução traz a loss *em direção* a `ln(C)`, mas ela nunca cruza para baixo: a rede
converge para a uniforme e trava.

**Comparação decisiva** — mesma faixa de LR, histórias diferentes:

| Condição | LR ~5e-5 | Escapes |
|---|---|---|
| começa **direto** em 9e-5 (exp 3) | desde o passo 0 | **2/3** |
| x=62: chega a 6,3e-5 na ep1 | após 62 batches a 1e-3 | **0/3** |

**Limitação do desenho, registrada:** o piso da sonda ficou em `1e-6`, e o escape só foi
observado perto de `4,5e-5`. Todo `x ≤ 31` despencava para *abaixo* da faixa onde aprender
é possível — esses runs estavam condenados pelo piso, não pela sonda. Os únicos testes
limpos foram x=125 e x=62. Refazer com piso em ~`4e-5` seria o teste correto.

---

## `kaiming_check.py` — a init medida, não deduzida

Roda em CPU local. Compara a init default do PyTorch com a Kaiming do `init_v2`:

| Camada | fan_in | fan_out | std default | std Kaiming | razão |
|---|---|---|---|---|---|
| Conv1d(2→32, k=11) | 22 | 352 | 0,1238 | 0,0732 | **0,59×** |
| Conv1d(32→64, k=7) | 224 | 448 | 0,0385 | 0,0671 | 1,74× |
| Conv1d(64→128, k=5) | 320 | 640 | 0,0323 | 0,0559 | 1,73× |
| Conv1d(128→256, k=3) | 384 | 768 | 0,0295 | 0,0511 | 1,73× |

Não é o "6× menor" que a narrativa usual sugere: o código usa `mode="fan_out"` e o default
usa `fan_in`, então a razão é `6·fan_in/fan_out`. Como os canais dobram, isso fixa em 3 da
segunda camada em diante — mas na primeira, com só 2 canais de entrada, inverte.

Efeito do ganho 0,01 na cabeça (C=5):

| | \|logit\| médio | p_max médio | loss inicial |
|---|---|---|---|
| sem ganho | 1,102 | **0,545** | 2,2512 |
| com ganho 0,01 | 0,011 | 0,203 | **1,6094** = ln(5) |

Sem o ganho a rede nasce confiantemente errada (54% numa classe, sem ter visto dado). Com
ele, nasce na assinatura exata do colapso — mas pelo motivo oposto: uniforme **com**
gradiente, e não uniforme **sem** gradiente. A loss não distingue os dois estados; só o
gradiente distingue.

**Ressalva:** esta rede é `Conv → BatchNorm → ReLU → Pool`. A BN renormaliza a cada camada,
neutralizando boa parte do argumento de propagação de variância. Sob BN, a init age menos
como "preservar sinal" e mais como ajuste do passo efetivo por camada, já que escalar `W`
por `c` não muda a saída mas escala o gradiente por `1/c`.

---

## `calib_cpu.py`

CPU do Colab: **0,95 s/batch** para a 4L, e `set_num_threads(2)` não muda nada (0,956) —
os 2 vCPUs não dão paralelismo real para esta carga. Contra ~12 ms/batch na T4. Treinar
em CPU é inviável.
