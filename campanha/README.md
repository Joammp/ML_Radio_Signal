# Campanha de busca de HP — modulações digitais

Infraestrutura para rodar a busca de hiperparâmetros das quatro modulações **digitais**
(ASK, PSK, APSK, QAM) no Colab, de forma retomável. AM e FM ficam de fora por serem
analógicas; além disso FM tem uma única classe no `GROUP_MAP`, o que torna a
classificação degenerada.

Trabalho de 19-21/08/2026. **A aplicação com interface ainda NÃO foi construída** — o
que existe aqui é a Fase 0 (medições que decidem a arquitetura) e o script de treino
parametrizado. Ver "Estado atual" no fim.

---

## Escala do problema

12 arquiteturas × 5 folds × 4 grupos = **240 folds**, cada um treinando sobre o subset
completo do grupo (não uma amostra):

| Grupo | classes | amostras | batches/época | ~por fold (120 ép.) | 60 folds |
|---|---|---|---|---|---|
| ASK | 3 | 196.608 | 1.474 | ~40 min | ~40 h |
| PSK | 7 | 458.752 | 3.440 | ~94 min | ~94 h |
| APSK | 4 | 262.144 | 1.966 | ~53 min | ~53 h |
| QAM | 5 | 327.680 | 2.457 | ~67 min | ~67 h |

**~254 h de T4 em fp32.** É uma campanha de semanas, retomada entre sessões.

---

## Fase 0 — as duas medições que decidiram a arquitetura

### Gate 1: paralelismo numa GPU — **REPROVADO** (`bench_sintetico.py`)

Hipótese testada: a T4 estaria subutilizada por esta rede pequena, então rodar os 4
grupos concorrentes renderia. **Falso.**

| | passos/s por job | agregado |
|---|---|---|
| 1 job sozinho | 90,0 | 90,0 |
| 4 jobs concorrentes | 20,4 | **81,6** |

`nvidia-smi` com **um** job: 100% de utilização. A GPU já satura sozinha, e a
concorrência *reduz* a vazão agregada em ~9%. Projeção para a carga real:
`90 / (4 × 20,4) = 1,10` → concorrente é **10% mais lento** que sequencial.

O script reporta um "speedup de 1,28×" que é **artefato**: vem de amortizar ~6,5 s de
init de contexto CUDA por job. Em jobs de 11 s isso é 60% do tempo; nos folds reais de
40-90 min é 0,2%. Ao ler a saída, olhar `passos_por_s`, não o wall-clock.

Memória não é o gargalo: 4 contextos da maior arquitetura ocupam 3,8 GB de 15,4 GB.

### Gate 2: precisão mista — **APROVADO** (`bench_amp.py`)

Se uma GPU satura com um job, o ganho vem de baratear cada job. Todos os canais da busca
(32…1024) são múltiplos de 8, condição para o cuDNN usar tensor cores.

| Arquitetura | fp32 determin. | AMP determin. | ganho |
|---|---|---|---|
| 2L_32-64 | 6,60 ms | 6,09 ms | 1,08× |
| 4L_32-64-128-256 | 12,18 ms | 7,92 ms | 1,54× |
| 5L_64-128-256-512-1024 | 42,71 ms | 21,51 ms | **1,99×** |
| **soma (ponderado por custo)** | 61,49 ms | 35,52 ms | **1,73×** |

Quanto maior a rede, maior o ganho — redes grandes são limitadas por computação, as
pequenas por latência de lançamento. Como a campanha é dominada pelas arquiteturas
grandes, o ganho real deve ficar entre 1,7× e 1,9×: **254 h → ~135-150 h**.

Dois achados laterais:

- **Determinismo custa pouco.** `cudnn.benchmark=True` sozinho rende só 1,09×. Estávamos
  pagando ~9% por reprodutibilidade.
- **Não há trade-off.** `amp_determin` (1,99×) empata com `amp_benchmark` (2,14×) na 5L e
  ganha na 4L. Dá para ter AMP **mantendo** `cudnn.deterministic=True`.

### Gate 3: o AMP muda o colapso? — **INCOMPLETO** (`valida_amp.py`)

AMP muda a numérica, e este projeto gira em torno de um colapso detectado comparando a
loss com `ln(C)` a tolerância de 0,02. A validação compara fp32 e AMP no QAM 4L a 9e-5,
contra referência conhecida.

**O braço fp32 reproduziu a referência exatamente** (outra VM, dataset regenerado):

| Seed | Referência | fp32 na validação |
|---|---|---|
| 0 | escape `None`, best 20,05% | escape `None`, best 20,05% @ep13 |
| 1 | escape ep 16,2, best 24,77% | escape ep 16,2, best 24,77% @ep34 |

**O braço AMP não chegou a rodar** — a VM do Colab caiu antes (quarta queda da sessão).

> **Por isso `busca_hp.py` NÃO tem AMP habilitado.** Rodar `valida_amp.py` até o fim é o
> primeiro passo ao retomar. O critério está escrito no script e não deve ser afrouxado
> depois de ver o resultado: AMP passa se concordar com fp32 sobre **quem escapa e quem
> morre**. Diferença de acurácia final é tolerável; mudar o seed 1 de vivo para morto, ou
> fazer o seed 0 escapar, reprova.

---

## Hiperparâmetros e o porquê

`busca_hp.py` é a **fonte de verdade** e se edita à mão. Ele nasceu gerado por
`gerar_busca_hp.py` a partir do notebook do branch
`atualiza-busca-hp-estratificacao`, mas o gerador foi aposentado em 28/08/2026
(ver `aposentados/LEIA-ME.md`): a busca de redução foi escrita direto no
`busca_hp.py` e regenerar a apagaria.

| Parâmetro | Notebook | Aqui | Motivo |
|---|---|---|---|
| `LEARNING_RATE` | 1e-3 | **4,5e-5** (CNN) · **4,5e-4** (redução/ResNet) | 1e-3 colapsa o QAM em 15/15 folds; 4,5e-5 truncava a redução no teto de épocas |
| `LR_PATIENCE` | 5 | **8** | pedido |
| `EARLY_STOP` | 15 | **20** | acompanha a paciência maior |
| `drive_push/pull` | ativo na VM | **no-op** | o token de escopo `drive` não sai da máquina local |

> **`9e-5` não é garantia.** Em 3 seeds testadas no QAM 4L, 1 ainda colapsou nesse valor, e
> o escape observado ocorreu de fato em **4,5e-5**, após a primeira redução de platô. Se o
> QAM voltar a colapsar em massa, 4,5e-5 é o próximo valor a tentar.

Aumentar a paciência **encarece** a campanha: o early stop só dispara depois de o LR
atingir `LR_MIN`, o que exige no mínimo 6×paciência épocas de estagnação.

---

## Como continuar em outro PC

### 1. Colab CLI (Windows exige remendo)

O `google-colab-cli` (v0.6.0) **não suporta Windows oficialmente**, mas roda com um patch
de uma linha. Requer Python ≥ 3.12.

```bash
python -m venv C:/Users/<voce>/.colab-cli
C:/Users/<voce>/.colab-cli/Scripts/python.exe -m pip install google-colab-cli "jupyter-kernel-client==0.15.0"
C:/Users/<voce>/.colab-cli/Scripts/python.exe campanha/ferramentas/colab_cli_windows_patch.py
```

Três armadilhas, todas já pagas:

- **`termios`**: `colab_cli/console.py` importa `termios`/`tty` (POSIX) no topo, e
  `commands/execution.py` o importa incondicionalmente — o CLI inteiro morre no import. O
  patch torna esse import preguiçoso. Reaplicar após qualquer upgrade.
- **`jupyter-kernel-client`**: a dependência é declarada **sem pin**, e a 1.0.x renomeou
  `KernelClient` → `JupyterKernelClient`. Todo `colab exec` morre com `AttributeError`.
  Fixar em `0.15.0`.
- **Não instalar via `uv tool install` de dentro de um app empacotado (MSIX)**: o
  `%APPDATA%` é redirecionado e o trampolim aponta para um caminho que só existe dentro do
  sandbox (`uv trampoline failed to canonicalize script path`). Instalar fora de
  `%APPDATA%`.

Autenticação: `colab whoami` num terminal interativo, aprovar no navegador, colar o
código. Não precisa de `gcloud`. Fica em cache com refresh token.

### 2. Dado

```bash
colab new -s campanha --gpu T4
colab exec -s campanha -f campanha/prep_all.py     # ~9 min download + ~2 min build
```

Baixa o RadioML 2018.01A do Kaggle (público, **sem credencial**) e constrói os 5 subsets.
O processo é **determinístico** — filtro puro por (grupo, SNR), sem RNG. Conferência:

```
QAM fingerprint = 847f44cb75a4c7b653744e34d30c039473e8daa805ec532ea46f233310ab9981
```

Validado em **três VMs independentes** nesta sessão. Se divergir, algo mudou na fonte.

### 3. Treinar

```bash
colab exec -s campanha -f campanha/busca_hp.py --grupos QAM
```

`busca_hp.py` aceita `--grupos`, `--lr`, `--lr-patience`, `--early-stop`, `--device`.
Retoma sozinho: `load_results` e `ensure_folds` releem os artefatos e pulam folds já
concluídos, com asserts de consistência (`k_folds`, `seed`, `n_train`, `strat`).

---

## Limites do Colab gratuito, medidos

| | |
|---|---|
| GPU | T4 aloca; cota esgota após ~2 h → `Service Unavailable` por horas. L4 exige Pro |
| VM | **cai sozinha a cada ~40-55 min** — 4 quedas nesta sessão, inclusive em CPU |
| Repreparo | ~11 min por sessão nova (18 GB do Kaggle + build) |
| CPU | 0,95 s/batch; `set_num_threads(2)` não ajuda. ~250× mais lenta que a T4. Inviável |

**Atribuição órfã bloqueia nova GPU.** Ao perder a sessão, o estado local é limpo mas a
atribuição continua no servidor, e `colab new --gpu` falha com `TooManyAssignmentsError`.
`colab stop` **não** resolve (depende do registro local). Usar:

```bash
C:/Users/<voce>/.colab-cli/Scripts/python.exe campanha/ferramentas/unassign_orfa.py
```

**Dimensionar por passos, não por épocas.** O escape do colapso no QAM veio por volta do
passo 4.000; reduzir `N_TRAIN` encolhe os batches por época e não barateia chegar lá.

**Um job por subprocesso, nunca por thread no mesmo kernel.** Scripts distintos rodando no
mesmo kernel do Colab colidem globais (`log`, `LOG`): uma thread antiga passa a escrever no
log da nova. Aconteceu aqui e travou `prep.log` em 49 bytes enquanto o build corria normal.

---

## Estado atual

**Pronto:** os dois gates de infraestrutura, o gerador e o script de treino parametrizado,
o pipeline determinístico dos 5 subsets, e as ferramentas de recuperação do CLI.

**Pendente, em ordem:**

1. Terminar `valida_amp.py` (braço AMP). Decide se AMP entra — vale ~100 h de campanha.
2. Se aprovar, acrescentar `autocast`/`GradScaler` ao `train_one_fold` em `busca_hp.py`.
3. Construir a aplicação: painel com job por grupo, log ao vivo, detalhe de execução,
   seleção de GPU, espelho local do progresso e upload ao Drive **a partir da máquina
   local** (o `save_fold_result` já grava JSON a cada fold; falta só sincronizar).
4. Decidir o modelo de execução. O paralelismo numa GPU foi reprovado; sessões paralelas
   do Colab Pro são a alternativa, mas a FAQ oficial **não publica** limites de sessões
   simultâneas e as fontes secundárias sugerem 1 GPU ativa mesmo no Pro+.
