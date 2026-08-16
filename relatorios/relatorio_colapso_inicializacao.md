# Colapso de inicialização na busca de arquitetura — diagnóstico e correção

**Data:** 16 de agosto de 2026
**Escopo:** `BUSCA_HP_Classes_1cap(3).ipynb` — busca de arquitetura com K-Fold CV sobre RadioML 2018.01A
**Grupos analisados:** ASK, PSK, APSK, QAM (AM e FM fora do escopo)

---

## Resumo

Uma fração grande dos folds da busca de arquitetura termina com acurácia de validação exatamente igual ao *chance level* (`1/num_classes`) e loss de treino exatamente igual a `ln(num_classes)`. Esses folds não estão treinando mal — **não estão treinando**: a rede produz a distribuição uniforme desde a primeira época e o gradiente é nulo.

O diagnóstico inicial da equipe era que APSK e QAM precisavam de um *learning rate* menor. Os dados não sustentam isso: em folds colapsados o LR caiu de `1e-3` para `2.5e-4` sem alterar a loss **na quarta casa decimal**. Reduzir o LR multiplica um gradiente nulo por um número menor.

A causa é a inicialização dos pesos. A correção — inicialização Kaiming explícita, *warmup* de LR, *clipping* de gradiente e, como rede de segurança, detecção de colapso com reinicialização — eliminou o colapso em ASK, PSK e APSK. Em QAM o colapso também foi eliminado, mas revelou um segundo problema, de generalização, que permanece em aberto.

| Grupo | C | baseline mortos | corrigido mortos |
|---|---|---|---|
| ASK | 3 | 0/4 | 0/4 |
| PSK | 7 | 2/4 | 0/4 |
| APSK | 4 | 4/4 | 0/4 |
| QAM | 5 | 4/4 | 0/4 (ver §6) |

---

## 1. O problema observado

Resultados salvos no Drive de execuções anteriores:

- **APSK** (4 classes): **9 de 9 folds** colapsados, 46 épocas cada. Loss final `1.3863` em todos — `ln(4) = 1.38629`.
- **PSK** (7 classes): **38 de 60 folds** colapsados. Loss `1.9459` — `ln(7) = 1.94591`.

O padrão mais revelador é a dependência de *seed*, não de arquitetura:

```
4L_32-64-128-256   fold0 → 14.29%   (colapsado)
4L_32-64-128-256   fold2 → 96.00%   (excelente)
4L_32-64-128-256   fold3 → 14.29%   (colapsado)
```

Mesma arquitetura, mesmo LR, mesmos dados, mesma configuração. O que varia é a seed do fold.

Histórico de um fold colapsado (PSK, `4L_32-64-128-256`, fold 0):

```
ep  1  tr=14.23%  vl=14.29%  trloss=1.9709  lr=1.00e-03
ep  2  tr=14.26%  vl=14.29%  trloss=1.9460  lr=1.00e-03
...
ep 11  tr=14.18%  vl=14.29%  trloss=1.9460  lr=5.00e-04
...
ep 20  tr=14.07%  vl=14.29%  trloss=1.9459  lr=2.50e-04
```

O LR foi reduzido quatro vezes. A loss não se moveu.

**Custo estimado:** somando APSK e PSK, aproximadamente 1.100 épocas de GPU produziram registros sem conteúdo informativo. Pior que o desperdício: esses folds entram no JSON de resultados como se fossem medições válidas, distorcendo o ranking de arquiteturas.

---

## 2. Diagnóstico

`ln(C)` é o valor exato da entropia cruzada de uma predição uniforme sobre `C` classes. Uma loss cravada nesse valor significa que o modelo emite a mesma probabilidade para todas as classes, independentemente da entrada.

Nessa condição o gradiente da camada de saída é praticamente nulo e não há sinal para propagar. Reduzir o LR não recupera — apenas congela o estado morto mais devagar. Isso é confirmado empiricamente pelo trecho acima.

Duas consequências práticas:

1. **A ação correta é reinicializar, não reduzir o LR.** Nova seed, pesos novos.
2. **A detecção pode ser feita na primeira época.** Um fold saudável já sai da primeira época bem acima do chance (no exemplo do fold 2 acima: `vl=70.63%`, loss `1.06`). A separação entre vivo e morto é imediata, o que torna possível abortar cedo em vez de gastar 46 épocas.

### Um bug adicional encontrado no caminho

No `train_one_fold` original, a referência de paciência começava em zero:

```python
best_val_acc = 0.0
```

A primeira época, mesmo colapsada em `1/C`, era maior que zero e portanto contava como "melhora", zerando o contador de paciência. Isso atrasava toda a reação do *scheduler* em `LR_PATIENCE` épocas logo na largada. A correção é iniciar a referência no próprio chance level.

---

## 3. Metodologia

Todos os experimentos rodaram em **CPU** (6 threads), sobre os HDF5 por grupo já disponíveis no Drive — os mesmos usados pelo notebook, com SNR de 0 a 30 dB.

- Subconjuntos estratificados por classe, seed fixa (42) para o sorteio.
- Métrica de colapso: `acc ≤ chance × 1.15` **e** `|loss − ln(C)| ≤ 0.02`, avaliados na última época.
- Comparação sempre pareada: mesma seed, mesma arquitetura, mesmos índices de treino e validação.

**Configurações comparadas:**

- `baseline` — o código atual: inicialização default do PyTorch, sem warmup, sem clipping, sem normalização por amostra.
- `v2` — inicialização Kaiming explícita + warmup + clipping + detecção de colapso com restart + normalização por amostra.

---

## 4. Ablação: qual ingrediente resolve?

**Setup:** APSK, `2L_32-64`, 3 seeds, 4.000 treino / 1.500 validação, 4 épocas.

| Config | Descrição | Mortos | Acurácias |
|---|---|---|---|
| A | baseline (código atual) | **3/3** | 25.47 / 26.67 / 27.27 |
| B | + normalização por amostra | **3/3** | 25.93 / 26.07 / 25.13 |
| C | + init Kaiming, warmup, clipping | **0/3** | 45.20 / 41.53 / 47.40 |
| D | C + normalização | **0/3** | 46.00 / 37.67 / 52.07 |

**Conclusão.** A normalização por amostra — que está comentada no `H5PyDataset` — **não resolve o colapso**. Essa hipótese foi levantada (os grupos afetados, APSK e QAM, são justamente os de amplitude variável, enquanto PSK tem envelope constante) e **descartada pelo experimento**. Ela melhora a acurácia depois que a rede está viva, mas não a traz de volta.

Quem destrava é a combinação inicialização + warmup + clipping.

---

## 5. Validação estendida

**Setup:** 4 grupos × 2 arquiteturas × 2 seeds × 2 configs, 4.000/1.500, 5 épocas.

| Grupo | C | Arquitetura | baseline | v2 |
|---|---|---|---|---|
| ASK | 3 | 2L_32-64 | 78.93 / 78.47 | 74.53 / 72.40 |
| ASK | 3 | 4L_32-64-128-256 | 76.73 / 79.73 | 75.60 / 78.87 |
| PSK | 7 | 2L_32-64 | 28.30 / 51.47 | 55.07 / 57.14 |
| PSK | 7 | 4L_32-64-128-256 | **14.29 ☠ / 14.29 ☠** | **61.48 / 63.28** |
| APSK | 4 | 2L_32-64 | **25.47 ☠ / 26.67 ☠** | 51.20 / 46.40 |
| APSK | 4 | 4L_32-64-128-256 | **25.00 ☠ / 25.00 ☠** | 60.20 / 59.27 |
| QAM | 5 | 2L_32-64 | **20.67 ☠ / 20.60 ☠** | 20.53 / 22.53 |
| QAM | 5 | 4L_32-64-128-256 | **20.13 ☠ / 20.87 ☠** | 20.80 / 20.27 |

**Total: baseline 10/16 mortos → v2 0/16.**

### 5.1 Impacto no ranking de arquiteturas

A `4L_32-64-128-256` morria em 100% das seeds no PSK e no APSK. Pelo ranking atual ela seria descartada. Com a correção ela se torna a **melhor** das duas arquiteturas testadas em ambos os grupos (61–63% no PSK, 59–60% no APSK, contra 55–57% e 46–51% da `2L_32-64`).

**As conclusões de arquitetura extraídas das execuções anteriores estão invertidas nos grupos afetados.**

### 5.2 Redução de variância entre seeds

| PSK, `2L_32-64` | seed 0 | seed 1 | dispersão |
|---|---|---|---|
| baseline | 28.30% | 51.47% | **23,2 p.p.** |
| v2 | 55.07% | 57.14% | **2,1 p.p.** |

Numa busca de arquitetura, 23 pontos percentuais de variância por seed significam que o ranking mede sorte de inicialização e não mérito arquitetural. Com 5 folds isso não desaparece — vira ruído em torno da média. Este é, possivelmente, o ganho mais relevante da correção para o objetivo do trabalho.

### 5.3 Custo em grupos saudáveis

No ASK, onde não há colapso a corrigir, o v2 fica de 1 a 6 p.p. **abaixo** do baseline. A hipótese é custo de arranque da inicialização de cabeça pequena (`head_gain = 0.01`, que faz os logits nascerem próximos de zero). Em 5 épocas isso aparece; em 120 épocas espera-se que se dissolva, **mas isso não foi verificado**.

---

## 6. QAM: o colapso é resolvido, a generalização não

O QAM manteve-se no chance level mesmo com o v2 na validação acima. Para separar "limitação do método" de "subconjunto pequeno demais", foi feita uma rodada dedicada com **4× mais dado e 2,4× mais épocas**: 16.000 treino / 4.000 validação (≈200 amostras por par classe×SNR contra ≈50), 12 épocas.

| Arquitetura | baseline | v2 |
|---|---|---|
| 2L_32-64 | 20.62 ☠ / 20.02 ☠ | 21.70 (2 restarts) / 22.00 (2 restarts) |
| 4L_32-64-128-256 | 20.00 ☠ / 20.02 ☠ | **☠ (desistiu)** / 24.10 (2 restarts) |

**Mais dado não altera o baseline.** A loss de treino permaneceu imóvel em `1.6095` pelas 12 épocas inteiras, nas quatro execuções. O colapso do QAM não é falta de dado.

**O v2 muda o regime de treino:**

```
baseline:  trloss 1.6095 → 1.6095 → 1.6095    val 20.00%   (nada acontece)
v2:        trloss 1.6106 → 1.5038 → 1.2737    val ~21%     (treina, não generaliza)
```

Isso reclassifica o problema. O QAM tinha **dois** problemas empilhados: colapso de inicialização e generalização ruim. O primeiro escondia o segundo — não é possível diagnosticar generalização numa rede de gradiente nulo. Com o colapso removido, o que resta é uma rede que reduz a loss de treino de `1.61` para `1.24` sem melhorar a validação, o que é assinatura de sobreajuste, não de colapso.

Distinguir 16QAM / 32QAM / 64QAM / 128QAM / 256QAM em janelas de 1024 amostras é reconhecidamente a parte mais difícil do RadioML, e é onde os modelos da literatura também apresentam maior confusão. **Este relatório não conclui se o teto de ~24% é limitação do modelo, do pré-processamento ou do horizonte de treino.**

### 6.1 A falha do método

`QAM 4L`, seed 0: as três tentativas colapsaram e o v2 desistiu. É o único `status="dead"` em 20 execuções do v2.

O log aponta uma **falha de projeto**: cada restart reduz o LR inicial (`1e-3 → 3e-4 → 9e-5`). Mas a ablação (§4) já havia mostrado que o que resolve é a inicialização, não o LR — a configuração C funcionou com `lr=1e-3` intacto. No QAM, que arranca lentamente, reduzir o LR torna o arranque ainda mais lento, enquanto o detector corta na terceira época de qualquer forma.

**Correção pendente:** nas primeiras tentativas trocar apenas a seed, mantendo o LR; reduzir o LR só se a troca de seed não bastar.

Vale registrar a diferença de comportamento mesmo na falha: o v2 gastou 370 s e reportou `dead=True` explicitamente; o baseline gastou 530 s para produzir um `20.00%` que entraria no JSON de resultados com aparência de medição válida.

---

## 7. Solução implementada

Arquivo: [`colapso_inicializacao/train_one_fold_v2.py`](colapso_inicializacao/train_one_fold_v2.py)

### Camada 1 — Prevenir

- **Inicialização Kaiming explícita.** A default do PyTorch para `Conv1d` usa `a=√5`, que subdimensiona o ganho para ReLU e favorece ativação morta em redes fundas.
- **Cabeça de classificação com ganho 0.01.** Os logits nascem próximos de zero, o que dá saída uniforme **com** gradiente, em vez de uniforme **sem** gradiente. A distinção é exatamente o modo de falha em questão.
- **Warmup linear** de `lr/100` até `lr`, limitado a 30% do orçamento total de passos.
- **Clipping de gradiente** com norma máxima 5.

### Camada 2 — Detectar

```python
def is_dead(loss, acc_pct, num_classes, prev_acc=None):
    chance_acc, chance_loss = 100.0/num_classes, math.log(num_classes)
    if not (acc_pct <= chance_acc * 1.15 and abs(loss - chance_loss) <= 0.005):
        return False
    if prev_acc is None:
        return True
    return (acc_pct - prev_acc) <= 0.5   # sem progresso
```

Três sinais simultâneos: acurácia no chance, loss em `ln(C)` e **acurácia sem progresso**. Dois strikes consecutivos confirmam. Há uma sonda intra-época a cada 150 batches, que detecta em segundos em vez de uma época inteira.

### Camada 3 — Reagir

Reinicialização com nova seed. Até 4 tentativas; se o LR inicial cair abaixo de `1e-5` ainda colapsando, o fold é marcado como falho e a busca segue — em vez de gastar 120 épocas.

### Distribuição de trabalho entre as camadas

Em 20 execuções do v2, o restart disparou apenas 3 vezes. **A prevenção faz quase todo o trabalho**; o restart é seguro de cauda.

---

## 8. Erros cometidos durante o desenvolvimento

Registrados porque cada um custou uma rodada de teste e todos foram encontrados apenas ao **executar** o código, não ao revisá-lo.

1. **Warmup de 300 passos fixos** consumia 5 épocas inteiras num loader curto (63 batches/época) — o treino rodava o tempo todo com LR ~50× abaixo do alvo. Corrigido com teto proporcional ao orçamento de passos. Um teto de "uma época" também falhou, por ficar abaixo do warmup mínimo eficaz.
2. **Detector rodava durante o warmup**, marcando como morta uma rede que estava apenas aquecendo. Guarda adicionada.
3. **Estagnação da loss como sinal de morte** falhou nos dois sentidos: uma rede colapsando ainda tem a loss escorregando em direção a `ln(C)` (lido como vida), e uma rede lenta pode ter a loss quase parada (lido como morte). Substituído pela tendência da acurácia.
4. **Contadores de strike compartilhados** entre a sonda e a checagem por época se zeravam mutuamente, e o colapso nunca chegava a dois strikes. Separados.

---

## 9. Limitações

Estas limitações são materiais para a interpretação dos números acima.

- **Execução em CPU, subconjuntos pequenos, treinos curtos** (4–12 épocas contra 120 do notebook). Os experimentos confirmam o **mecanismo** de colapso e sua correção; **não** estabelecem as acurácias finais alcançáveis.
- **Duas arquiteturas de doze.** As arquiteturas de 5 e 6 camadas, que colapsavam com mais frequência no histórico, não foram testadas.
- **Duas seeds por célula** (três na ablação). Suficiente para detectar colapso, insuficiente para estimativa de variância com confiança.
- **A regressão em grupos saudáveis não foi investigada** em treino longo.
- **Os resultados históricos usados no §1 vêm de execuções com estratificação apenas por classe**, considerada inadequada pela equipe. Isso compromete a comparabilidade das *acurácias* entre folds, mas **não** o diagnóstico do colapso: `loss = ln(C)` é propriedade da saída do modelo, não da divisão dos dados.
- **QAM permanece em aberto**, conforme §6.

---

## 10. Conclusões e próximos passos

**Conclusões.**

1. O colapso é de inicialização, não de learning rate. Reduzir o LR não recupera.
2. A correção elimina o colapso em ASK, PSK e APSK, e reduz a variância entre seeds em uma ordem de grandeza.
3. O ranking de arquiteturas das execuções anteriores está invertido nos grupos afetados e precisa ser refeito.
4. O QAM tem um segundo problema, de generalização, que o colapso vinha mascarando.

**Próximos passos, em ordem de valor.**

1. Corrigir o restart para trocar a seed sem reduzir o LR (§6.1).
2. Aplicar a correção ao notebook e refazer a busca para PSK e APSK, onde o retorno é imediato.
3. Investigar o QAM separadamente: normalização por amostra, *data augmentation*, janelas maiores.
4. Verificar em treino longo se a regressão observada no ASK persiste.

---

## Apêndice — Reprodução

Todos os scripts e resultados brutos estão em [`colapso_inicializacao/`](colapso_inicializacao/).

| Arquivo | Conteúdo |
|---|---|
| `train_one_fold_v2.py` | Implementação das três camadas |
| `ablacao.py` / `ablacao_result.json` | Ablação da §4 |
| `validacao_5grupos.py` / `validacao5.json` | Validação estendida da §5 |
| `qam_maisdados.py` / `qam_maisdados.json` | Rodada dedicada de QAM da §6 |
| `test_v2.py` | Testes do módulo: caminho feliz e colapso forçado |

Os scripts baixam os HDF5 por grupo diretamente do Drive via `token.json` (não versionado) e apagam os arquivos ao final de cada grupo. Requisitos: `torch`, `h5py`, `numpy`, `scikit-learn`, `tqdm`, `google-api-python-client`.
