# Resultados anteriores ao reinicio (arquivados em 28/08/2026)

Estado da campanha antes de recomecar sob condicoes unificadas. Nada aqui deve
ser continuado nem mesclado com folds novos: a semeadura mudou, e misturar as
duas geracoes no mesmo `kfold_fold_results.json` produz um conjunto que nenhuma
analise consegue desfazer depois.

Todos os folds foram produzidos com

    fold_seed = SEED + arch_idx * 100 + fold_i

ou seja, **cada candidata usava um conjunto proprio de sementes**: gap1 via
142-146, gap8_head128 via 242-246, gap4 via 342-346, e assim por diante.

## 1. A semeadura confunde arquitetura com inicializacao

Este projeto gira em torno de um colapso que depende da INICIALIZACAO. Ligar a
semente ao indice da candidata confunde exatamente as duas coisas.

Medido no ASK (5 folds por candidata):

| candidata      | media  | desvio | dif. p/ a anterior |
|----------------|--------|--------|--------------------|
| 6pools         | 97,68% | 0,03   | -                  |
| gap4           | 96,67% | 0,16   | 1,01               |
| gap8           | 96,41% | 0,21   | 0,26               |
| gap1           | 96,30% | 0,29   | **0,10 < desvio**  |
| gap8_head128   | 96,03% | 0,41   | **0,27 < desvio**  |
| head64         | 92,28% | 0,34   | 3,75               |

As quatro do meio estao separadas por menos que a propria dispersao entre
folds. A ordenacao entre elas NAO e sustentada.

O colapso da gap8_head128 no QAM (4 de 5 folds em 20,00%, a chance) fica
ambiguo pelo mesmo motivo: so ela viu as sementes 242-246, e o colapso e
sabidamente dependente da semente.

## 2. O LR difere entre os reduzidos e a referencia CNN

| conjunto                               | lr       |
|----------------------------------------|----------|
| ASK, PSK, APSK, QAM (referencia CNN)   | 4,5e-05  |
| ASK_reducao, QAM_reducao, APSK_reducao | 4,5e-04  |

**Correcao de uma versao anterior deste arquivo**, que listava "reduzidos
superam a referencia por 10 a 34 p.p." entre as conclusoes validas. Nao e
valida: compara modelos treinados com LRs dez vezes diferentes.

O caso pior e o QAM, onde a referencia fica em 25,81% com chance em 20% -- a
assinatura do colapso, cuja causa documentada neste projeto e justamente o LR
(ver `campanha/escolhe_lr.py` e a medicao de 21/08/2026). O "+34 p.p." pode ser
efeito de LR, nao de arquitetura.

O controle correto existe e nunca rodou: `baseline_2L_512` e a candidata 8 do
`ARQ_REDUCAO`, com o mesmo LR, as mesmas sementes e a mesma regra de parada.

## 3. Cobertura de folds desigual

`head64` tem 3 de 5 folds (ASK); `gap4` tem 2 de 5 (QAM e APSK). E os folds nao
tem a mesma dificuldade:

| conjunto     | media por fold (f0..f4)               | amplitude |
|--------------|---------------------------------------|-----------|
| ASK_reducao  | 96,69  96,62  96,57  96,66  96,55     | 0,13 p.p. |
| APSK_reducao | 81,57  83,44  83,37  82,91  82,32     | 1,87 p.p. |
| QAM_reducao  | 39,43  38,94  **46,95**  40,42  41,46 | 8,01 p.p. |

Comparar media de 2 folds com media de 5 no QAM inventa diferenca. Pareado nos
mesmos folds, gap1 vs gap4 cai de 6,55 para 4,63 p.p.

## 4. A GPU nao foi registrada

Os folds rodaram em L4 ou T4 conforme a vaga livre, e o formato antigo nao
gravava qual. Nenhum dos conjuntos aqui tem as chaves `gpu`, `torch`, `cuda` ou
`tf32`. A partir de 27/08 o `fold_result` grava as quatro, e o `busca_hp` avisa
quando a GPU muda no meio de um conjunto.

**Ressalva:** uma versao anterior deste arquivo afirmava que "medimos que L4 e
T4 nao reproduzem o mesmo resultado com a mesma semente". Isso nao esta
estabelecido. O controle que sustentava a frase (`escolhe_lr.py`, 21/08) rodou
com o scheduler DESLIGADO contra uma referencia que o tinha ligado, e o proprio
log registra a ressalva. O scheduler sozinho explica as diferencas observadas.
Que GPUs de arquiteturas diferentes possam divergir continua verdadeiro em
teoria (ordem de reducao, escolha de kernel do cuDNN), mas nao foi medido.
`campanha/teste_gpu_colab.ipynb` foi escrito para medir isso.

## O que continua valido

- `head64` (unica sem pooling) fica ~4 p.p. abaixo das candidatas com pooling,
  apesar de ter 23x mais parametros que a gap1. A margem sobrevive ao
  pareamento por fold e e grande demais para efeito de semente. O mecanismo e o
  pooling, nao o tamanho.
- Dentro de cada conjunto nenhum hiperparametro varia. Auditados campo a campo:
  `lr`, `min_delta`, `lr_patience`, `early_stop`, `lr_min`, `stagnation_*`,
  `dropout`, `classifier`, `num_classes`. O problema nunca foi dentro do
  conjunto, e sim nas comparacoes que atravessam conjuntos.
