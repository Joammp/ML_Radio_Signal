# Busca de reducao no ASK — regime ANTERIOR (lr 4.5e-5, teto 120 epocas)

Quatro folds da candidata `gap1` (45.795 parametros, 183x menor que o baseline).

| fold | epocas | best | onde |
|---|---|---|---|
| 0 |  70 | 88,49% | ep. 52 (parou por estagnacao) |
| 1 | 120 | 92,02% | ep. 114 |
| 2 | 101 | 89,81% | ep. 36 |
| 3 | 120 | **91,21%** | **ep. 120 -- a ULTIMA** |

## O que provaram

1. `gap1` supera o baseline `2L_32-64` (8.401.635 parametros, 88,30%) com 0,5%
   dos parametros. Os 8,39 M da camada densa eram capacidade desperdicada.
2. O teto de 120 epocas estava CORTANDO o treino: no fold 3 o melhor resultado
   caiu na ultima epoca e o bloco final ainda ganhava +1,06 p.p.

## Por que esta arquivado

O regime mudou: lr 4.5e-5 -> 4.5e-4 e teto 120 -> 1000 (na pratica, so para por
estagnacao). Manter estes folds em resultados/ faria o busca_hp pula-los na
retomada e misturar dois regimes no mesmo kfold_fold_results.json.

    python campanha/analisa_folds.py --raiz campanha/medicoes/2026-08-22_reducao_lr4.5e-5_teto120 --replay

## QAM_reducao — history CORROMPIDO, nao usar

O unico fold (`gap1`, fold 0, best 50,89%) tem **185 entradas de history para um
teto de 120**: as epocas 1-65 do checkpoint mais um treino completo 1-120,
com 65 epocas repetidas.

Causa: o bloco de retomada do `train_one_fold` atribuia direto nas variaveis, e
o `except` so zerava `ep_inicial`. Quando a restauracao do RNG falhava, o
`history` restaurado sobrevivia e o treino recomecava por cima dele.

Corrigido em 22/08/2026: a retomada passou a ser atomica -- nada e comprometido
antes de tudo dar certo, e o `except` limpa TODO o estado.

O numero de 50,89% (chance do QAM = 20%) ainda indica que a `gap1` aprende no
grupo dificil, mas o history nao serve para analise de plato.
