# Folds de 22/08/2026 — anteriores a parada por estagnacao absoluta

Rodados com `MIN_DELTA=0.10` e `LR_MIN=1e-07`, ja com a primeira correcao de
parada, mas ainda **sem** o criterio de estagnacao absoluta.

| Grupo | Epocas | best_val_acc | |
|---|---|---|---|
| ASK  |  89 | 88,29% | parou sozinho |
| QAM  |  77 | 27,17% | parou sozinho |
| APSK | 120 | 72,99% | **bateu no teto** |
| PSK  |  -  |   -    | nao concluiu nenhum fold |

## O que estes dados provaram

O early stop antigo depende do LR chegar ao piso, e o caminho ate la e zerado
por qualquer oscilacao da val_acc. Ruido p90 medido entre epocas:

    ASK 0,33 p.p.   QAM 0,28 p.p.   APSK 1,33 p.p.

Todos acima do MIN_DELTA de 0,10. No APSK o escalonamento nunca engatou (ficou
em x0,5), o LR parou em 3,5e-7 sem alcancar o piso de 1e-7, e o fold rodou as
120 epocas. Varrer MIN_DELTA nao resolve: o resultado e nao-monotonico, porque
mexer no limiar muda o proprio caminho do LR.

Dai veio a parada por estagnacao absoluta (`--stagnation-patience 25`,
`--stagnation-margin 0.25`), simulada sobre estes mesmos historicos:

    ASK  ep 51 (-43%)   APSK ep 92 (-23%)   QAM ep 47 (-39%)   perda max 0,13 p.p.

## Por que esta arquivado

Fora de `campanha/resultados/` para nao contaminar a campanha: o `busca_hp.py`
retoma lendo `kfold_fold_results.json` e pularia estes folds em vez de refaze-los
sob a regra nova.

Para reler a analise:

    python campanha/analisa_folds.py --raiz campanha/medicoes/2026-08-22_pre_parada_absoluta --replay
