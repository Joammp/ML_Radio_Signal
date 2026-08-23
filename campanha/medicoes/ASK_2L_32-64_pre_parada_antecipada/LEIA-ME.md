# ASK / 2L_32-64 / fold 0 — medicao anterior a parada antecipada

Rodado em 21/08/2026 com a configuracao **antiga** de parada:

    lr = 4.5e-05   lr_patience = 8   early_stop = 20
    lr_min = 1e-09   min_delta = 0.05      <-- valores antigos

Resultado: best_val_acc = 88,30%, **120 epocas** (bateu no teto; o early stop
nunca disparou), 41,1 min de L4.

## Por que esta arquivado, e nao apagado

Foi tirado de `campanha/resultados/` para nao contaminar a campanha: o
`busca_hp.py` retoma lendo `kfold_fold_results.json` e pularia este fold em vez
de refaze-lo com os parametros novos, e o resultado ficaria comparado com folds
rodados sob outra regra de parada.

Mas o dado em si continua valendo: e a **base de evidencia** de
`MIN_DELTA=0.1` e `LR_MIN=1e-7`. O plato foi alcancado na ep. 27 e as 93 epocas
seguintes renderam 0,25 p.p., contra ruido p90 de 0,28 p.p. entre epocas.

Para reproduzir a analise que motivou a mudanca:

    python campanha/analisa_folds.py --raiz campanha/medicoes --replay

Nao e lido por `analisa_folds.py`, `sincroniza_drive.py` nem pelo painel, que
so olham para `campanha/resultados/`.
