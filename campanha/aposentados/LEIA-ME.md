# Aposentados

## gerar_busca_hp.py (aposentado em 28/08/2026)

Reconstruia o `busca_hp.py` a partir do notebook
`BUSCA_HP_Classes_1cap(3).ipynb` (branch `atualiza-busca-hp-estratificacao`),
aplicando uma lista de substituicoes com marcadores e falhando se alguma nao
casasse.

Funcionou enquanto o `busca_hp.py` era so o notebook com hiperparametros
parametrizados. Deixou de funcionar quando a busca de reducao de parametros foi
escrita direto no `busca_hp.py` e nao voltou para o gerador.

Diferenca medida em 28/08/2026, regenerando e comparando com o arquivo em uso.
O regenerado **perde**:

- a opcao `--modelo reducao` e a lista `ARQ_REDUCAO` (as 8 candidatas)
- `pool_saida` e o `AdaptiveAvgPool1d` no `FlexCNN` -- o mecanismo de reducao
- os argumentos `--drive-base` e `--epochs`; `EPOCHS_PER_FOLD` volta a 120 fixo
- `stagnation_patience` e `stagnation_margin` no `fold_result`

A assercao interna do gerador nao pegava isso: ela verifica que cada
substituicao encontra sua ancora, nao que a saida seja igual ao arquivo em uso.
Sao coisas diferentes.

**Nao rode este script.** Esta aqui como registro de como o `busca_hp.py`
nasceu, e das medicoes que justificaram cada divergencia em relacao ao
notebook original (elas estao nos comentarios do proprio gerador).

A fonte de verdade agora e `campanha/busca_hp.py`, editado a mao.
