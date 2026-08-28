# Resultados anteriores a unificacao das condicoes de treino (27/08/2026)

Todos os folds aqui foram produzidos com

    fold_seed = SEED + arch_idx * 100 + fold_i

ou seja, **cada candidata usava um conjunto proprio de sementes**: gap1 via
142-146, gap8_head128 via 242-246, gap4 via 342-346, e assim por diante.

## Por que isso invalida a ordenacao fina

Este projeto gira em torno de um colapso que depende da INICIALIZACAO. Ligar a
semente ao indice da candidata confunde arquitetura com semente.

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

O colapso da gap8_head128 no QAM (4 de 5 folds em 20,00%, a chance) tambem fica
ambiguo: so ela viu as sementes 242-246, e o colapso e sabidamente dependente
da semente. Nao da para dizer se a arquitetura e fragil ou se o conjunto de
sementes era ruim.

## O que CONTINUA valido

- Modelos reduzidos superam a CNN de referencia por margens de 10 a 34 p.p. --
  muito acima de qualquer efeito de semente.
- head64 (unica sem pooling) fica ~4 p.p. abaixo das que usam pooling, apesar
  de ter 23x mais parametros que a gap1. O mecanismo e o pooling, nao o tamanho.
- 6pools lidera por 1,01 p.p. com desvio de 0,03 -- acima da margem de ruido,
  ainda que com um unico conjunto de sementes.

## Segunda variavel nao controlada

A GPU. Os folds rodaram em L4 ou T4 conforme a vaga livre, e o formato antigo
nao registrava qual. Medimos em 22/08/2026 que L4 e T4 nao reproduzem o mesmo
resultado com a mesma semente. A partir de 27/08 o `fold_result` grava `gpu`,
`torch`, `cuda` e `tf32`, e o busca_hp avisa quando a GPU muda no meio de um
conjunto.
