"""Gera busca_hp.py a partir do notebook, mantendo o notebook como fonte da verdade.

Fonte: branch `atualiza-busca-hp-estratificacao` do repo ML_Radio_Signal, que ja
tem STRAT_BY_SNR=True, MIN_DELTA e a reducao escalonada do LR. NAO usar a versao
de Downloads, que e anterior a tudo isso.

A celula de busca (18) e quase autocontida: depende de apenas 5 nomes externos
(H5PyDataset, drive_push, drive_pull, path, modulation_classes_path). Este script
fornece esses 5, aplica as substituicoes de hiperparametro e troca o `run_all()`
final por um guard de __main__.

Cada substituicao e VERIFICADA: se o notebook mudar e um padrao nao casar, o
script falha em vez de gerar codigo silenciosamente errado.

Uso:
    python gerar_busca_hp.py [--notebook CAMINHO] [--saida busca_hp.py]
"""
import argparse
import json
import os
import sys
import urllib.request

RAW = ("https://raw.githubusercontent.com/Joammp/ML_Radio_Signal/"
       "atualiza-busca-hp-estratificacao/BUSCA_HP_Classes_1cap(3).ipynb")

CEL_DATASET = 13   # H5PyDataset
CEL_BUSCA = 18     # a celula de busca inteira

# (descricao, de, para) — todas obrigatorias
SUBS = [
    ("learning rate inicial",
     "LEARNING_RATE    = 1e-3",
     "LEARNING_RATE    = _ARGS.lr"),
    ("paciencia do LR",
     "LR_PATIENCE      = 5",
     "LR_PATIENCE      = _ARGS.lr_patience"),
    ("early stop",
     "EARLY_STOP       = 15",
     "EARLY_STOP       = _ARGS.early_stop"),
    ("grupos alvo",
     'GRUPOS_ALVO = ["APSK","QAM"]',
     "GRUPOS_ALVO = _ARGS.grupos"),
    ("selecao de device",
     'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
     "device = torch.device(_ARGS.device)"),
    ("execucao automatica no import",
     "\nrun_all()",
     '\nif __name__ == "__main__":\n    run_all()'),
]

CABECALHO = '''# -*- coding: utf-8 -*-
"""GERADO POR gerar_busca_hp.py — NAO EDITAR A MAO.

Fonte: {fonte}
Celulas {cel_ds} (H5PyDataset) e {cel_busca} (busca), com hiperparametros
parametrizados por linha de comando.

Diferencas em relacao ao notebook, e por que:
  LEARNING_RATE  1e-3 -> {lr:g}   1e-3 colapsa o QAM (15/15 folds medidos em 19/08/2026)
  LR_PATIENCE    5    -> {pat}      pedido do usuario
  EARLY_STOP     15   -> {es}     acompanha a paciencia maior
  drive_push/pull     -> no-op    o upload ao Drive e feito pelo app LOCAL, para
                                  o token de escopo `drive` nunca sair da maquina

RESSALVA: lr={lr:g} nao e garantia. Em 3 seeds testadas no QAM 4L, 1 ainda colapsou
nesse valor, e o escape observado ocorreu de fato em 4.5e-5, apos a primeira
reducao de plato. Se o QAM voltar a colapsar em massa, 4.5e-5 e o proximo valor.
"""
import argparse
import sys

_p = argparse.ArgumentParser(description="Busca de HP de um grupo de modulacao")
_p.add_argument("--grupos", nargs="+", required=True,
                help="grupos a processar, ex.: QAM  (ASK PSK APSK QAM)")
_p.add_argument("--lr", type=float, default={lr:g},
                help="learning rate inicial (default: %(default)g)")
_p.add_argument("--lr-patience", type=int, default={pat},
                help="epocas sem melhora ate reduzir o LR (default: %(default)d)")
_p.add_argument("--early-stop", type=int, default={es},
                help="epocas sem melhora, com LR no piso, ate encerrar (default: %(default)d)")
_p.add_argument("--device", default="cuda",
                help="cuda, cuda:0, cuda:1, cpu (default: %(default)s)")
_p.add_argument("--base-dir", default="/content",
                help="raiz local dos artefatos (default: %(default)s)")
_ARGS = _p.parse_args()

# ── Fonte de dados: Kaggle publico, sem credencial ────────────────────────────
import kagglehub
path = kagglehub.dataset_download("pinxau1000/radioml2018")
modulation_classes_path = path + "/classes-fixed.json"
print("[busca_hp] dataset em %s" % path, flush=True)

# ── Drive desativado NA VM ────────────────────────────────────────────────────
# O notebook original fazia drive_push de dentro da VM, o que exige subir um
# token de escopo `drive` completo para uma maquina remota. Aqui os artefatos
# ficam apenas em disco local; quem sincroniza com o Drive e o app local.
DRIVE_FROM_VM = False


def drive_push(local_path):
    return False


def drive_pull(local_path):
    return False


'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", default=None,
                    help="caminho local do .ipynb; se omitido, baixa do branch")
    ap.add_argument("--saida", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "busca_hp.py"))
    ap.add_argument("--lr", type=float, default=9e-5)
    ap.add_argument("--lr-patience", type=int, default=8)
    ap.add_argument("--early-stop", type=int, default=20)
    a = ap.parse_args()

    if a.notebook:
        fonte = a.notebook
        nb = json.load(open(a.notebook, encoding="utf-8"))
    else:
        fonte = RAW
        print("baixando notebook do branch...")
        with urllib.request.urlopen(RAW) as r:
            nb = json.loads(r.read().decode("utf-8"))

    cells = nb["cells"]
    if len(cells) <= CEL_BUSCA:
        sys.exit("notebook tem %d celulas; esperava mais de %d" % (len(cells), CEL_BUSCA))

    ds = "".join(cells[CEL_DATASET]["source"])
    busca = "".join(cells[CEL_BUSCA]["source"])

    if "class H5PyDataset" not in ds:
        sys.exit("celula %d nao contem H5PyDataset — o notebook mudou de forma" % CEL_DATASET)
    if "def run_all" not in busca:
        sys.exit("celula %d nao contem run_all — o notebook mudou de forma" % CEL_BUSCA)

    aplicadas = []
    for desc, de, para in SUBS:
        n = busca.count(de)
        if n != 1:
            sys.exit("substituicao '%s' casou %d vezes (esperava 1).\n"
                     "  padrao: %r\n"
                     "  o notebook mudou; ajuste SUBS em gerar_busca_hp.py" % (desc, n, de))
        busca = busca.replace(de, para, 1)
        aplicadas.append(desc)

    cabecalho = CABECALHO.format(fonte=fonte, cel_ds=CEL_DATASET, cel_busca=CEL_BUSCA,
                                 lr=a.lr, pat=a.lr_patience, es=a.early_stop)
    saida = cabecalho + ds.rstrip() + "\n\n\n" + busca

    with open(a.saida, "w", encoding="utf-8") as f:
        f.write(saida)

    print("gerado: %s  (%d linhas)" % (a.saida, saida.count("\n") + 1))
    for d in aplicadas:
        print("   aplicada: %s" % d)

    import ast
    try:
        ast.parse(saida)
        print("   sintaxe OK")
    except SyntaxError as e:
        sys.exit("   SINTAXE INVALIDA na linha %s: %s" % (e.lineno, e.msg))


if __name__ == "__main__":
    main()
