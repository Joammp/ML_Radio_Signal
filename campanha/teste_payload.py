"""Payload sintetico para validar a orquestracao do painel, sem treinar nada.

Imita a FORMA do busca_hp.py, que e o que o painel precisa suportar:
  - roda SINCRONO sob `if __name__ == "__main__"` (nao sobe thread propria)
  - so faz print, sem arquivo de log (quem captura e o runner_vm)
  - grava kfold_fold_results.json incrementalmente, um fold por vez, no mesmo
    caminho que o busca_hp usa -- e dai que o painel le o progresso
  - le parametros da env BUSCA_HP_ARGS, com --grupos obrigatorio

Nao substitui um teste do busca_hp real; serve para exercitar sessao, upload,
runner, log ao vivo, contagem de folds e espelho local em ~90 s em vez de das
40-90 h que um fold de verdade levaria.
"""
import argparse
import json
import os
import shlex
import sys
import time

_p = argparse.ArgumentParser(description="Payload de teste do painel")
_p.add_argument("--grupos", nargs="+", required=True)
_p.add_argument("--lr", type=float, default=4.5e-5)
_p.add_argument("--folds", type=int, default=5)
_p.add_argument("--seg-por-fold", type=float, default=15.0)


def _argv_tp():
    raw = os.environ.get("BUSCA_HP_ARGS")
    if raw is not None:
        return shlex.split(raw)
    argv = sys.argv[1:]
    if any("kernel" in a and a.endswith(".json") for a in argv):
        return []
    return argv


_A = _p.parse_args(_argv_tp())
DRIVE_BASE = "/content/drive_cache/radioml_v2"   # casa com RAIZ_VM do painel


def run_all():
    for modelo in _A.grupos:
        d = os.path.join(DRIVE_BASE, modelo)
        os.makedirs(d, exist_ok=True)
        alvo = os.path.join(d, "kfold_fold_results.json")
        print("[teste] GRUPO %s | lr=%g | %d folds simulados"
              % (modelo, _A.lr, _A.folds), flush=True)
        feitos = []
        for k in range(1, _A.folds + 1):
            time.sleep(_A.seg_por_fold)
            feitos.append({"fold": k, "label": "sintetico_%d" % k,
                           "val_acc": 20.0 + k, "complete": True,
                           "ts": time.strftime("%H:%M:%S")})
            with open(alvo, "w") as f:
                json.dump(feitos, f, indent=1)
            print("  fold %d/%d concluido -> val_acc %.2f%%"
                  % (k, _A.folds, feitos[-1]["val_acc"]), flush=True)
        print("[teste] %s concluido" % modelo, flush=True)


if __name__ == "__main__":
    run_all()
