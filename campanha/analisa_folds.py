"""Telemetria de plato dos folds da campanha -- roda local, so leitura.

Le os kfold_fold_results.json espelhados em campanha/resultados/<GRUPO>/ (mesmo
formato que save_fold_result grava) e responde duas perguntas:

  1. Quanto tempo de GPU cada fold gastou DEPOIS de ja ter estagnado?
  2. O MIN_DELTA configurado ainda esta acima do ruido da val_acc NAQUELE grupo
     e arquitetura?

A segunda importa porque MIN_DELTA=0.1 foi calibrado no ASK 2L_32-64 -- o grupo
mais facil e a rede menor. Se num PSK 5L o ruido entre epocas for maior que
isso, oscilacoes contam como melhora, a paciencia nunca estoura e o fold roda o
teto inteiro. Foi exatamente a falha diagnosticada em 21/08/2026.

USO
    python campanha/analisa_folds.py                    # todos os grupos
    python campanha/analisa_folds.py --grupos QAM PSK
    python campanha/analisa_folds.py --replay           # simula outros limiares
"""
import argparse
import json
import os
import statistics as st

AQUI = os.path.dirname(os.path.abspath(__file__))
RESULTADOS = os.path.join(AQUI, "resultados")

_p = argparse.ArgumentParser(description="Analisa o plato dos folds ja rodados")
_p.add_argument("--grupos", nargs="+", default=None,
                help="grupos a analisar (default: todos os encontrados)")
_p.add_argument("--raiz", default=RESULTADOS)
_p.add_argument("--tol", type=float, default=0.25,
                help="p.p. do melhor que definem 'plato alcancado' (default: %(default)g)")
_p.add_argument("--teto", type=int, default=120,
                help="EPOCHS_PER_FOLD, para marcar folds que bateram no teto")
_p.add_argument("--replay", action="store_true",
                help="simula outros (min_delta, lr_min) sobre os historicos salvos")
_p.add_argument("--min-delta", type=float, default=0.10,
                help="MIN_DELTA em uso, para contar resets espurios (default: %(default)g)")
_A = _p.parse_args()


def carrega():
    """Devolve {grupo: [fold_result, ...]} do que houver em disco."""
    fora = {}
    if not os.path.isdir(_A.raiz):
        return fora
    for g in sorted(os.listdir(_A.raiz)):
        if _A.grupos and g not in _A.grupos:
            continue
        caminho = os.path.join(_A.raiz, g, "kfold_fold_results.json")
        if not os.path.exists(caminho):
            continue
        try:
            d = json.load(open(caminho, encoding="utf-8"))
        except Exception as e:
            print("  [aviso] %s ilegivel: %s" % (caminho, e))
            continue
        if isinstance(d, list) and d and isinstance(d[0], dict) and "history" in d[0]:
            fora[g] = d
        elif isinstance(d, list):
            print("  [aviso] %s sem 'history' (%d itens) -- sintetico? ignorando"
                  % (g, len(d)))
    return fora


def metricas(r):
    """Plato, desperdicio e ruido de um fold."""
    h = r.get("history") or []
    if len(h) < 5:
        return None
    va = [e["val_acc"] for e in h]
    best = max(va)
    ep_plato = next(e["epoch"] for e in h if e["val_acc"] >= best - _A.tol)
    corte = max(ep_plato - 1, 0)
    plateau = va[corte:]
    deltas = [abs(va[i + 1] - va[i]) for i in range(corte, len(va) - 1)]
    seg_ep = (r.get("elapsed_s") or 0) / max(len(h), 1)
    lrs = sorted({e["lr"] for e in h}, reverse=True)
    ref, resets = -1e9, 0
    for e in h:
        if e["val_acc"] > ref + _A.min_delta:
            ref = e["val_acc"]
            if e["epoch"] > ep_plato:
                resets += 1              # recorde de ruido: zerou a paciencia
    return {
        "resets": resets,
        "epocas": len(h), "best": best, "ep_plato": ep_plato,
        "desperdicio": len(h) - ep_plato,
        "min_desperdicados": (len(h) - ep_plato) * seg_ep / 60.0,
        "seg_ep": seg_ep,
        "desvio": st.pstdev(plateau) if len(plateau) > 1 else 0.0,
        "p90": sorted(deltas)[int(.9 * len(deltas))] if deltas else 0.0,
        "n_lrs": len(lrs), "lr0": lrs[0] if lrs else None,
        "lr_fim": h[-1]["lr"], "no_teto": len(h) >= _A.teto,
    }


def replay(h, min_delta, lr_min, lr0, pat=8, early=20, base=0.5, floor=0.01):
    """Reproduz o laco de train_one_fold do busca_hp sobre um history salvo.

    Fiel de proposito ao detalhe que mais importa: epochs_no_improve zera A CADA
    REDUCAO de LR (busca_hp.py, dentro do bloco de reducao escalonada), nao so
    quando ha melhora. E isso que torna o early stop tao dificil de alcancar.
    """
    ref, sem, consec, lr, best = -1e9, 0, 0, lr0, -1e9
    for e in h:
        best = max(best, e["val_acc"])
        if e["val_acc"] > ref + min_delta:
            ref, sem, consec = e["val_acc"], 0, 0
        else:
            sem += 1
        if sem >= pat and lr > lr_min:
            lr = max(lr * max(base ** (consec + 1), floor), lr_min)
            consec += 1
            sem = 0
        elif sem >= early and lr <= lr_min * 1.001:
            return e["epoch"], best
    return None, best


def main():
    dados = carrega()
    if not dados:
        print("nada para analisar em %s" % _A.raiz)
        return

    tot_ep = tot_desp = tot_min = 0
    for g, folds in dados.items():
        print("\n=== %s ===" % g)
        print("%-22s %-5s %-8s %-7s %-9s %-11s %-8s %s"
              % ("arquitetura", "fold", "best", "epocas", "plato ep", "desperdicio",
                 "ruido p90", "LR"))
        for r in folds:
            m = metricas(r)
            if m is None:
                continue
            tot_ep += m["epocas"]; tot_desp += m["desperdicio"]; tot_min += m["min_desperdicados"]
            print("%-22s %-5s %-8.2f %-7s %-9d %-11s %-8.3f %.1e->%.1e (%d red)"
                  % (r.get("label", "?"), r.get("fold", "?"), m["best"],
                     "%d%s" % (m["epocas"], " TETO" if m["no_teto"] else ""),
                     m["ep_plato"],
                     "%d ep / %.0f min" % (m["desperdicio"], m["min_desperdicados"]),
                     m["p90"], m["lr0"] or 0, m["lr_fim"], m["n_lrs"] - 1))
            if m["resets"] and m["no_teto"]:
                print("   ATENCAO: %d recorde(s) de ruido apos o plato zeraram a"
                      " paciencia com MIN_DELTA=%g, e o fold bateu no teto."
                      % (m["resets"], _A.min_delta))
                print("            Suba MIN_DELTA acima do ruido deste grupo"
                      " (p90 medido: %.3f p.p.) e confirme com --replay." % m["p90"])
            elif m["resets"]:
                print("   nota: %d recorde(s) de ruido apos o plato, mas o fold parou"
                      " sozinho -- MIN_DELTA=%g esta dando conta aqui."
                      % (m["resets"], _A.min_delta))

            if _A.replay:
                print("   replay (min_delta / lr_min -> epoca de parada, best):")
                for md, lrm in ((0.05, 1e-9), (0.10, 1e-7), (0.20, 1e-6), (0.30, 1e-6)):
                    ep, best = replay(r["history"], md, lrm, r.get("lr", 4.5e-5))
                    print("     %.2f / %-8.0e -> %-12s best=%.2f"
                          % (md, lrm, ("ep %d (-%.0f%%)" % (ep, 100 * (m["epocas"] - ep) / m["epocas"]))
                             if ep else "nao para", best))

    if tot_ep:
        print("\n" + "=" * 78)
        print("TOTAL: %d epocas rodadas, %d apos o plato (%.0f%%), ~%.1f h de GPU"
              % (tot_ep, tot_desp, 100.0 * tot_desp / tot_ep, tot_min / 60.0))


if __name__ == "__main__":
    main()
