"""Escolha do LR inicial da campanha — sessao unica no QAM 4L.

Contexto. O LR e o unico hiperparametro que decide entre rede viva e rede morta
neste projeto. O que ja se sabe, de `experimentos_lr/README.md`:

    1e-3   -> morto em 15/15 folds. Tocar nele mata em qualquer forma testada
              (comecar, rampar ate, voltar depois do escape, passar e reduzir).
    9e-5   -> 2/3 seeds escaparam. E o default atual do busca_hp.py.
    4,5e-5 -> valor em que o escape foi de fato observado, apos reducao de plato.

Ou seja: o default de 9e-5 nunca foi comparado contra o valor onde o escape
acontece. Esta e a medicao que falta, e ela decide o LR de ~240 folds.

DESENHO — LR FIXO, SEM SCHEDULER (default).

O exp 3 registrou que "o scheduler de plato trabalha contra o escape": enquanto
colapsada nao ha melhora, a paciencia estoura e o LR despenca (no seed 0, para
1,4e-6 ate a ep15). Como os escapes conhecidos ocorreram em ep 10,8 e ep 16,2, e
a campanha usa LR_PATIENCE=8, o scheduler agiria ANTES do escape e contaminaria
a atribuicao: nao daria para dizer se o LR inicial foi bom ou se a reducao salvou.

Com LR fixo, o resultado e atribuivel ao LR inicial — que e exatamente o que se
quer escolher. Use --sched para a variante realista, depois de escolher.

TF32 desligado: a L4 (Ada) tem TF32 e o PyTorch liga cudnn.allow_tf32 por padrao.
A T4 (Turing) nao tem. Manter desligado preserva a comparabilidade com toda a
referencia do projeto.

NOTA: globais com sufixo _LR de proposito. Rodar varios scripts no mesmo kernel
do Colab colide globais (`log`, `LOG`) e uma thread antiga passa a escrever no
log da nova — aconteceu com prep_all.py.

Uso:
    colab exec -s campanha -f campanha/escolhe_lr.py
    colab exec -s campanha -f campanha/escolhe_lr.py --lrs 9e-5 4.5e-5 --seeds 0 1 2 3 4
"""
import argparse, json, math, os, random, shlex, sys, threading, time, traceback
import numpy as np, h5py, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

_p = argparse.ArgumentParser(description="Escolhe o LR inicial no QAM 4L")
_p.add_argument("--lrs", nargs="+", type=float, default=[1.8e-4, 9e-5, 4.5e-5, 2.2e-5],
                help="escada de LR a testar (default: %(default)s)")
_p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                help="seeds por LR (default: %(default)s)")
_p.add_argument("--epochs", type=int, default=60,
                help="epocas por run (default: %(default)d)")
_p.add_argument("--sched", action="store_true",
                help="liga a reducao de plato (default: LR fixo)")
_p.add_argument("--device", default="cuda",
                help="cuda, cuda:0, cpu (default: %(default)s)")
_p.add_argument("--base-dir", default="/content",
                help="raiz local dos artefatos (default: %(default)s)")


def _argv_lr():
    """De onde vem os parametros.

    Sob `colab exec` o script roda DENTRO do kernel IPython: sys.argv e o do
    colab_kernel_launcher (-f kernel-xxx.json), nao o que se digitou. E o
    `colab exec` 0.6.0 sequer aceita args extras -- rejeita com "No such
    option". Medido em 21/08/2026; o uso documentado no README nunca funcionou.

    Saida: variavel de ambiente, que persiste entre execs no mesmo kernel.
    Antes de rodar este script, num exec separado:
        import os; os.environ["ESCOLHE_LR_ARGS"] = "--lrs 9e-5 --sched"
    """
    raw = os.environ.get("ESCOLHE_LR_ARGS")
    if raw is not None:
        return shlex.split(raw)
    argv = sys.argv[1:]
    if any("kernel" in a and a.endswith(".json") for a in argv):
        return []                       # argv do kernel launcher: descarta
    return argv


_A_LR = _p.parse_args(_argv_lr())

LOG_LR = os.path.join(_A_LR.base_dir, "escolhe_lr.log")
OUT_LR = os.path.join(_A_LR.base_dir, "escolhe_lr.json")
H5_LR = os.path.join(_A_LR.base_dir, "QAM_subset.hdf5")
MAN_LR = os.path.join(_A_LR.base_dir, "grupos_manifest.json")

# identicos aos da referencia de 19/08 — nao mexer sem invalidar a comparacao
N_TRAIN_LR, N_VAL_LR, BATCH_LR, DROPOUT_LR = 16000, 4000, 64, 0.5
PROBE_LR = 50
LR_PAT_LR, LR_MIN_LR = 5, 1e-9
FBASE_LR, FFLOOR_LR, MINDELTA_LR = 0.5, 0.01, 0.05

ARCH_LR = [{"out_channels": 32, "kernel_size": 11, "pool": True},
           {"out_channels": 64, "kernel_size": 7, "pool": True},
           {"out_channels": 128, "kernel_size": 5, "pool": True},
           {"out_channels": 256, "kernel_size": 3, "pool": True}]

# referencia fp32/T4 de 19/08, lr=9e-5 — controle de sanidade da VM nova
REF_LR = {0: {"escape": None, "best": 20.05},
          1: {"escape": 16.2, "best": 24.77},
          2: {"escape": 10.8, "best": 21.43}}
FP_QAM_LR = "847f44cb75a4c7b653744e34d30c039473e8daa805ec532ea46f233310ab9981"

DEV_LR = torch.device(_A_LR.device)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def log_lr(m):
    with open(LOG_LR, "a") as f:
        f.write("%s  %s\n" % (time.strftime("%H:%M:%S"), m))
    print(m, flush=True)


class FlexCNN_LR(nn.Module):
    def __init__(self, num_classes, arch, classifier=(512,), dropout=0.5,
                 in_channels=2, input_length=1024):
        super().__init__()
        layers, ch = [], in_channels
        for b in arch:
            ks = b["kernel_size"]
            layers += [nn.Conv1d(ch, b["out_channels"], ks, padding=ks // 2),
                       nn.BatchNorm1d(b["out_channels"]), nn.ReLU()]
            if b.get("pool"):
                layers.append(nn.MaxPool1d(2))
            ch = b["out_channels"]
        self.features = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        with torch.no_grad():
            n = self.features(torch.zeros(1, in_channels, input_length))
            n = self.flatten(n).size(1)
        head = []
        for h in classifier:
            head += [nn.Linear(n, h), nn.ReLU(), nn.Dropout(dropout)]
            n = h
        head.append(nn.Linear(n, num_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x):
        return self.classifier(self.flatten(self.features(x)))


def carrega_lr(idx):
    with h5py.File(H5_LR, "r") as f:
        X = np.asarray(f["X"][np.sort(idx), :], dtype=np.float32)
        Y = f["Y"][np.sort(idx)]
    y = np.argmax(Y, axis=1) if Y.ndim > 1 else Y
    return TensorDataset(torch.from_numpy(X).permute(0, 2, 1).contiguous(),
                         torch.from_numpy(np.asarray(y)).long())


def treina_lr(lr0, seed, ds_tr, ds_vl, C):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    g = torch.Generator(); g.manual_seed(seed)
    tr = DataLoader(ds_tr, batch_size=BATCH_LR, shuffle=True, generator=g)
    vl = DataLoader(ds_vl, batch_size=BATCH_LR, shuffle=False)
    model = FlexCNN_LR(C, ARCH_LR, dropout=DROPOUT_LR).to(DEV_LR)
    crit = nn.CrossEntropyLoss()
    lr = lr0
    opt = optim.Adam(model.parameters(), lr=lr)
    chance, lnC = 100.0 / C, math.log(C)
    ref = chance
    best, best_ep, pat, consec = 0.0, 0, 0, 0
    esc, hist = None, []
    t0 = time.time()

    for ep in range(1, _A_LR.epochs + 1):
        model.train()
        L = corr = tot = 0
        wl, wa = [], []
        for bi, (xb, yb) in enumerate(tr, 1):
            xb, yb = xb.to(DEV_LR, non_blocking=True), yb.to(DEV_LR, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            l = loss.item()
            pred = out.argmax(1)
            L += l; corr += (pred == yb).sum().item(); tot += yb.size(0)
            wl.append(l); wa.append(100.0 * (pred == yb).float().mean().item())
            if len(wl) > PROBE_LR:
                wl.pop(0); wa.pop(0)
            # mesmo detector de escape da referencia: media movel de 50 batches,
            # loss > 0,02 ABAIXO de ln(C) e acuracia 15% acima da chance
            if esc is None and bi % PROBE_LR == 0 and len(wl) == PROBE_LR:
                ml, ma = sum(wl) / PROBE_LR, sum(wa) / PROBE_LR
                if (ma > chance * 1.15) and ((lnC - ml) > 0.02):
                    esc = round(ep - 1 + bi / len(tr), 3)
        trl, tra = L / len(tr), 100.0 * corr / tot

        model.eval()
        L = corr = tot = 0
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(DEV_LR), yb.to(DEV_LR)
                out = model(xb)
                L += crit(out, yb).item()
                corr += (out.argmax(1) == yb).sum().item()
                tot += yb.size(0)
        vll, vla = L / len(vl), 100.0 * corr / tot

        hist.append({"ep": ep, "lr": lr, "trl": round(trl, 4), "tra": round(tra, 2),
                     "vll": round(vll, 4), "vla": round(vla, 2)})
        if vla > best:
            best, best_ep = vla, ep
        if _A_LR.sched:
            if vla > ref + MINDELTA_LR:
                ref = vla; pat = 0; consec = 0
            else:
                pat += 1
            if pat >= LR_PAT_LR and lr > LR_MIN_LR:
                lr = max(lr * max(FBASE_LR ** (consec + 1), FFLOOR_LR), LR_MIN_LR)
                for pg in opt.param_groups:
                    pg["lr"] = lr
                consec += 1; pat = 0

    return {"lr0": lr0, "seed": seed, "best_vla": round(best, 2), "best_ep": best_ep,
            "escape_ep": esc, "dead": (best <= chance * 1.15),
            "sched": bool(_A_LR.sched), "segundos": round(time.time() - t0, 1),
            "hist": hist}


def work_lr():
    try:
        # retomada: se a VM caiu, nao refaz o que ja terminou
        res = []
        if os.path.exists(OUT_LR):
            try:
                prev = json.load(open(OUT_LR))
                res = prev["runs"] if isinstance(prev, dict) else prev
                log_lr("retomando: %d runs ja concluidos em %s" % (len(res), OUT_LR))
            except Exception:
                res = []
        # a chave inclui o modo do scheduler: um run com --sched NAO conta
        # como feito para uma rodada de LR fixo, e vice-versa.
        feitos = {(round(r["lr0"], 12), r["seed"], bool(r.get("sched", False)))
                  for r in res}

        if os.path.exists(MAN_LR):
            man = json.load(open(MAN_LR))
            fp = man.get("QAM", {}).get("fingerprint")
            ok = "OK" if fp == FP_QAM_LR else "DIVERGE - dado nao e o da referencia!"
            log_lr("fingerprint QAM: %s...  [%s]" % (str(fp)[:16], ok))

        with h5py.File(H5_LR, "r") as f:
            C = int(f.attrs.get("num_classes", f["Y"].shape[1]))
            Y = f["Y"][:]
        y = np.argmax(Y, axis=1) if Y.ndim > 1 else Y
        rng = np.random.RandomState(42)          # mesmo split da referencia
        ptr, pvl = N_TRAIN_LR // C, N_VAL_LR // C
        tr_idx, vl_idx = [], []
        for c in range(C):
            ic = np.where(y == c)[0]
            rng.shuffle(ic)
            tr_idx += ic[:ptr].tolist()
            vl_idx += ic[ptr:ptr + pvl].tolist()
        ds_tr, ds_vl = carrega_lr(np.sort(tr_idx)), carrega_lr(np.sort(vl_idx))
        chance = 100.0 / C

        log_lr("QAM 4L | C=%d chance=%.2f%% | scheduler=%s | TF32=off | %d epocas"
               % (C, chance, "ON" if _A_LR.sched else "OFF (LR fixo)", _A_LR.epochs))
        log_lr("escada de LR: %s | seeds: %s | total %d runs"
               % (_A_LR.lrs, _A_LR.seeds, len(_A_LR.lrs) * len(_A_LR.seeds)))
        log_lr("-" * 78)

        for lr0 in _A_LR.lrs:
            for sd in _A_LR.seeds:
                if (round(lr0, 12), sd, bool(_A_LR.sched)) in feitos:
                    log_lr("  lr=%-8.2g seed %d | ja feito, pulando" % (lr0, sd))
                    continue
                r = treina_lr(lr0, sd, ds_tr, ds_vl, C)
                res.append(r)
                json.dump(res, open(OUT_LR, "w"), indent=1)
                log_lr("  lr=%-8.2g seed %d | best=%6.2f%% @ep%-3d | escape=%-7s | %-5s | %.0fs"
                       % (lr0, sd, r["best_vla"], r["best_ep"], str(r["escape_ep"]),
                          "MORTO" if r["dead"] else "vivo", r["segundos"]))

        # ── consolidacao ──────────────────────────────────────────────────────
        log_lr("=" * 78)
        log_lr("%-10s %-9s %-9s %-10s %-10s %s"
               % ("lr", "escapes", "vivos", "best med", "best max", "escape medio"))
        placar = []
        for lr0 in _A_LR.lrs:
            rs = [r for r in res if round(r["lr0"], 12) == round(lr0, 12)
                  and bool(r.get("sched", False)) == bool(_A_LR.sched)]
            if not rs:
                continue
            n = len(rs)
            esc = [r["escape_ep"] for r in rs if r["escape_ep"] is not None]
            vivos = sum(0 if r["dead"] else 1 for r in rs)
            bmed = sum(r["best_vla"] for r in rs) / n
            bmax = max(r["best_vla"] for r in rs)
            placar.append({"lr": lr0, "n": n, "escapes": len(esc), "vivos": vivos,
                           "best_med": round(bmed, 2), "best_max": bmax,
                           "escape_med": round(sum(esc) / len(esc), 2) if esc else None})
            log_lr("%-10.2g %-9s %-9s %-10.2f %-10.2f %s"
                   % (lr0, "%d/%d" % (len(esc), n), "%d/%d" % (vivos, n), bmed, bmax,
                      ("%.1f" % (sum(esc) / len(esc))) if esc else "-"))

        # criterio: sobreviver primeiro, acuracia depois. Chegar mais alto so
        # importa entre LRs que mantenham a rede viva.
        if placar:
            melhor = sorted(placar, key=lambda p: (-p["vivos"], -p["escapes"],
                                                   -p["best_med"]))[0]
            log_lr("")
            log_lr("MELHOR LR: %.2g  (%d/%d vivos, %d/%d escapes, best medio %.2f%%)"
                   % (melhor["lr"], melhor["vivos"], melhor["n"],
                      melhor["escapes"], melhor["n"], melhor["best_med"]))
            if melhor["vivos"] < melhor["n"]:
                log_lr("AVISO: nem todas as seeds sobreviveram nem no melhor LR.")
                log_lr("       Testar um degrau abaixo antes de lancar 240 folds.")
            if melhor["lr"] in (max(_A_LR.lrs), min(_A_LR.lrs)) and len(_A_LR.lrs) > 1:
                log_lr("AVISO: o vencedor esta na BORDA da escada - o otimo pode")
                log_lr("       estar fora do intervalo testado. Estender a escada.")

        # controle de sanidade contra a referencia de 19/08 (so vale para 9e-5)
        r95 = [r for r in res if abs(r["lr0"] - 9e-5) < 1e-12 and r["seed"] in REF_LR
               and bool(r.get("sched", False)) == bool(_A_LR.sched)]
        if r95:
            log_lr("")
            log_lr("controle vs referencia fp32/T4 de 19/08 (lr=9e-5):")
            for r in sorted(r95, key=lambda r: r["seed"]):
                rf = REF_LR[r["seed"]]
                log_lr("  seed %d | ref: esc=%-6s best=%-6.2f | agora: esc=%-7s best=%.2f"
                       % (r["seed"], str(rf["escape"]), rf["best"],
                          str(r["escape_ep"]), r["best_vla"]))
            if not _A_LR.sched:
                log_lr("  (LR fixo aqui vs scheduler la - divergencia esperada apos o plato)")

        json.dump({"runs": res, "placar": placar}, open(OUT_LR, "w"), indent=1)
        log_lr("ESCOLHE_LR_PRONTO")
    except Exception:
        log_lr("ERRO:\n" + traceback.format_exc())


# TRAVA CONTRA INSTANCIA DUPLA.
# Em 21/08/2026 duas instancias subiram no mesmo kernel do Colab e rodaram
# concorrentes: mesmo log, mesmo JSON, e -- pior -- mesmo RNG global, porque
# vivem no mesmo processo. As chamadas de torch.manual_seed de uma interleavam
# com as da outra, e a mesma (lr, seed) devolveu resultados diferentes. Os 12
# runs daquela sessao foram perdidos. E a colisao de globais que o README ja
# avisava, na sua forma mais cara.
_viva_lr = [t for t in threading.enumerate()
            if t.name == "escolhe_lr" and t.is_alive()]
if _viva_lr:
    print("RECUSADO: ja existe uma instancia de escolhe_lr viva neste kernel.")
    print("          Espere ela terminar (ou reinicie o kernel) antes de relancar.")
    print("          Rodar duas em paralelo corrompe o RNG e invalida os resultados.")
else:
    open(LOG_LR, "w").close()
    threading.Thread(target=work_lr, daemon=True, name="escolhe_lr").start()
    print("escolha de LR iniciada em background")
