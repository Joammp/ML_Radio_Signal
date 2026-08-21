# Auto-contido: espera o dataset, confere o fingerprint, e compara 2 estrategias de LR.
#   A "escapa_e_restaura": comeca em LR_LOW; ao SAIR do chance sobe para LR_STD (1e-3)
#                          e segue com a reducao escalonada atual.
#   B "baixo_e_reduz"    : comeca em LR_LOW e so reduz (escalonada), nunca sobe.
# Sem init/warm/clip nos dois bracos, para isolar a estrategia de LR.
import os, json, math, random, threading, traceback, time
import numpy as np, h5py, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

LOG3 = "/content/exp3.log"
OUT3 = "/content/qam_lrstrat_result.json"
H5 = "/content/QAM_subset.hdf5"
MAN = "/content/qam_manifest.json"
FP_ESPERADO = "847f44cb75a4c7b653744e34d30c039473e8daa805ec532ea46f233310ab9981"

N_TRAIN, N_VAL, BATCH, DROPOUT = 16000, 4000, 64, 0.5
LR_LOW, LR_STD, EPOCHS3 = 9e-5, 1e-3, 60
LR_PATIENCE, EARLY_STOP, LR_MIN = 5, 15, 1e-9
LR_FACTOR_BASE, LR_FACTOR_FLOOR, MIN_DELTA = 0.5, 0.01, 0.05
PROBE = 50

ARCH = [{"out_channels": 32,  "kernel_size": 11, "pool": True},
        {"out_channels": 64,  "kernel_size": 7,  "pool": True},
        {"out_channels": 128, "kernel_size": 5,  "pool": True},
        {"out_channels": 256, "kernel_size": 3,  "pool": True}]

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def log3(m):
    with open(LOG3, "a") as f:
        f.write("%s  %s\n" % (time.strftime("%H:%M:%S"), m))
    print(m, flush=True)


class FlexCNN(nn.Module):
    def __init__(self, num_classes, arch, classifier=(512,), dropout=0.5,
                 in_channels=2, input_length=1024):
        super().__init__()
        layers, ch = [], in_channels
        for b in arch:
            ks = b["kernel_size"]
            layers += [nn.Conv1d(ch, b["out_channels"], ks, padding=ks // 2),
                       nn.BatchNorm1d(b["out_channels"]), nn.ReLU(inplace=True)]
            if b.get("pool", True):
                layers.append(nn.MaxPool1d(2))
            ch = b["out_channels"]
        self.features = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        with torch.no_grad():
            n = self.features(torch.zeros(1, in_channels, input_length)).view(1, -1).size(1)
        head, prev = [], n
        for u in classifier:
            head += [nn.Linear(prev, u), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = u
        head.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x):
        return self.classifier(self.flatten(self.features(x)))


def load(idx):
    with h5py.File(H5, "r") as f:
        X = np.asarray(f["X"][np.sort(idx), :], dtype=np.float32)
        Y = f["Y"][np.sort(idx)]
    y = np.argmax(Y, axis=1) if Y.ndim > 1 else Y
    return TensorDataset(torch.from_numpy(X).permute(0, 2, 1).contiguous(),
                         torch.from_numpy(np.asarray(y)).long())


def escapou(wl, wa, chance, lnC):
    # espelha o criterio de morto: acc acima do chance E loss ABAIXO de ln(C)
    return (wa > chance * 1.15) and ((lnC - wl) > 0.02)


def run_strat(strategy, seed, ds_tr, ds_vl, C):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    g = torch.Generator(); g.manual_seed(seed)
    tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, generator=g)
    vl = DataLoader(ds_vl, batch_size=BATCH, shuffle=False)
    model = FlexCNN(C, ARCH, dropout=DROPOUT).to(DEV)
    crit = nn.CrossEntropyLoss()
    lr = LR_LOW
    opt = optim.Adam(model.parameters(), lr=lr)
    chance, lnC = 100.0 / C, math.log(C)
    ref = chance
    best, best_ep, no_imp, pat, consec = 0.0, 0, 0, 0, 0
    esc, raised, hist, drops = None, False, [], []

    for ep in range(1, EPOCHS3 + 1):
        model.train()
        L = corr = tot = 0
        wl, wa = [], []
        for bi, (xb, yb) in enumerate(tr, 1):
            xb, yb = xb.to(DEV, non_blocking=True), yb.to(DEV, non_blocking=True)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            l = loss.item()
            pred = out.argmax(1)
            L += l; corr += (pred == yb).sum().item(); tot += yb.size(0)
            wl.append(l); wa.append(100.0 * (pred == yb).float().mean().item())
            if len(wl) > PROBE:
                wl.pop(0); wa.pop(0)
            if esc is None and bi % PROBE == 0 and len(wl) == PROBE:
                if escapou(sum(wl) / PROBE, sum(wa) / PROBE, chance, lnC):
                    esc = round(ep - 1 + bi / len(tr), 3)
                    log3("      [saiu do chance em ep %.2f]" % esc)
                    if strategy == "A" and not raised:
                        lr = LR_STD
                        for pg in opt.param_groups:
                            pg["lr"] = lr
                        raised = True
                        log3("      [A: LR %.1e -> %.1e]" % (LR_LOW, LR_STD))
        trl, tra = L / len(tr), 100.0 * corr / tot

        model.eval()
        L = corr = tot = 0
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                out = model(xb)
                L += crit(out, yb).item()
                corr += (out.argmax(1) == yb).sum().item()
                tot += yb.size(0)
        vll, vla = L / len(vl), 100.0 * corr / tot

        hist.append({"ep": ep, "lr": lr, "trl": round(trl, 4), "tra": round(tra, 2),
                     "vll": round(vll, 4), "vla": round(vla, 2)})
        if vla > best:
            best, best_ep = vla, ep
        if vla > ref + MIN_DELTA:
            ref = vla; pat = 0; consec = 0; no_imp = 0
        else:
            pat += 1; no_imp += 1
        if pat >= LR_PATIENCE and lr > LR_MIN:
            f = max(LR_FACTOR_BASE ** (consec + 1), LR_FACTOR_FLOOR)
            lr = max(lr * f, LR_MIN)
            for pg in opt.param_groups:
                pg["lr"] = lr
            consec += 1; pat = 0
            drops.append({"ep": ep, "fator": round(f, 4), "lr": lr})
            log3("      ep%-2d reducao #%d fator=%.4f -> lr=%.3e" % (ep, consec, f, lr))
        if ep <= 3 or ep % 5 == 0:
            log3("      ep%-2d lr=%.2e tr=%.4f/%.2f%% vl=%.4f/%.2f%%"
                 % (ep, lr, trl, tra, vll, vla))
        if no_imp >= EARLY_STOP:
            log3("      early stop ep%d" % ep)
            break

    def eps_to(th):
        for h in hist:
            if h["vla"] >= th:
                return h["ep"]
        return None

    return {"seed": seed, "strategy": strategy, "best_vla": round(best, 2),
            "best_ep": best_ep, "escape_ep": esc, "epochs_run": len(hist),
            "final_lr": lr, "n_drops": len(drops), "dead": (best <= chance * 1.15),
            "drops": drops,
            "eps_to": {str(t): eps_to(t) for t in (25, 30, 35, 40, 50, 60)},
            "eps_to_90pct_best": (eps_to(0.9 * best) if best > chance * 1.15 else None),
            "hist": hist}


def work3():
    try:
        log3("aguardando o preparo do dataset...")
        for _ in range(240):
            if os.path.exists(MAN):
                break
            time.sleep(10)
        man = json.load(open(MAN))
        fp = man["fingerprint"]
        ok = (fp == FP_ESPERADO)
        log3("FINGERPRINT obtido : %s" % fp)
        log3("FINGERPRINT esperado: %s" % FP_ESPERADO)
        log3("DETERMINISMO: %s" % ("CONFERE" if ok else "DIVERGIU"))
        if not ok:
            log3("AVISO: dataset difere do da primeira geracao; resultados nao comparaveis.")

        with h5py.File(H5, "r") as f:
            C = int(f.attrs.get("num_classes", f["Y"].shape[1]))
            Y = f["Y"][:]
        y = np.argmax(Y, axis=1) if Y.ndim > 1 else Y
        rng = np.random.RandomState(42)
        ptr, pvl = N_TRAIN // C, N_VAL // C
        tr_idx, vl_idx = [], []
        for c in range(C):
            ic = np.where(y == c)[0]
            rng.shuffle(ic)
            tr_idx += ic[:ptr].tolist()
            vl_idx += ic[ptr:ptr + pvl].tolist()
        tr_idx, vl_idx = np.sort(tr_idx), np.sort(vl_idx)
        ds_tr, ds_vl = load(tr_idx), load(vl_idx)
        log3("QAM 4L | C=%d chance=%.2f%% | LR_LOW=%.1e LR_STD=%.1e | ate %d epocas | %s"
             % (C, 100.0 / C, LR_LOW, LR_STD, EPOCHS3, DEV))

        res = {"A_escapa_e_restaura": [], "B_baixo_e_reduz": []}
        for st, key in (("A", "A_escapa_e_restaura"), ("B", "B_baixo_e_reduz")):
            for sd in (0, 1, 2):
                log3("=== %s | seed %d ===" % (key, sd))
                t0 = time.time()
                r = run_strat(st, sd, ds_tr, ds_vl, C)
                r["secs"] = round(time.time() - t0, 1)
                log3("    -> best=%.2f%% @ep%d | escape=%s | %d eps | lr_final=%.2e | dead=%s (%.0fs)"
                     % (r["best_vla"], r["best_ep"], r["escape_ep"], r["epochs_run"],
                        r["final_lr"], r["dead"], r["secs"]))
                res[key].append(r)
                json.dump(res, open(OUT3, "w"), indent=1)

        log3("=" * 62)
        for key in res:
            rs = res[key]
            log3("%-20s best: %s | ep_best: %s | escape: %s" % (
                key,
                ", ".join("%.2f%%" % r["best_vla"] for r in rs),
                ", ".join(str(r["best_ep"]) for r in rs),
                ", ".join(str(r["escape_ep"]) for r in rs)))
        log3("PRONTO3")
    except Exception:
        log3("ERRO:\n" + traceback.format_exc())


open(LOG3, "w").close()
threading.Thread(target=work3, daemon=True).start()
print("comparacao agendada; roda assim que o dataset ficar pronto")
