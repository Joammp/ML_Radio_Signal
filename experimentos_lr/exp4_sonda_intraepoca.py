# Sonda intra-epoca na PRIMEIRA epoca: a cada x batches, se nao houver aprendizado,
# reduz o LR. A partir da 2a epoca segue a progressao normal (plato escalonado).
# Varredura sobre x por halving sucessivo (busca binaria): 1/2, 1/4, 1/8, 1/16 da epoca.
import os, json, math, random, threading, traceback, time
import numpy as np, h5py, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

LOG4 = "/content/exp4.log"
OUT4 = "/content/qam_probe_result.json"
H5 = "/content/QAM_subset.hdf5"
GATE = "/content/exp3.log"          # espera o experimento anterior terminar

N_TRAIN, N_VAL, BATCH, DROPOUT = 16000, 4000, 64, 0.5
LR0, EPOCHS4 = 1e-3, 60
LR_PATIENCE, EARLY_STOP, LR_MIN = 5, 15, 1e-9
LR_FACTOR_BASE, LR_FACTOR_FLOOR, MIN_DELTA = 0.5, 0.01, 0.05

# Sonda da epoca 1
PROBE_FACTOR = 0.5      # "um pouco": cada disparo corta o LR pela metade
PROBE_FLOOR = 1e-6      # piso durante a epoca 1
# x = batches entre sondas. 250 batches/epoca -> 1/2, 1/4, 1/8, 1/16 da epoca.
X_VALUES = [125, 62, 31, 15]
SEEDS = [0, 1, 2]

ARCH = [{"out_channels": 32,  "kernel_size": 11, "pool": True},
        {"out_channels": 64,  "kernel_size": 7,  "pool": True},
        {"out_channels": 128, "kernel_size": 5,  "pool": True},
        {"out_channels": 256, "kernel_size": 3,  "pool": True}]

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def log4(m):
    with open(LOG4, "a") as f:
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


def aprendendo(win_loss, win_acc, chance, lnC):
    """Mesmo criterio usado como 'saiu do chance': acc acima do chance E loss abaixo de ln(C)."""
    return (win_acc > chance * 1.15) and ((lnC - win_loss) > 0.02)


def run_probe(x, seed, ds_tr, ds_vl, C):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    g = torch.Generator(); g.manual_seed(seed)
    tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, generator=g)
    vl = DataLoader(ds_vl, batch_size=BATCH, shuffle=False)
    model = FlexCNN(C, ARCH, dropout=DROPOUT).to(DEV)
    crit = nn.CrossEntropyLoss()
    lr = LR0
    opt = optim.Adam(model.parameters(), lr=lr)
    chance, lnC = 100.0 / C, math.log(C)
    ref = chance
    best, best_ep, no_imp, pat, consec = 0.0, 0, 0, 0, 0
    esc, hist, drops, probe_events = None, [], [], []
    lr_fim_ep1 = None

    for ep in range(1, EPOCHS4 + 1):
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
            if len(wl) > x:
                wl.pop(0); wa.pop(0)

            if bi % x == 0 and len(wl) == x:
                ml, ma = sum(wl) / x, sum(wa) / x
                viva = aprendendo(ml, ma, chance, lnC)
                if esc is None and viva:
                    esc = round(ep - 1 + bi / len(tr), 3)
                    log4("        [saiu do chance em ep %.2f]" % esc)
                # sonda de reducao SO na primeira epoca, e so se nao houver aprendizado
                if ep == 1 and not viva and lr > PROBE_FLOOR:
                    novo = max(lr * PROBE_FACTOR, PROBE_FLOOR)
                    probe_events.append({"batch": bi, "de": lr, "para": novo,
                                         "loss": round(ml, 4), "acc": round(ma, 2)})
                    lr = novo
                    for pg in opt.param_groups:
                        pg["lr"] = lr
        if ep == 1:
            lr_fim_ep1 = lr
            log4("      ep1: %d reducoes de sonda -> lr=%.3e" % (len(probe_events), lr))

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
        # progressao normal a partir da 2a epoca
        if ep >= 2 and pat >= LR_PATIENCE and lr > LR_MIN:
            f = max(LR_FACTOR_BASE ** (consec + 1), LR_FACTOR_FLOOR)
            lr = max(lr * f, LR_MIN)
            for pg in opt.param_groups:
                pg["lr"] = lr
            consec += 1; pat = 0
            drops.append({"ep": ep, "fator": round(f, 4), "lr": lr})
        if ep <= 2 or ep % 10 == 0:
            log4("      ep%-2d lr=%.2e tr=%.4f/%.2f%% vl=%.4f/%.2f%%"
                 % (ep, lr, trl, tra, vll, vla))
        if no_imp >= EARLY_STOP:
            log4("      early stop ep%d" % ep)
            break

    def eps_to(th):
        for h in hist:
            if h["vla"] >= th:
                return h["ep"]
        return None

    return {"x": x, "seed": seed, "best_vla": round(best, 2), "best_ep": best_ep,
            "escape_ep": esc, "epochs_run": len(hist), "final_lr": lr,
            "lr_fim_ep1": lr_fim_ep1, "n_probe_reducoes": len(probe_events),
            "dead": (best <= chance * 1.15), "n_drops_platô": len(drops),
            "probe_events": probe_events,
            "eps_to": {str(t): eps_to(t) for t in (25, 30, 35, 40, 50)},
            "hist": hist}


def work4():
    try:
        log4("aguardando o experimento anterior terminar...")
        for _ in range(360):
            if os.path.exists(GATE) and "PRONTO3" in open(GATE).read():
                break
            time.sleep(10)
        log4("GPU livre; iniciando varredura de x")

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
        nb = N_TRAIN // BATCH
        log4("QAM 4L | C=%d chance=%.2f%% | LR0=%.1e | %d batches/epoca | %s"
             % (C, 100.0 / C, LR0, nb, DEV))
        log4("x testados: %s (= 1/%s da epoca)"
             % (X_VALUES, [round(nb / v) for v in X_VALUES]))

        res = {}
        for x in X_VALUES:
            key = "x=%d" % x
            res[key] = []
            for sd in SEEDS:
                log4("=== %s (1/%d da epoca) | seed %d ===" % (key, round(nb / x), sd))
                t0 = time.time()
                r = run_probe(x, sd, ds_tr, ds_vl, C)
                r["secs"] = round(time.time() - t0, 1)
                log4("    -> best=%.2f%% @ep%d | escape=%s | lr_fim_ep1=%.2e (%d red.) | %d eps | dead=%s (%.0fs)"
                     % (r["best_vla"], r["best_ep"], r["escape_ep"], r["lr_fim_ep1"],
                        r["n_probe_reducoes"], r["epochs_run"], r["dead"], r["secs"]))
                res[key].append(r)
                json.dump(res, open(OUT4, "w"), indent=1)

        log4("=" * 66)
        for key in res:
            rs = res[key]
            nd = sum(1 for r in rs if r["dead"])
            log4("%-8s mortos %d/%d | best: %s | lr_fim_ep1: %s" % (
                key, nd, len(rs),
                ", ".join("%.2f%%" % r["best_vla"] for r in rs),
                ", ".join("%.1e" % r["lr_fim_ep1"] for r in rs)))
        log4("PRONTO4")
    except Exception:
        log4("ERRO:\n" + traceback.format_exc())


open(LOG4, "w").close()
threading.Thread(target=work4, daemon=True).start()
print("varredura de x enfileirada; comeca quando o experimento anterior terminar")
