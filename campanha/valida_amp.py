"""Fase 0, parte C — o AMP muda o comportamento do colapso?

O AMP rendeu 1,54x-1,99x, mas muda a numerica (fp16 tem menos alcance). Este
projeto inteiro gira em torno de um colapso detectado comparando a loss de treino
com ln(C) a uma tolerancia de 0,02. Adotar AMP numa campanha de ~140 h sem
verificar seria imprudente.

Referencia conhecida (fp32, QAM 4L, lr=9e-5, subset 16k/4k, medido em 19/08/2026):
    seed 0 -> sem escape,        best 20,05% @ep13
    seed 1 -> escape em ep 16,2, best 24,77% @ep34   (vivo)
    seed 2 -> escape em ep 10,8, best 21,43% @ep30

Se o AMP reproduzir esse padrao, e seguro. Se mudar quem escapa ou quando, nao e.

NOTA: nomes globais com sufixo _VA de proposito. Rodar varios scripts no mesmo
kernel do Colab colide globais (`log`, `LOG`) e uma thread antiga passa a escrever
no log da nova — aconteceu com prep_all.py nesta sessao.
"""
import json, math, random, threading, time, traceback
import numpy as np, h5py, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

LOG_VA = "/content/valida_amp.log"
OUT_VA = "/content/valida_amp.json"
H5_VA = "/content/QAM_subset.hdf5"

N_TRAIN_VA, N_VAL_VA, BATCH_VA, DROPOUT_VA = 16000, 4000, 64, 0.5
LR_VA, EPOCHS_VA = 9e-5, 60
LR_PAT_VA, EARLY_VA, LR_MIN_VA = 5, 15, 1e-9          # iguais aos da referencia
FBASE_VA, FFLOOR_VA, MINDELTA_VA = 0.5, 0.01, 0.05
PROBE_VA = 50
SEEDS_VA = [0, 1, 2]

ARCH_VA = [{"out_channels": 32,  "kernel_size": 11, "pool": True},
           {"out_channels": 64,  "kernel_size": 7,  "pool": True},
           {"out_channels": 128, "kernel_size": 5,  "pool": True},
           {"out_channels": 256, "kernel_size": 3,  "pool": True}]

REFERENCIA = {0: {"escape": None, "best": 20.05},
              1: {"escape": 16.2, "best": 24.77},
              2: {"escape": 10.8, "best": 21.43}}

DEV_VA = torch.device("cuda")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def log_va(m):
    with open(LOG_VA, "a") as f:
        f.write("%s  %s\n" % (time.strftime("%H:%M:%S"), m))
    print(m, flush=True)


class FlexCNN_VA(nn.Module):
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


def carrega_va(idx):
    with h5py.File(H5_VA, "r") as f:
        X = np.asarray(f["X"][np.sort(idx), :], dtype=np.float32)
        Y = f["Y"][np.sort(idx)]
    y = np.argmax(Y, axis=1) if Y.ndim > 1 else Y
    return TensorDataset(torch.from_numpy(X).permute(0, 2, 1).contiguous(),
                         torch.from_numpy(np.asarray(y)).long())


def treina_va(usa_amp, seed, ds_tr, ds_vl, C):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    g = torch.Generator(); g.manual_seed(seed)
    tr = DataLoader(ds_tr, batch_size=BATCH_VA, shuffle=True, generator=g)
    vl = DataLoader(ds_vl, batch_size=BATCH_VA, shuffle=False)
    model = FlexCNN_VA(C, ARCH_VA, dropout=DROPOUT_VA).to(DEV_VA)
    crit = nn.CrossEntropyLoss()
    lr = LR_VA
    opt = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=usa_amp)
    chance, lnC = 100.0 / C, math.log(C)
    ref = chance
    best, best_ep, no_imp, pat, consec = 0.0, 0, 0, 0, 0
    esc, hist = None, []
    t0 = time.time()

    for ep in range(1, EPOCHS_VA + 1):
        model.train()
        L = corr = tot = 0
        wl, wa = [], []
        for bi, (xb, yb) in enumerate(tr, 1):
            xb, yb = xb.to(DEV_VA, non_blocking=True), yb.to(DEV_VA, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=usa_amp):
                out = model(xb)
                loss = crit(out, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            l = loss.item()
            pred = out.argmax(1)
            L += l; corr += (pred == yb).sum().item(); tot += yb.size(0)
            wl.append(l); wa.append(100.0 * (pred == yb).float().mean().item())
            if len(wl) > PROBE_VA:
                wl.pop(0); wa.pop(0)
            if esc is None and bi % PROBE_VA == 0 and len(wl) == PROBE_VA:
                ml, ma = sum(wl) / PROBE_VA, sum(wa) / PROBE_VA
                if (ma > chance * 1.15) and ((lnC - ml) > 0.02):
                    esc = round(ep - 1 + bi / len(tr), 3)
        trl, tra = L / len(tr), 100.0 * corr / tot

        model.eval()
        L = corr = tot = 0
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(DEV_VA), yb.to(DEV_VA)
                with torch.cuda.amp.autocast(enabled=usa_amp):
                    out = model(xb)
                    L += crit(out, yb).item()
                corr += (out.argmax(1) == yb).sum().item()
                tot += yb.size(0)
        vll, vla = L / len(vl), 100.0 * corr / tot

        hist.append({"ep": ep, "lr": lr, "trl": round(trl, 4), "tra": round(tra, 2),
                     "vll": round(vll, 4), "vla": round(vla, 2)})
        if vla > best:
            best, best_ep = vla, ep
        if vla > ref + MINDELTA_VA:
            ref = vla; pat = 0; consec = 0; no_imp = 0
        else:
            pat += 1; no_imp += 1
        if pat >= LR_PAT_VA and lr > LR_MIN_VA:
            f = max(FBASE_VA ** (consec + 1), FFLOOR_VA)
            lr = max(lr * f, LR_MIN_VA)
            for pg in opt.param_groups:
                pg["lr"] = lr
            consec += 1; pat = 0
        if no_imp >= EARLY_VA:
            break

    return {"amp": usa_amp, "seed": seed, "best_vla": round(best, 2),
            "best_ep": best_ep, "escape_ep": esc, "epochs_run": len(hist),
            "dead": (best <= chance * 1.15), "segundos": round(time.time() - t0, 1),
            "hist": hist}


def work_va():
    try:
        with h5py.File(H5_VA, "r") as f:
            C = int(f.attrs.get("num_classes", f["Y"].shape[1]))
            Y = f["Y"][:]
        y = np.argmax(Y, axis=1) if Y.ndim > 1 else Y
        rng = np.random.RandomState(42)
        ptr, pvl = N_TRAIN_VA // C, N_VAL_VA // C
        tr_idx, vl_idx = [], []
        for c in range(C):
            ic = np.where(y == c)[0]
            rng.shuffle(ic)
            tr_idx += ic[:ptr].tolist()
            vl_idx += ic[ptr:ptr + pvl].tolist()
        tr_idx, vl_idx = np.sort(tr_idx), np.sort(vl_idx)
        ds_tr, ds_vl = carrega_va(tr_idx), carrega_va(vl_idx)
        log_va("QAM 4L | C=%d chance=%.2f%% | lr=%.1e | fp32 vs AMP | ate %d epocas"
               % (C, 100.0 / C, LR_VA, EPOCHS_VA))

        res = {"fp32": [], "amp": []}
        for usa_amp, chave in ((False, "fp32"), (True, "amp")):
            for sd in SEEDS_VA:
                r = treina_va(usa_amp, sd, ds_tr, ds_vl, C)
                res[chave].append(r)
                log_va("  %-4s seed %d | best=%6.2f%% @ep%-2d | escape=%-6s | %2d eps | dead=%-5s | %.0fs"
                       % (chave, sd, r["best_vla"], r["best_ep"], str(r["escape_ep"]),
                          r["epochs_run"], r["dead"], r["segundos"]))
                json.dump(res, open(OUT_VA, "w"), indent=1)

        log_va("=" * 72)
        log_va("%-6s %-22s %-22s %s" % ("seed", "REFERENCIA (fp32 ant.)", "fp32 agora", "AMP"))
        ok = True
        for sd in SEEDS_VA:
            f32 = next(r for r in res["fp32"] if r["seed"] == sd)
            amp = next(r for r in res["amp"] if r["seed"] == sd)
            rf = REFERENCIA[sd]
            log_va("%-6d esc=%-6s best=%-6.2f  esc=%-6s best=%-6.2f  esc=%-6s best=%-6.2f"
                   % (sd, str(rf["escape"]), rf["best"], str(f32["escape_ep"]),
                      f32["best_vla"], str(amp["escape_ep"]), amp["best_vla"]))
            # criterio: AMP tem de concordar com fp32 sobre QUEM escapa
            if (amp["escape_ep"] is None) != (f32["escape_ep"] is None):
                ok = False
            if amp["dead"] != f32["dead"]:
                ok = False
        log_va("")
        log_va("VEREDITO: %s" % ("AMP PRESERVA o comportamento do colapso — seguro adotar"
                                 if ok else
                                 "AMP MUDA quem escapa/morre — NAO adotar sem investigar"))
        json.dump(res, open(OUT_VA, "w"), indent=1)
        log_va("VALIDA_AMP_PRONTO")
    except Exception:
        log_va("ERRO:\n" + traceback.format_exc())


open(LOG_VA, "w").close()
threading.Thread(target=work_va, daemon=True, name="valida_amp").start()
print("validacao numerica do AMP iniciada em background")
