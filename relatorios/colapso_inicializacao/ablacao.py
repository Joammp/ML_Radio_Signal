# -*- coding: utf-8 -*-
"""
Ablacao do colapso APSK — CPU, dado real, subconjunto.

Objetivo: descobrir QUAL ingrediente evita o colapso (loss == ln(C)).
Configs testadas, cada uma com N seeds:
  A baseline   — exatamente o codigo atual (sem norm, init default, sem warmup)
  B norm       — so acrescenta normalizacao por amostra
  C init+warm  — so acrescenta init Kaiming + warmup + clip (sem norm)
  D completo   — norm + init + warmup + clip
Restart NAO entra aqui: primeiro descobrimos se da pra PREVENIR; restart e o
seguro para o que sobrar.
"""
import json, math, os, random, sys, time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

HERE   = os.path.dirname(os.path.abspath(__file__))
H5     = os.path.join(HERE, "APSK.hdf5")

N_TRAIN   = 4000
N_VAL     = 1500
EPOCHS    = 4
BATCH     = 64
LR        = 1e-3
DROPOUT   = 0.5
SEEDS     = [0, 1, 2]
WARMUP    = 150
GRAD_CLIP = 5.0

ARCH = [  # 2L_32-64 — a mais rasa, que ja colapsava 5/5 folds no APSK
    {"out_channels": 32, "kernel_size": 7, "pool": True},
    {"out_channels": 64, "kernel_size": 5, "pool": True},
]

torch.set_num_threads(6)


# ──────────────────────────────────────────────────────────────────────────────
class H5Sub(Dataset):
    def __init__(self, path, idx, normalize):
        self.path, self.idx, self.normalize = path, np.asarray(idx), normalize
        self.f = None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self.f is None:
            self.f = h5py.File(self.path, "r")
            self.X, self.Y = self.f["X"], self.f["Y"]
        r = self.idx[i]
        x = np.asarray(self.X[r, :], dtype=np.float32)
        y = self.Y[r]
        if hasattr(y, "__len__"):
            y = int(np.argmax(y))
        if self.normalize:
            x = (x - x.mean()) / (x.std() + 1e-8)
        t = torch.from_numpy(x).float().permute(1, 0)
        return t, torch.tensor(int(y)).long()


class FlexCNN(nn.Module):
    def __init__(self, num_classes, arch, classifier=(512,), dropout=0.5,
                 in_channels=2, input_length=1024):
        super().__init__()
        layers, ch_in = [], in_channels
        for b in arch:
            ks = b["kernel_size"]
            layers += [nn.Conv1d(ch_in, b["out_channels"], ks, padding=ks // 2),
                       nn.BatchNorm1d(b["out_channels"]), nn.ReLU(inplace=True)]
            if b.get("pool", True):
                layers.append(nn.MaxPool1d(2))
            ch_in = b["out_channels"]
        self.features, self.flatten = nn.Sequential(*layers), nn.Flatten()
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, in_channels, input_length)).view(1, -1).size(1)
        head, prev = [], n_flat
        for u in classifier:
            head += [nn.Linear(prev, u), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = u
        head.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x):
        return self.classifier(self.flatten(self.features(x)))


def init_v2(model, head_gain=0.01):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias); last = m
    if last is not None:
        with torch.no_grad():
            last.weight.mul_(head_gain)
    return model


def run(cfg, seed, tr_idx, vl_idx, C):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    g = torch.Generator(); g.manual_seed(seed)

    tr = DataLoader(H5Sub(H5, tr_idx, cfg["norm"]), batch_size=BATCH,
                    shuffle=True, num_workers=0, generator=g)
    vl = DataLoader(H5Sub(H5, vl_idx, cfg["norm"]), batch_size=BATCH, shuffle=False)

    model = FlexCNN(C, ARCH, dropout=DROPOUT)
    if cfg["init"]:
        init_v2(model)

    crit = nn.CrossEntropyLoss()
    opt  = optim.Adam(model.parameters(), lr=LR)
    step = 0
    hist = []

    for ep in range(EPOCHS):
        model.train()
        L = corr = tot = 0
        for xb, yb in tr:
            if cfg["warm"] and step < WARMUP:
                w = LR * (0.01 + 0.99 * step / WARMUP)
                for pg in opt.param_groups:
                    pg["lr"] = w
            elif cfg["warm"]:
                for pg in opt.param_groups:
                    pg["lr"] = LR
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            if cfg["clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            step += 1
            L += loss.item(); corr += (out.argmax(1) == yb).sum().item(); tot += yb.size(0)
        trl, tra = L / len(tr), 100.0 * corr / tot

        model.eval()
        L = corr = tot = 0
        with torch.no_grad():
            for xb, yb in vl:
                out = model(xb); L += crit(out, yb).item()
                corr += (out.argmax(1) == yb).sum().item(); tot += yb.size(0)
        vll, vla = L / len(vl), 100.0 * corr / tot
        hist.append({"ep": ep + 1, "trl": round(trl, 4), "tra": round(tra, 2),
                     "vll": round(vll, 4), "vla": round(vla, 2)})
        print("      ep%d tr=%.4f/%.2f%% vl=%.4f/%.2f%%" % (ep + 1, trl, tra, vll, vla),
              flush=True)

    chance_acc, chance_loss = 100.0 / C, math.log(C)
    dead = (hist[-1]["tra"] <= chance_acc * 1.15
            and abs(hist[-1]["trl"] - chance_loss) <= 0.02)
    return {"best_vla": max(h["vla"] for h in hist), "dead": dead, "hist": hist}


def main():
    with h5py.File(H5, "r") as f:
        C = int(f.attrs.get("num_classes", f["Y"].shape[1]))
        N = f["X"].shape[0]
        Y = f["Y"][:]
        y = np.argmax(Y, axis=1) if Y.ndim > 1 else Y
    print("HDF5: N=%d  C=%d  chance=%.2f%%  ln(C)=%.4f" % (N, C, 100.0 / C, math.log(C)),
          flush=True)
    print("distribuicao de classes:", np.bincount(y, minlength=C).tolist(), flush=True)

    rng = np.random.RandomState(42)
    per_tr, per_vl = N_TRAIN // C, N_VAL // C
    tr_idx, vl_idx = [], []
    for c in range(C):
        ic = np.where(y == c)[0]
        rng.shuffle(ic)
        tr_idx += ic[:per_tr].tolist()
        vl_idx += ic[per_tr:per_tr + per_vl].tolist()
    tr_idx, vl_idx = np.sort(tr_idx), np.sort(vl_idx)
    print("subset estratificado: train=%d val=%d" % (len(tr_idx), len(vl_idx)), flush=True)

    CONFIGS = [
        ("A baseline  ", {"norm": False, "init": False, "warm": False, "clip": False}),
        ("B norm      ", {"norm": True,  "init": False, "warm": False, "clip": False}),
        ("C init+warm ", {"norm": False, "init": True,  "warm": True,  "clip": True}),
        ("D completo  ", {"norm": True,  "init": True,  "warm": True,  "clip": True}),
    ]

    results = {}
    for name, cfg in CONFIGS:
        print("\n=== %s %s ===" % (name, cfg), flush=True)
        rs = []
        for s in SEEDS:
            t0 = time.time()
            print("   seed %d" % s, flush=True)
            r = run(cfg, s, tr_idx, vl_idx, C)
            r["seed"] = s; r["secs"] = round(time.time() - t0)
            print("   -> best_val=%.2f%%  dead=%s  (%ds)" % (r["best_vla"], r["dead"], r["secs"]),
                  flush=True)
            rs.append(r)
        results[name.strip()] = rs
        nd = sum(1 for r in rs if r["dead"])
        print("   RESUMO %s: mortos %d/%d | melhor val %.2f%%"
              % (name.strip(), nd, len(rs), max(r["best_vla"] for r in rs)), flush=True)

    json.dump(results, open(os.path.join(HERE, "ablacao_result.json"), "w"), indent=1)
    print("\n" + "=" * 60, flush=True)
    print("%-14s %-10s %-12s" % ("CONFIG", "MORTOS", "MELHOR VAL"), flush=True)
    for name, _ in CONFIGS:
        rs = results[name.strip()]
        print("%-14s %d/%-8d %.2f%%" % (name.strip(), sum(1 for r in rs if r["dead"]),
                                        len(rs), max(r["best_vla"] for r in rs)), flush=True)


if __name__ == "__main__":
    main()
