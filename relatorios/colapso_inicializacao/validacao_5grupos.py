# -*- coding: utf-8 -*-
"""
Validacao estendida: 5 grupos x 2 arquiteturas x 2 seeds, baseline vs v2.

Baixa um HDF5 por vez do Drive, testa, apaga (disco apertado: ~12 GB livres
contra ~12.4 GB de arquivos).

  baseline = codigo atual do notebook: init default, sem warmup, sem clip,
             sem normalizacao por amostra
  v2       = train_one_fold_v2.py real: init Kaiming + warmup + clip +
             deteccao + restart, com normalizacao por amostra

Metrica: o fold termina MORTO (acc no chance e loss em ln(C))?
"""
import io, json, math, os, sys, time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

HERE = os.path.dirname(os.path.abspath(__file__))
# train_one_fold_v2.py fica nesta mesma pasta; permite override por env var
sys.path.insert(0, os.environ.get("V2_DIR", HERE))
sys.path.insert(0, HERE)

import train_one_fold_v2 as v2
from ablacao import H5Sub, FlexCNN

_real_tqdm = v2.tqdm
v2.tqdm = lambda it, **kw: _real_tqdm(it, **{**kw, "disable": True})
v2.DEAD_PROBE_BATCH = 40

TOKEN = os.environ.get("DRIVE_TOKEN", os.path.join(HERE, "token.json"))
GRUPOS  = ["ASK", "PSK", "APSK", "QAM", "AM"]
ARCHS   = [
    ("2L_32-64", [
        {"out_channels": 32, "kernel_size": 7, "pool": True},
        {"out_channels": 64, "kernel_size": 5, "pool": True},
    ]),
    ("4L_32-64-128-256", [
        {"out_channels": 32,  "kernel_size": 11, "pool": True},
        {"out_channels": 64,  "kernel_size": 7,  "pool": True},
        {"out_channels": 128, "kernel_size": 5,  "pool": True},
        {"out_channels": 256, "kernel_size": 3,  "pool": True},
    ]),
]
SEEDS   = [0, 1]
N_TRAIN, N_VAL, BATCH, EPOCHS, LR = 4000, 1500, 64, 5, 1e-3

torch.set_num_threads(6)
creds = Credentials.from_authorized_user_file(TOKEN, ["https://www.googleapis.com/auth/drive"])
drive = build("drive", "v3", credentials=creds)


def baixar(grupo):
    q = f"name contains '{grupo}_subset' and trashed=false"
    fs = drive.files().list(q=q, fields="files(id,name,size)").execute()["files"]
    fs = [f for f in fs if f["name"].startswith(grupo + "_subset")]
    if not fs:
        raise RuntimeError("HDF5 nao encontrado para " + grupo)
    f = fs[0]
    out = os.path.join(HERE, grupo + ".hdf5")
    print("  baixando %s (%.0f MB)..." % (f["name"], int(f["size"]) / 1e6), flush=True)
    t0 = time.time()
    with io.FileIO(out, "wb") as fh:
        dl = MediaIoBaseDownload(fh, drive.files().get_media(fileId=f["id"]),
                                 chunksize=64 * 1024 * 1024)
        done = False
        while not done:
            _, done = dl.next_chunk()
    print("  baixado em %ds" % (time.time() - t0), flush=True)
    return out


def splits(path):
    with h5py.File(path, "r") as f:
        C = int(f.attrs.get("num_classes", f["Y"].shape[1]))
        Y = f["Y"][:]
    y = np.argmax(Y, axis=1) if Y.ndim > 1 else Y
    rng = np.random.RandomState(42)
    tr, vl = [], []
    for c in range(C):
        ic = np.where(y == c)[0]
        rng.shuffle(ic)
        tr += ic[:N_TRAIN // C].tolist()
        vl += ic[N_TRAIN // C:N_TRAIN // C + N_VAL // C].tolist()
    return np.sort(tr), np.sort(vl), C


def loaders(path, tr_idx, vl_idx, norm, seed):
    g = torch.Generator(); g.manual_seed(seed)
    tr = DataLoader(H5Sub(path, tr_idx, norm), batch_size=BATCH, shuffle=True, generator=g)
    vl = DataLoader(H5Sub(path, vl_idx, norm), batch_size=BATCH, shuffle=False)
    return tr, vl


def run_baseline(path, tr_idx, vl_idx, C, arch, seed):
    """Replica o comportamento atual do notebook."""
    torch.manual_seed(seed); np.random.seed(seed)
    tr, vl = loaders(path, tr_idx, vl_idx, False, seed)
    model = FlexCNN(C, arch, dropout=0.5)
    crit, opt = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=LR)
    best, last_trl, last_tra = 0.0, None, None
    for _ in range(EPOCHS):
        model.train(); L = c = t = 0
        for xb, yb in tr:
            opt.zero_grad(); out = model(xb); loss = crit(out, yb)
            loss.backward(); opt.step()
            L += loss.item(); c += (out.argmax(1) == yb).sum().item(); t += yb.size(0)
        last_trl, last_tra = L / len(tr), 100.0 * c / t
        model.eval(); c = t = 0
        with torch.no_grad():
            for xb, yb in vl:
                out = model(xb); c += (out.argmax(1) == yb).sum().item(); t += yb.size(0)
        best = max(best, 100.0 * c / t)
    dead = (last_tra <= 100.0 / C * 1.15 and abs(last_trl - math.log(C)) <= 0.02)
    return best, dead, 0


def run_v2(path, tr_idx, vl_idx, C, arch, seed):
    tr, vl = loaders(path, tr_idx, vl_idx, True, seed)
    acc, hist, info = v2.train_one_fold(
        model_fn=lambda: FlexCNN(C, arch, dropout=0.5),
        tr_loader=tr, vl_loader=vl, device=torch.device("cpu"), num_classes=C,
        lr=LR, epochs=EPOCHS, lr_patience=5, early_stop=15, lr_min=1e-9,
        fold_seed=seed, max_restarts=2,
    )
    return acc, info["status"] == "dead", info["restarts"]


def main():
    todos = {}
    for grupo in GRUPOS:
        print("\n" + "=" * 66, flush=True)
        print("GRUPO %s" % grupo, flush=True)
        print("=" * 66, flush=True)
        path = baixar(grupo)
        try:
            tr_idx, vl_idx, C = splits(path)
            print("  C=%d  chance=%.2f%%  ln(C)=%.4f  train=%d val=%d"
                  % (C, 100.0 / C, math.log(C), len(tr_idx), len(vl_idx)), flush=True)
            for alabel, arch in ARCHS:
                for seed in SEEDS:
                    for cfg, fn in (("baseline", run_baseline), ("v2", run_v2)):
                        t0 = time.time()
                        acc, dead, rs = fn(path, tr_idx, vl_idx, C, arch, seed)
                        rec = {"grupo": grupo, "C": C, "arch": alabel, "seed": seed,
                               "cfg": cfg, "best_val": round(acc, 2), "dead": dead,
                               "restarts": rs, "secs": round(time.time() - t0)}
                        todos.setdefault(grupo, []).append(rec)
                        print("  RES %-5s %-18s s%d %-8s val=%6.2f%% dead=%-5s rst=%d (%ds)"
                              % (grupo, alabel, seed, cfg, acc, dead, rs, rec["secs"]),
                              flush=True)
                        json.dump(todos, open(os.path.join(HERE, "validacao5.json"), "w"),
                                  indent=1)
        finally:
            os.remove(path)
            print("  (hdf5 apagado)", flush=True)

    print("\n" + "=" * 66, flush=True)
    print("%-6s %-4s %-20s %-10s %-10s" % ("GRUPO", "C", "ARQ", "BASELINE", "V2"), flush=True)
    print("=" * 66, flush=True)
    for grupo in GRUPOS:
        for alabel, _ in ARCHS:
            b = [r for r in todos.get(grupo, []) if r["arch"] == alabel and r["cfg"] == "baseline"]
            v = [r for r in todos.get(grupo, []) if r["arch"] == alabel and r["cfg"] == "v2"]
            if not b:
                continue
            print("%-6s %-4d %-20s %d/%d morto  %d/%d morto   base=%.1f%% v2=%.1f%%"
                  % (grupo, b[0]["C"], alabel,
                     sum(r["dead"] for r in b), len(b),
                     sum(r["dead"] for r in v), len(v),
                     max(r["best_val"] for r in b), max(r["best_val"] for r in v)),
                  flush=True)
    nb = sum(r["dead"] for g in todos.values() for r in g if r["cfg"] == "baseline")
    nv = sum(r["dead"] for g in todos.values() for r in g if r["cfg"] == "v2")
    tb = sum(1 for g in todos.values() for r in g if r["cfg"] == "baseline")
    print("\nTOTAL: baseline %d/%d mortos | v2 %d/%d mortos" % (nb, tb, nv, tb), flush=True)


if __name__ == "__main__":
    main()
