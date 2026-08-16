# -*- coding: utf-8 -*-
"""
QAM com mais dado — separa "metodo nao resolve" de "subset pequeno demais".

Na validacao anterior o QAM ficou no chance level (20.3-22.5%) nas 4 rodadas do
v2, com subset de 4000/1500 = ~50 amostras por par (classe, SNR). Aqui: 16000/
4000 = ~200 por par, e 12 epocas em vez de 5.

Imprime a curva por epoca para dar pra ver SE esta subindo, nao so onde parou.
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

TOKEN = os.environ.get("DRIVE_TOKEN", os.path.join(HERE, "token.json"))
ARCHS = [
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
SEEDS = [0, 1]
N_TRAIN, N_VAL, BATCH, EPOCHS, LR = 16000, 4000, 64, 12, 1e-3

torch.set_num_threads(6)
creds = Credentials.from_authorized_user_file(TOKEN, ["https://www.googleapis.com/auth/drive"])
drive = build("drive", "v3", credentials=creds)


def baixar():
    fs = drive.files().list(q="name contains 'QAM_subset' and trashed=false",
                            fields="files(id,name,size)").execute()["files"]
    f = [x for x in fs if x["name"].startswith("QAM_subset")][0]
    out = os.path.join(HERE, "QAM.hdf5")
    print("baixando %s (%.0f MB)..." % (f["name"], int(f["size"]) / 1e6), flush=True)
    t0 = time.time()
    with io.FileIO(out, "wb") as fh:
        dl = MediaIoBaseDownload(fh, drive.files().get_media(fileId=f["id"]),
                                 chunksize=64 * 1024 * 1024)
        done = False
        while not done:
            _, done = dl.next_chunk()
    print("baixado em %ds" % (time.time() - t0), flush=True)
    return out


def run_baseline(path, tr_idx, vl_idx, C, arch, seed, tag):
    torch.manual_seed(seed); np.random.seed(seed)
    g = torch.Generator(); g.manual_seed(seed)
    tr = DataLoader(H5Sub(path, tr_idx, False), batch_size=BATCH, shuffle=True, generator=g)
    vl = DataLoader(H5Sub(path, vl_idx, False), batch_size=BATCH, shuffle=False)
    model = FlexCNN(C, arch, dropout=0.5)
    crit, opt = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=LR)
    best, curva, trl, tra = 0.0, [], None, None
    for ep in range(EPOCHS):
        model.train(); L = c = t = 0
        for xb, yb in tr:
            opt.zero_grad(); out = model(xb); loss = crit(out, yb)
            loss.backward(); opt.step()
            L += loss.item(); c += (out.argmax(1) == yb).sum().item(); t += yb.size(0)
        trl, tra = L / len(tr), 100.0 * c / t
        model.eval(); c = t = 0
        with torch.no_grad():
            for xb, yb in vl:
                out = model(xb); c += (out.argmax(1) == yb).sum().item(); t += yb.size(0)
        va = 100.0 * c / t
        best = max(best, va); curva.append(round(va, 2))
        print("    %s ep%02d trloss=%.4f vl=%.2f%%" % (tag, ep + 1, trl, va), flush=True)
    dead = (tra <= 100.0 / C * 1.15 and abs(trl - math.log(C)) <= 0.02)
    return best, dead, 0, curva


def run_v2(path, tr_idx, vl_idx, C, arch, seed, tag):
    g = torch.Generator(); g.manual_seed(seed)
    tr = DataLoader(H5Sub(path, tr_idx, True), batch_size=BATCH, shuffle=True, generator=g)
    vl = DataLoader(H5Sub(path, vl_idx, True), batch_size=BATCH, shuffle=False)
    acc, hist, info = v2.train_one_fold(
        model_fn=lambda: FlexCNN(C, arch, dropout=0.5),
        tr_loader=tr, vl_loader=vl, device=torch.device("cpu"), num_classes=C,
        lr=LR, epochs=EPOCHS, lr_patience=5, early_stop=15, lr_min=1e-9,
        fold_seed=seed, max_restarts=2,
    )
    curva = [h["val_acc"] for h in hist]
    for h in hist:
        print("    %s ep%02d trloss=%.4f vl=%.2f%%" % (tag, h["epoch"], h["train_loss"],
                                                       h["val_acc"]), flush=True)
    return acc, info["status"] == "dead", info["restarts"], curva


def main():
    path = baixar()
    try:
        with h5py.File(path, "r") as f:
            C = int(f.attrs.get("num_classes", f["Y"].shape[1]))
            Y = f["Y"][:]
        y = np.argmax(Y, axis=1) if Y.ndim > 1 else Y
        rng = np.random.RandomState(42)
        tr_idx, vl_idx = [], []
        for c in range(C):
            ic = np.where(y == c)[0]; rng.shuffle(ic)
            tr_idx += ic[:N_TRAIN // C].tolist()
            vl_idx += ic[N_TRAIN // C:N_TRAIN // C + N_VAL // C].tolist()
        tr_idx, vl_idx = np.sort(tr_idx), np.sort(vl_idx)
        print("QAM C=%d chance=%.2f%% ln(C)=%.4f train=%d val=%d epocas=%d"
              % (C, 100.0 / C, math.log(C), len(tr_idx), len(vl_idx), EPOCHS), flush=True)

        res = []
        for alabel, arch in ARCHS:
            for seed in SEEDS:
                for cfg, fn in (("baseline", run_baseline), ("v2", run_v2)):
                    tag = "%s/s%d/%s" % (alabel[:2], seed, cfg[:4])
                    t0 = time.time()
                    acc, dead, rs, curva = fn(path, tr_idx, vl_idx, C, arch, seed, tag)
                    r = {"arch": alabel, "seed": seed, "cfg": cfg, "best_val": round(acc, 2),
                         "dead": dead, "restarts": rs, "secs": round(time.time() - t0),
                         "curva": curva}
                    res.append(r)
                    print("  RES %-18s s%d %-8s val=%6.2f%% dead=%-5s rst=%d (%ds)"
                          % (alabel, seed, cfg, acc, dead, rs, r["secs"]), flush=True)
                    json.dump(res, open(os.path.join(HERE, "qam_maisdados.json"), "w"), indent=1)

        print("\n" + "=" * 62, flush=True)
        for alabel, _ in ARCHS:
            for cfg in ("baseline", "v2"):
                rs = [r for r in res if r["arch"] == alabel and r["cfg"] == cfg]
                if rs:
                    print("%-18s %-8s  val=%s  mortos=%d/%d"
                          % (alabel, cfg, "/".join("%.1f" % r["best_val"] for r in rs),
                             sum(r["dead"] for r in rs), len(rs)), flush=True)
        print("TOTAL v2 mortos: %d/%d"
              % (sum(r["dead"] for r in res if r["cfg"] == "v2"),
                 sum(1 for r in res if r["cfg"] == "v2")), flush=True)
    finally:
        if os.path.exists(path):
            os.remove(path)
            print("(hdf5 apagado)", flush=True)


if __name__ == "__main__":
    main()
