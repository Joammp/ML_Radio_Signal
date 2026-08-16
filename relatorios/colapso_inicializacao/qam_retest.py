# -*- coding: utf-8 -*-
"""
Reteste do unico caso que o v2 perdeu: QAM, 4L_32-64-128-256, seed 0.

Antes (restart baixando LR a cada tentativa): 3 tentativas colapsadas,
status='dead', 370s. Agora as 3 primeiras tentativas mantem lr=1e-3 e trocam
so a seed. Mesma configuracao de dado do relatorio: 16000/4000, 12 epocas.
"""
import io, json, math, os, sys, time

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("V2_DIR", HERE))
sys.path.insert(0, HERE)

import train_one_fold_v2 as v2
from ablacao import H5Sub, FlexCNN

_real = v2.tqdm
v2.tqdm = lambda it, **kw: _real(it, **{**kw, "disable": True})

TOKEN = os.environ.get("DRIVE_TOKEN", os.path.join(HERE, "token.json"))
ARCH = [
    {"out_channels": 32,  "kernel_size": 11, "pool": True},
    {"out_channels": 64,  "kernel_size": 7,  "pool": True},
    {"out_channels": 128, "kernel_size": 5,  "pool": True},
    {"out_channels": 256, "kernel_size": 3,  "pool": True},
]
N_TRAIN, N_VAL, BATCH, EPOCHS, LR, SEED = 16000, 4000, 64, 12, 1e-3, 0

torch.set_num_threads(6)
creds = Credentials.from_authorized_user_file(TOKEN, ["https://www.googleapis.com/auth/drive"])
drive = build("drive", "v3", credentials=creds)

fs = drive.files().list(q="name contains 'QAM_subset' and trashed=false",
                        fields="files(id,name,size)").execute()["files"]
f = [x for x in fs if x["name"].startswith("QAM_subset")][0]
path = os.path.join(HERE, "QAM.hdf5")
print("baixando (%.0f MB)..." % (int(f["size"]) / 1e6), flush=True)
t0 = time.time()
with io.FileIO(path, "wb") as fh:
    dl = MediaIoBaseDownload(fh, drive.files().get_media(fileId=f["id"]),
                             chunksize=64 * 1024 * 1024)
    done = False
    while not done:
        _, done = dl.next_chunk()
print("baixado em %ds" % (time.time() - t0), flush=True)

try:
    with h5py.File(path, "r") as fh:
        C = int(fh.attrs.get("num_classes", fh["Y"].shape[1]))
        Y = fh["Y"][:]
    y = np.argmax(Y, axis=1) if Y.ndim > 1 else Y
    rng = np.random.RandomState(42)
    tr_idx, vl_idx = [], []
    for c in range(C):
        ic = np.where(y == c)[0]; rng.shuffle(ic)
        tr_idx += ic[:N_TRAIN // C].tolist()
        vl_idx += ic[N_TRAIN // C:N_TRAIN // C + N_VAL // C].tolist()
    tr_idx, vl_idx = np.sort(tr_idx), np.sort(vl_idx)
    print("QAM C=%d chance=%.2f%% ln(C)=%.4f train=%d val=%d"
          % (C, 100.0 / C, math.log(C), len(tr_idx), len(vl_idx)), flush=True)
    print("RESTART_SEED_ONLY=%d  MAX_RESTARTS=%d" % (v2.RESTART_SEED_ONLY, 4), flush=True)

    g = torch.Generator(); g.manual_seed(SEED)
    tr = DataLoader(H5Sub(path, tr_idx, True), batch_size=BATCH, shuffle=True, generator=g)
    vl = DataLoader(H5Sub(path, vl_idx, True), batch_size=BATCH, shuffle=False)

    t0 = time.time()
    acc, hist, info = v2.train_one_fold(
        model_fn=lambda: FlexCNN(C, ARCH, dropout=0.5),
        tr_loader=tr, vl_loader=vl, device=torch.device("cpu"), num_classes=C,
        lr=LR, epochs=EPOCHS, lr_patience=5, early_stop=15, lr_min=1e-9,
        fold_seed=SEED, max_restarts=4,
    )
    print("\nRESULTADO acc=%.2f%% info=%s epocas=%d tempo=%ds"
          % (acc, info, len(hist), time.time() - t0), flush=True)
    print("ANTES: dead=True, val=0.00, 3 tentativas, 370s", flush=True)
    print("VEREDITO: %s" % ("RECUPEROU" if info["status"] == "ok" else "AINDA FALHA"),
          flush=True)
    json.dump({"acc": acc, "info": info, "hist": hist},
              open(os.path.join(HERE, "qam_retest.json"), "w"), indent=1)
finally:
    if os.path.exists(path):
        os.remove(path)
        print("(hdf5 apagado)", flush=True)
