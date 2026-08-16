# -*- coding: utf-8 -*-
"""
Testa o ARQUIVO train_one_fold_v2.py de verdade (nao uma reimplementacao).

Cenario 1 — feliz : init v2 ativa. Espera status="ok", restarts=0.
Cenario 2 — morto : init v2 neutralizada (monkeypatch) para forcar o colapso.
                    Exercita detector -> restart -> desistencia.
                    Espera status="dead" e MUITO menos epocas que as 46 do
                    codigo antigo.
"""
import os, sys, time
import numpy as np, torch, h5py
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
# train_one_fold_v2.py fica nesta mesma pasta; permite override por env var
sys.path.insert(0, os.environ.get("V2_DIR", HERE))
sys.path.insert(0, HERE)

import train_one_fold_v2 as v2
from ablacao import H5Sub, FlexCNN, ARCH, H5

# Sonda mais curta: o subset tem so 62 batches por epoca (4000/64).
v2.DEAD_PROBE_BATCH = 40
# Sem barra de progresso no teste (polui o log capturado). Mantem o objeto tqdm
# real porque o codigo usa pbar.set_postfix()/close(); so desliga a renderizacao.
_real_tqdm = v2.tqdm
v2.tqdm = lambda it, **kw: _real_tqdm(it, **{**kw, "disable": True})

N_TRAIN, N_VAL, BATCH = 4000, 1500, 64

with h5py.File(H5, "r") as f:
    C = int(f.attrs.get("num_classes", f["Y"].shape[1]))
    Y = f["Y"][:]
y = np.argmax(Y, axis=1)

rng = np.random.RandomState(42)
tr_idx, vl_idx = [], []
for c in range(C):
    ic = np.where(y == c)[0]; rng.shuffle(ic)
    tr_idx += ic[:N_TRAIN // C].tolist()
    vl_idx += ic[N_TRAIN // C:N_TRAIN // C + N_VAL // C].tolist()
tr_idx, vl_idx = np.sort(tr_idx), np.sort(vl_idx)

tr = DataLoader(H5Sub(H5, tr_idx, True), batch_size=BATCH, shuffle=True)
vl = DataLoader(H5Sub(H5, vl_idx, True), batch_size=BATCH, shuffle=False)
dev = torch.device("cpu")
torch.set_num_threads(6)

mk = lambda: FlexCNN(C, ARCH, dropout=0.5)

common = dict(tr_loader=tr, vl_loader=vl, device=dev, num_classes=C,
              lr=1e-3, epochs=5, lr_patience=5, early_stop=15, lr_min=1e-9,
              fold_seed=0)

print("\n" + "#" * 70)
print("# CENARIO 1 — caminho feliz (init v2 ativa)")
print("#" * 70, flush=True)
t0 = time.time()
acc1, hist1, info1 = v2.train_one_fold(model_fn=mk, **common)
print(">>> acc=%.2f%% info=%s epocas=%d tempo=%ds"
      % (acc1, info1, len(hist1), time.time() - t0), flush=True)

print("\n" + "#" * 70)
print("# CENARIO 2 — colapso forcado (init v2 neutralizada)")
print("#" * 70, flush=True)
v2.init_weights = lambda m, head_gain=None: m      # volta a init default
v2.MAX_RESTARTS = 2                                # encurta o teste
t0 = time.time()
acc2, hist2, info2 = v2.train_one_fold(model_fn=mk, max_restarts=2, **common)
print(">>> acc=%.2f%% info=%s epocas_completas=%d tempo=%ds"
      % (acc2, info2, len(hist2), time.time() - t0), flush=True)

print("\n" + "=" * 70)
# C1: com a init v2 a rede tem que viver SEM precisar de restart.
ok1 = info1["status"] == "ok" and info1["restarts"] == 0 and acc1 > 100.0 / C * 1.2
# C2: com a init default ela colapsa; o exigido e que o detector DISPARE
# (restarts>=1) e que o custo total fique muito abaixo das 46 epocas que o
# codigo antigo gastava por fold morto. Recuperar e bonus, nao requisito.
ok2 = info2["restarts"] >= 1 and len(hist2) < 20
print("CENARIO 1 (vive sem restart): %s  status=%s restarts=%d acc=%.2f%%"
      % ("PASSOU" if ok1 else "FALHOU", info1["status"], info1["restarts"], acc1))
print("CENARIO 2 (detecta e reage) : %s  status=%s restarts=%d epocas=%d"
      % ("PASSOU" if ok2 else "FALHOU", info2["status"], info2["restarts"], len(hist2)))
print("=" * 70, flush=True)
sys.exit(0 if (ok1 and ok2) else 1)
