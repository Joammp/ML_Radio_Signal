# Baixa a fonte do Kaggle e constroi os subsets dos 5 grupos reais numa unica
# varredura, com a mesma logica deterministica do notebook (filtro puro, sem RNG).
import os, json, threading, traceback, hashlib, time
import numpy as np, h5py

LOG = "/content/prep.log"
DESIRED_SNRS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
BUILD_BATCH = 2048
GROUP_MAP = {
    "OOK": 0, "4ASK": 0, "8ASK": 0,
    "BPSK": 1, "QPSK": 1, "8PSK": 1, "16PSK": 1, "32PSK": 1, "GMSK": 1, "OQPSK": 1,
    "16APSK": 2, "32APSK": 2, "64APSK": 2, "128APSK": 2,
    "16QAM": 3, "32QAM": 3, "64QAM": 3, "128QAM": 3, "256QAM": 3,
    
    # AM e FM retirados: analogicas, e o FM tem classe unica.
}
GRUPOS = {0: "ASK", 1: "PSK", 2: "APSK", 3: "QAM"}   # so digitais
FP_QAM = "847f44cb75a4c7b653744e34d30c039473e8daa805ec532ea46f233310ab9981"


def log(m):
    with open(LOG, "a") as f:
        f.write("%s  %s\n" % (time.strftime("%H:%M:%S"), m))
    print(m, flush=True)


def fingerprint(path):
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        h.update(np.asarray(f["X"].shape).tobytes())
        h.update(f["Y"][:].tobytes())
        h.update(f["Z"][:].tobytes())
        for i in range(0, f["X"].shape[0], 5000):
            h.update(np.ascontiguousarray(f["X"][i]).tobytes())
    return h.hexdigest()


def work():
    try:
        import kagglehub
        log("baixando RadioML 2018.01A do Kaggle...")
        t0 = time.time()
        base = kagglehub.dataset_download("pinxau1000/radioml2018")
        log("download em %ds -> %s" % (time.time() - t0, base))
        SRC = os.path.join(base, "GOLD_XYZ_OSC.0001_1024.hdf5")
        mods = json.load(open(os.path.join(base, "classes-fixed.json")))
        snr_set = set(DESIRED_SNRS)

        gi = {g: np.array(sorted(i for i, m in enumerate(mods) if GROUP_MAP[m] == g),
                          dtype=np.int64) for g in GRUPOS}
        for g, nome in GRUPOS.items():
            log("%-5s %d classes: %s" % (nome, len(gi[g]), [mods[i] for i in gi[g]]))

        sel = {g: [] for g in GRUPOS}
        with h5py.File(SRC, "r") as src:
            N = src["X"].shape[0]
            nb = int(np.ceil(N / BUILD_BATCH))
            log("varrendo %d amostras em %d batches..." % (N, nb))
            for i in range(nb):
                s, e = i * BUILD_BATCH, min((i + 1) * BUILD_BATCH, N)
                z = src["Z"][s:e, 0]
                y = np.argmax(src["Y"][s:e], axis=1)
                msnr = np.isin(z, list(snr_set))
                for g in GRUPOS:
                    mk = msnr & np.isin(y, gi[g])
                    if mk.any():
                        sel[g].extend((np.where(mk)[0] + s).tolist())
                if (i + 1) % 400 == 0 or i == nb - 1:
                    log("  %d/%d" % (i + 1, nb))

        man = {}
        for g, nome in GRUPOS.items():
            out = "/content/%s_subset.hdf5" % nome
            idx = np.array(sel[g], dtype=np.int64)
            M, C = len(idx), len(gi[g])
            g2l = {int(v): l for l, v in enumerate(gi[g])}
            with h5py.File(SRC, "r") as src:
                xs, zs = src["X"].shape[1:], src["Z"].shape[1:]
                with h5py.File(out, "w") as o:
                    o.attrs["num_classes"] = C
                    o.attrs["group_indices"] = gi[g].tolist()
                    o.attrs["snrs"] = DESIRED_SNRS
                    o.attrs["total_samples"] = M
                    dX = o.create_dataset("X", shape=(M,) + xs, dtype=src["X"].dtype)
                    dZ = o.create_dataset("Z", shape=(M,) + zs, dtype=src["Z"].dtype)
                    dY = o.create_dataset("Y", shape=(M, C), dtype=np.int32)
                    nb2 = int(np.ceil(M / BUILD_BATCH))
                    for i in range(nb2):
                        s, e = i * BUILD_BATCH, min((i + 1) * BUILD_BATCH, M)
                        ii = idx[s:e]
                        dX[s:e] = src["X"][ii]
                        dZ[s:e] = src["Z"][ii]
                        yg = np.argmax(src["Y"][ii], axis=1)
                        yl = np.array([g2l[int(v)] for v in yg])
                        oh = np.zeros((len(yl), C), dtype=np.int32)
                        oh[np.arange(len(yl)), yl] = 1
                        dY[s:e] = oh
            fp = fingerprint(out)
            man[nome] = {"fingerprint": fp, "samples": int(M), "num_classes": int(C),
                         "chance": round(100.0 / C, 2), "classes": [mods[i] for i in gi[g]]}
            extra = ""
            if nome == "QAM":
                extra = "  [determinismo vs 1a geracao: %s]" % ("CONFERE" if fp == FP_QAM else "DIVERGIU")
            log("%-5s pronto | %7d amostras | C=%d | chance=%5.2f%% | fp=%s%s"
                % (nome, M, C, 100.0 / C, fp[:16], extra))
        json.dump(man, open("/content/grupos_manifest.json", "w"), indent=1)
        log("PREP_ALL_PRONTO")
    except Exception:
        log("ERRO:\n" + traceback.format_exc())


open(LOG, "w").close()
threading.Thread(target=work, daemon=True).start()
print("preparo dos 5 grupos iniciado em background")