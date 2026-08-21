# Recria o subset QAM do zero, deterministicamente, a partir do RadioML 2018.01A.
# Reproduz _build_hdf5 do notebook (celula 18): filtro puro por (grupo, SNR),
# ordem crescente de indice, sem nenhuma fonte de aleatoriedade.
import os, json, threading, traceback, hashlib, time
import numpy as np, h5py

LOG = "/content/prep.log"
OUT = "/content/QAM_subset.hdf5"
DESIRED_SNRS     = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
HDF5_BUILD_BATCH = 2048
GROUP_MAP = {
    "OOK":0,"4ASK":0,"8ASK":0,
    "BPSK":1,"QPSK":1,"8PSK":1,"16PSK":1,"32PSK":1,"GMSK":1,"OQPSK":1,
    "16APSK":2,"32APSK":2,"64APSK":2,"128APSK":2,
    "16QAM":3,"32QAM":3,"64QAM":3,"128QAM":3,"256QAM":3,
    "AM-SSB-WC":4,"AM-SSB-SC":4,"AM-DSB-WC":4,"AM-DSB-SC":4,
    "FM":5,
}
TARGET_GROUP = 3  # QAM

def log(m):
    with open(LOG,"a") as f: f.write("%s  %s\n" % (time.strftime("%H:%M:%S"), m))
    print(m, flush=True)

def work():
    try:
        import kagglehub
        log("baixando dataset do Kaggle (~21.5 GB)...")
        t0=time.time()
        path = kagglehub.dataset_download("pinxau1000/radioml2018")
        log("download concluido em %ds -> %s" % (time.time()-t0, path))
        log("VERSAO PINADA: %s" % path)

        mod_classes = json.load(open(os.path.join(path,"classes-fixed.json")))
        group_indices = np.array(sorted(i for i,m in enumerate(mod_classes)
                                        if GROUP_MAP[m]==TARGET_GROUP), dtype=np.int64)
        num_classes = len(group_indices)
        g2l = {int(g):l for l,g in enumerate(group_indices)}
        log("classes QAM: %s (ids globais %s)" %
            ([mod_classes[i] for i in group_indices], group_indices.tolist()))

        src_file = os.path.join(path,"GOLD_XYZ_OSC.0001_1024.hdf5")
        log("fonte: %s (%.1f GB)" % (src_file, os.path.getsize(src_file)/1e9))
        snr_set = set(DESIRED_SNRS)

        selected=[]
        with h5py.File(src_file,"r") as src:
            N = src["X"].shape[0]
            nb = int(np.ceil(N/HDF5_BUILD_BATCH))
            log("varrendo %d amostras em %d batches..." % (N, nb))
            for i in range(nb):
                s,e = i*HDF5_BUILD_BATCH, min((i+1)*HDF5_BUILD_BATCH, N)
                z = src["Z"][s:e,0]; y = np.argmax(src["Y"][s:e],axis=1)
                mask = np.isin(z,list(snr_set)) & np.isin(y,group_indices)
                selected.extend((np.where(mask)[0]+s).tolist())
                if (i+1)%200==0 or i==nb-1:
                    log("  varredura %d/%d | selecionados: %d" % (i+1,nb,len(selected)))
        selected=np.array(selected,dtype=np.int64); M=len(selected)
        log("selecionados: %d amostras" % M)

        with h5py.File(src_file,"r") as src:
            xs, zs = src["X"].shape[1:], src["Z"].shape[1:]
            with h5py.File(OUT,"w") as out:
                out.attrs["num_classes"]=num_classes
                out.attrs["group_indices"]=group_indices.tolist()
                out.attrs["snrs"]=DESIRED_SNRS
                out.attrs["total_samples"]=M
                out.attrs["labels_remapped"]=True
                dX=out.create_dataset("X",shape=(M,)+xs,dtype=src["X"].dtype)
                dZ=out.create_dataset("Z",shape=(M,)+zs,dtype=src["Z"].dtype)
                dY=out.create_dataset("Y",shape=(M,num_classes),dtype=np.int32)
                nb2=int(np.ceil(M/HDF5_BUILD_BATCH))
                for i in range(nb2):
                    s,e=i*HDF5_BUILD_BATCH, min((i+1)*HDF5_BUILD_BATCH,M)
                    idx=selected[s:e]
                    dX[s:e]=src["X"][idx]; dZ[s:e]=src["Z"][idx]
                    yg=np.argmax(src["Y"][idx],axis=1)
                    yl=np.array([g2l[int(v)] for v in yg])
                    oh=np.zeros((len(yl),num_classes),dtype=np.int32)
                    oh[np.arange(len(yl)),yl]=1
                    dY[s:e]=oh
                    if (i+1)%20==0 or i==nb2-1:
                        log("  gravando %d/%d" % (i+1,nb2))

        # fingerprint deterministico: prova de que a regeneracao e identica
        h=hashlib.sha256()
        with h5py.File(OUT,"r") as f:
            h.update(np.asarray(f["X"].shape).tobytes())
            h.update(f["Y"][:].tobytes())
            h.update(f["Z"][:].tobytes())
            for i in range(0, f["X"].shape[0], 5000):
                h.update(np.ascontiguousarray(f["X"][i]).tobytes())
        fp=h.hexdigest()
        log("FINGERPRINT sha256 = %s" % fp)
        json.dump({"fingerprint":fp,"samples":int(M),"num_classes":int(num_classes),
                   "snrs":DESIRED_SNRS,"kaggle_path":path,
                   "classes":[mod_classes[i] for i in group_indices]},
                  open("/content/qam_manifest.json","w"), indent=1)
        log("PRONTO")
    except Exception:
        log("ERRO:\n"+traceback.format_exc())

open(LOG,"w").close()
threading.Thread(target=work, daemon=True).start()
print("preparo iniciado em background; acompanhe /content/prep.log")
