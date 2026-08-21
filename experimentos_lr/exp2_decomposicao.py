# Completa a decomposicao no MESMO grupo/arquitetura, reusando o kernel:
#   I  init sozinho        -> isola a init Kaiming
#   K  clip sozinho        -> isola o clipping
#   C  init+warm+clip      -> CONTROLE POSITIVO (variante C da ablacao original)
import json, threading, traceback, time, numpy as np, h5py
LOG2="/content/exp2.log"; OUT2="/content/qam_decomp_result.json"
def log2(m):
    with open(LOG2,"a") as f: f.write("%s  %s\n"%(time.strftime("%H:%M:%S"),m))
    print(m,flush=True)
def work2():
  try:
    with h5py.File(H5,"r") as f:
        C=int(f.attrs.get("num_classes",f["Y"].shape[1])); Y=f["Y"][:]
    y=np.argmax(Y,axis=1) if Y.ndim>1 else Y
    rng=np.random.RandomState(42); ptr,pvl=N_TRAIN//C,N_VAL//C
    tr_idx,vl_idx=[],[]
    for c in range(C):
        ic=np.where(y==c)[0]; rng.shuffle(ic)
        tr_idx+=ic[:ptr].tolist(); vl_idx+=ic[ptr:ptr+pvl].tolist()
    tr_idx,vl_idx=np.sort(tr_idx),np.sort(vl_idx)
    ds_tr,ds_vl=load(tr_idx,False),load(vl_idx,False)   # norm=False em todas
    CFG=[("I_initonly",      {"norm":False,"init":True, "warm":False,"clip":False}),
         ("K_cliponly",      {"norm":False,"init":False,"warm":False,"clip":True}),
         ("C_init_warm_clip",{"norm":False,"init":True, "warm":True, "clip":True})]
    res={}
    for name,cfg in CFG:
        res[name]=[]
        for sd in SEEDS:
            log2("=== %s | seed %d ==="%(name,sd)); t0=time.time()
            r=run(cfg,sd,ds_tr,ds_vl,C); r["seed"]=sd; r["secs"]=round(time.time()-t0,1)
            log2("    -> best_vla=%.2f%%  dead=%s  (%.0fs)"%(r["best_vla"],r["dead"],r["secs"]))
            res[name].append(r); json.dump(res,open(OUT2,"w"),indent=1)
    log2("="*58)
    for name,_ in CFG:
        rs=res[name]; nd=sum(1 for r in rs if r["dead"])
        log2("%-17s mortos %d/%d | best_vla: %s"%(name,nd,len(rs),
             ", ".join("%.2f%%"%r["best_vla"] for r in rs)))
    log2("PRONTO2")
  except Exception: log2("ERRO:\n"+traceback.format_exc())
open(LOG2,"w").close()
threading.Thread(target=work2,daemon=True).start()
print("decomposicao iniciada em background")
