# QAM: isola o WARMUP como unica intervencao, com baseline como controle.
# Reproduz ablacao.py (mesmo FlexCNN, mesmo criterio de morto, mesmo split
# RandomState(42)), mudando: dado QAM, arquitetura 4L, escala do qam_retest.py.
import os, json, math, random, threading, traceback, time
import numpy as np, h5py, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

LOG="/content/exp.log"; OUT="/content/qam_warmup_result.json"
H5="/content/QAM_subset.hdf5"
N_TRAIN,N_VAL,BATCH,EPOCHS,LR,DROPOUT = 16000,4000,64,12,1e-3,0.5
WARMUP, GRAD_CLIP, SEEDS = 300, 5.0, [0,1,2]
ARCH=[{"out_channels":32,"kernel_size":11,"pool":True},
      {"out_channels":64,"kernel_size":7,"pool":True},
      {"out_channels":128,"kernel_size":5,"pool":True},
      {"out_channels":256,"kernel_size":3,"pool":True}]
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def log(m):
    with open(LOG,"a") as f: f.write("%s  %s\n"%(time.strftime("%H:%M:%S"),m))
    print(m,flush=True)

class FlexCNN(nn.Module):
    def __init__(self,num_classes,arch,classifier=(512,),dropout=0.5,in_channels=2,input_length=1024):
        super().__init__()
        layers,ch=[],in_channels
        for b in arch:
            ks=b["kernel_size"]
            layers+=[nn.Conv1d(ch,b["out_channels"],ks,padding=ks//2),
                     nn.BatchNorm1d(b["out_channels"]),nn.ReLU(inplace=True)]
            if b.get("pool",True): layers.append(nn.MaxPool1d(2))
            ch=b["out_channels"]
        self.features=nn.Sequential(*layers); self.flatten=nn.Flatten()
        with torch.no_grad():
            n=self.features(torch.zeros(1,in_channels,input_length)).view(1,-1).size(1)
        head=[];prev=n
        for u in classifier:
            head+=[nn.Linear(prev,u),nn.ReLU(inplace=True),nn.Dropout(dropout)];prev=u
        head.append(nn.Linear(prev,num_classes)); self.classifier=nn.Sequential(*head)
    def forward(self,x): return self.classifier(self.flatten(self.features(x)))

def init_v2(model,head_gain=0.01):
    last=None
    for m in model.modules():
        if isinstance(m,nn.Conv1d):
            nn.init.kaiming_normal_(m.weight,mode="fan_out",nonlinearity="relu")
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m,nn.BatchNorm1d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Linear):
            nn.init.kaiming_normal_(m.weight,nonlinearity="relu"); nn.init.zeros_(m.bias); last=m
    if last is not None:
        with torch.no_grad(): last.weight.mul_(head_gain)
    return model

def load(idx, normalize):
    """Le uma vez do HDF5 para RAM. Valores identicos ao H5Sub do ablacao.py."""
    with h5py.File(H5,"r") as f:
        X=np.asarray(f["X"][np.sort(idx),:],dtype=np.float32)
        Y=f["Y"][np.sort(idx)]
    y=np.argmax(Y,axis=1) if Y.ndim>1 else Y
    if normalize:
        mu=X.mean(axis=(1,2),keepdims=True); sd=X.std(axis=(1,2),keepdims=True)
        X=(X-mu)/(sd+1e-8)
    return TensorDataset(torch.from_numpy(X).permute(0,2,1).contiguous(),
                         torch.from_numpy(np.asarray(y)).long())

def run(cfg,seed,ds_tr,ds_vl,C):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    g=torch.Generator(); g.manual_seed(seed)
    tr=DataLoader(ds_tr,batch_size=BATCH,shuffle=True,generator=g)
    vl=DataLoader(ds_vl,batch_size=BATCH,shuffle=False)
    model=FlexCNN(C,ARCH,dropout=DROPOUT)
    if cfg["init"]: init_v2(model)
    model=model.to(DEV)
    crit=nn.CrossEntropyLoss(); opt=optim.Adam(model.parameters(),lr=LR)
    step=0; hist=[]
    for ep in range(EPOCHS):
        model.train(); L=corr=tot=0
        for xb,yb in tr:
            xb,yb=xb.to(DEV,non_blocking=True),yb.to(DEV,non_blocking=True)
            if cfg["warm"]:
                w=LR*(0.01+0.99*step/WARMUP) if step<WARMUP else LR
                for pg in opt.param_groups: pg["lr"]=w
            opt.zero_grad(); out=model(xb); loss=crit(out,yb); loss.backward()
            if cfg["clip"]: nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP)
            opt.step(); step+=1
            L+=loss.item(); corr+=(out.argmax(1)==yb).sum().item(); tot+=yb.size(0)
        trl,tra=L/len(tr),100.0*corr/tot
        model.eval(); L=corr=tot=0
        with torch.no_grad():
            for xb,yb in vl:
                xb,yb=xb.to(DEV),yb.to(DEV); out=model(xb)
                L+=crit(out,yb).item(); corr+=(out.argmax(1)==yb).sum().item(); tot+=yb.size(0)
        vll,vla=L/len(vl),100.0*corr/tot
        hist.append({"ep":ep+1,"trl":round(trl,4),"tra":round(tra,2),
                     "vll":round(vll,4),"vla":round(vla,2)})
        log("      ep%-2d tr=%.4f/%.2f%%  vl=%.4f/%.2f%%"%(ep+1,trl,tra,vll,vla))
    chance,cl=100.0/C,math.log(C)
    dead=(hist[-1]["tra"]<=chance*1.15 and abs(hist[-1]["trl"]-cl)<=0.02)
    return {"best_vla":max(h["vla"] for h in hist),"dead":dead,"hist":hist}

def work():
  try:
    with h5py.File(H5,"r") as f:
        C=int(f.attrs.get("num_classes",f["Y"].shape[1])); N=f["X"].shape[0]
        Y=f["Y"][:]; y=np.argmax(Y,axis=1) if Y.ndim>1 else Y
    log("QAM: N=%d C=%d chance=%.2f%% ln(C)=%.4f | device=%s"%(N,C,100.0/C,math.log(C),DEV))
    log("distribuicao de classes: %s"%np.bincount(y,minlength=C).tolist())
    rng=np.random.RandomState(42); ptr,pvl=N_TRAIN//C,N_VAL//C
    tr_idx,vl_idx=[],[]
    for c in range(C):
        ic=np.where(y==c)[0]; rng.shuffle(ic)
        tr_idx+=ic[:ptr].tolist(); vl_idx+=ic[ptr:ptr+pvl].tolist()
    tr_idx,vl_idx=np.sort(tr_idx),np.sort(vl_idx)
    log("split estratificado: train=%d val=%d"%(len(tr_idx),len(vl_idx)))
    cache={}
    CONFIGS=[("A_baseline ",{"norm":False,"init":False,"warm":False,"clip":False}),
             ("W_warmonly ",{"norm":False,"init":False,"warm":True, "clip":False})]
    res={}
    for name,cfg in CONFIGS:
        key=cfg["norm"]
        if key not in cache: cache[key]=(load(tr_idx,key),load(vl_idx,key))
        ds_tr,ds_vl=cache[key]
        res[name.strip()]=[]
        for sd in SEEDS:
            log("=== %s | seed %d ==="%(name.strip(),sd)); t0=time.time()
            r=run(cfg,sd,ds_tr,ds_vl,C); r["seed"]=sd; r["secs"]=round(time.time()-t0,1)
            log("    -> best_vla=%.2f%%  dead=%s  (%.0fs)"%(r["best_vla"],r["dead"],r["secs"]))
            res[name.strip()].append(r)
            json.dump(res,open(OUT,"w"),indent=1)
    log("="*58)
    for name,_ in CONFIGS:
        rs=res[name.strip()]; nd=sum(1 for r in rs if r["dead"])
        log("%-12s mortos %d/%d | best_vla: %s"%(name.strip(),nd,len(rs),
            ", ".join("%.2f%%"%r["best_vla"] for r in rs)))
    log("PRONTO")
  except Exception: log("ERRO:\n"+traceback.format_exc())

open(LOG,"w").close()
threading.Thread(target=work,daemon=True).start()
print("experimento iniciado em background; acompanhe /content/exp.log")
