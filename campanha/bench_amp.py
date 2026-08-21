"""Fase 0, parte B — quanto custa cada job, e quanto da para baratear.

O gate anterior mostrou que a T4 satura com UM job (100% de utilizacao, vazao
agregada CAI 9% com 4 concorrentes). Logo o ganho nao vem de rodar mais jobs,
vem de fazer cada job custar menos. Mede duas alavancas independentes:

  1. AMP (torch.cuda.amp)  — tensor cores; todos os canais da busca sao
     multiplos de 8, condicao para o cuDNN usa-los.
  2. cudnn.benchmark       — o notebook fixa deterministic=True/benchmark=False
     por reprodutibilidade. Isso tem preco, e o preco nunca foi medido.

Escreve /content/bench_amp.log e /content/bench_amp.json.
"""
import json, subprocess, sys, threading, time, traceback

LOG = "/content/bench_amp.log"
OUT = "/content/bench_amp.json"
WORKER = "/content/bench_worker_amp.py"
STEPS = 300
WARMUP = 30

ARCHS_TESTE = ["2L_32-64", "4L_32-64-128-256", "5L_64-128-256-512-1024"]
MODOS = [("fp32_determin", False, True),      # (rotulo, amp, deterministic)
         ("fp32_benchmark", False, False),
         ("amp_determin", True, True),
         ("amp_benchmark", True, False)]

WORKER_SRC = r'''
import json, sys, time
import torch, torch.nn as nn, torch.optim as optim

ARCHS = {
 "2L_32-64": [
   {"out_channels":32,"kernel_size":7,"pool":True},
   {"out_channels":64,"kernel_size":5,"pool":True}],
 "4L_32-64-128-256": [
   {"out_channels":32,"kernel_size":11,"pool":True},
   {"out_channels":64,"kernel_size":7,"pool":True},
   {"out_channels":128,"kernel_size":5,"pool":True},
   {"out_channels":256,"kernel_size":3,"pool":True}],
 "5L_64-128-256-512-1024": [
   {"out_channels":64,"kernel_size":7,"pool":True},
   {"out_channels":128,"kernel_size":5,"pool":True},
   {"out_channels":256,"kernel_size":5,"pool":True},
   {"out_channels":512,"kernel_size":3,"pool":True},
   {"out_channels":1024,"kernel_size":3,"pool":True}],
}

class FlexCNN(nn.Module):
    def __init__(self, num_classes, arch, classifier=(512,), dropout=0.5,
                 in_channels=2, input_length=1024):
        super().__init__()
        layers, ch = [], in_channels
        for b in arch:
            ks = b["kernel_size"]
            layers += [nn.Conv1d(ch, b["out_channels"], ks, padding=ks//2),
                       nn.BatchNorm1d(b["out_channels"]), nn.ReLU(inplace=True)]
            if b.get("pool", True):
                layers.append(nn.MaxPool1d(2))
            ch = b["out_channels"]
        self.features = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        with torch.no_grad():
            n = self.features(torch.zeros(1, in_channels, input_length)).view(1,-1).size(1)
        head, prev = [], n
        for u in classifier:
            head += [nn.Linear(prev,u), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = u
        head.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*head)
    def forward(self, x):
        return self.classifier(self.flatten(self.features(x)))

rotulo, arch_label = sys.argv[1], sys.argv[2]
usa_amp = sys.argv[3] == "1"
determin = sys.argv[4] == "1"
steps, warmup = int(sys.argv[5]), int(sys.argv[6])

torch.backends.cudnn.deterministic = determin
torch.backends.cudnn.benchmark = not determin
dev = torch.device("cuda")

try:
    m = FlexCNN(5, ARCHS[arch_label]).to(dev)
    opt = optim.Adam(m.parameters(), lr=9e-5)
    crit = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=usa_amp)
    xb = torch.randn(64, 2, 1024, device=dev)
    yb = torch.randint(0, 5, (64,), device=dev)

    def passo():
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=usa_amp):
            loss = crit(m(xb), yb)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return loss

    for _ in range(warmup):
        passo()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(steps):
        ultima = passo()
    torch.cuda.synchronize()
    dt = time.time() - t0

    res = {"rotulo": rotulo, "arch": arch_label, "ok": True,
           "amp": usa_amp, "determin": determin,
           "ms_por_passo": round(1000*dt/steps, 3),
           "passos_por_s": round(steps/dt, 2),
           "mem_pico_GB": round(torch.cuda.max_memory_allocated()/1e9, 3),
           "loss_final": round(float(ultima.item()), 5)}
except RuntimeError as e:
    res = {"rotulo": rotulo, "arch": arch_label, "ok": False,
           "erro": type(e).__name__, "msg": str(e)[:200],
           "oom": "out of memory" in str(e).lower()}
print("RESULT " + json.dumps(res), flush=True)
'''


def log(m):
    with open(LOG, "a") as f:
        f.write("%s  %s\n" % (time.strftime("%H:%M:%S"), m))
    print(m, flush=True)


def mede(rotulo, arch, amp, determin):
    p = subprocess.run(
        [sys.executable, WORKER, rotulo, arch, "1" if amp else "0",
         "1" if determin else "0", str(STEPS), str(WARMUP)],
        capture_output=True, text=True)
    for ln in p.stdout.splitlines():
        if ln.startswith("RESULT "):
            return json.loads(ln[7:])
    return {"ok": False, "erro": "sem RESULT",
            "raw": (p.stdout + p.stderr)[-300:]}


def work():
    try:
        open(WORKER, "w").write(WORKER_SRC)
        res = {}
        for arch in ARCHS_TESTE:
            log("=" * 62)
            log("ARQUITETURA %s" % arch)
            base = None
            res[arch] = {}
            for rotulo, amp, determin in MODOS:
                r = mede(rotulo, arch, amp, determin)
                res[arch][rotulo] = r
                if not r.get("ok"):
                    log("  %-16s FALHOU: %s %s" % (rotulo, r.get("erro"), r.get("msg", "")[:80]))
                    continue
                if base is None:
                    base = r["ms_por_passo"]
                ganho = base / r["ms_por_passo"]
                log("  %-16s %7.2f ms/passo | %6.1f passos/s | %.2f GB | ganho %.2fx | loss %.5f"
                    % (rotulo, r["ms_por_passo"], r["passos_por_s"],
                       r["mem_pico_GB"], ganho, r["loss_final"]))

        # projecao da campanha completa a partir do ganho medido
        log("=" * 62)
        log("PROJECAO DA CAMPANHA (254 h estimadas em fp32+determin)")
        for rotulo, _, _ in MODOS[1:]:
            ganhos = [res[a][MODOS[0][0]]["ms_por_passo"] / res[a][rotulo]["ms_por_passo"]
                      for a in ARCHS_TESTE
                      if res[a].get(rotulo, {}).get("ok") and res[a][MODOS[0][0]].get("ok")]
            if ganhos:
                g = sum(ganhos) / len(ganhos)
                log("  %-16s ganho medio %.2fx  ->  ~%.0f h" % (rotulo, g, 254 / g))
        json.dump(res, open(OUT, "w"), indent=1)
        log("BENCH_AMP_PRONTO")
    except Exception:
        log("ERRO:\n" + traceback.format_exc())


open(LOG, "w").close()
threading.Thread(target=work, daemon=True).start()
print("benchmark de AMP/determinismo iniciado em background")
