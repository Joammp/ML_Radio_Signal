"""Fase 0, parte A — contencao pura de GPU, sem dado real.

Mede se N treinos concorrentes numa unica GPU batem os mesmos N em sequencia.
Usa tensores sinteticos com a forma real (batch 64 x 2 x 1024), entao isola a
contencao de GPU do custo de I/O do HDF5 (medido na parte B, bench_real.py).

Roda como job de background e escreve em /content/bench_sint.log.
"""
import json, os, subprocess, sys, threading, time, traceback

LOG = "/content/bench_sint.log"
OUT = "/content/bench_sint.json"
WORKER = "/content/bench_worker_sint.py"
N_JOBS = 4          # um por grupo digital: ASK, PSK, APSK, QAM
STEPS = 400         # passos cronometrados, apos aquecimento
WARMUP = 20

# arquitetura media (a que usamos nos experimentos) e a maior do espaco de busca
ARCH_MEDIA = "4L_32-64-128-256"
ARCH_MAIOR = "5L_64-128-256-512-1024"

WORKER_SRC = r'''
import json, sys, time
import torch, torch.nn as nn, torch.optim as optim

ARCHS = {
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

tag, arch_label, steps, warmup = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
dev = torch.device("cuda")
t_import = time.time()

try:
    m = FlexCNN(5, ARCHS[arch_label]).to(dev)
    opt = optim.Adam(m.parameters(), lr=9e-5)
    crit = nn.CrossEntropyLoss()
    xb = torch.randn(64, 2, 1024, device=dev)
    yb = torch.randint(0, 5, (64,), device=dev)

    for _ in range(warmup):
        opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
    torch.cuda.synchronize()
    dt = time.time() - t0

    res = {"tag": tag, "arch": arch_label, "ok": True, "steps": steps,
           "segundos": round(dt, 3), "ms_por_passo": round(1000*dt/steps, 3),
           "passos_por_s": round(steps/dt, 2),
           "mem_pico_GB": round(torch.cuda.max_memory_allocated()/1e9, 3),
           "setup_s": round(t0 - t_import, 2)}
except RuntimeError as e:
    res = {"tag": tag, "arch": arch_label, "ok": False,
           "erro": type(e).__name__, "msg": str(e)[:200],
           "oom": "out of memory" in str(e).lower()}
print("RESULT " + json.dumps(res), flush=True)
'''


def log(m):
    with open(LOG, "a") as f:
        f.write("%s  %s\n" % (time.strftime("%H:%M:%S"), m))
    print(m, flush=True)


def roda(tag, arch, steps=STEPS):
    """Lanca um worker e devolve (popen, tag)."""
    return subprocess.Popen(
        [sys.executable, WORKER, tag, arch, str(steps), str(WARMUP)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def colhe(p):
    out, _ = p.communicate()
    for ln in out.splitlines():
        if ln.startswith("RESULT "):
            return json.loads(ln[7:])
    return {"ok": False, "erro": "sem RESULT", "raw": out[-400:]}


def gpu_info():
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        return q.stdout.strip()
    except Exception as e:
        return "nvidia-smi indisponivel: %s" % e


def bloco(nome, arch, modo):
    """modo: 'seq' roda um de cada vez; 'conc' roda os N juntos."""
    log("--- %s | %s | %s ---" % (nome, arch, modo))
    t0 = time.time()
    if modo == "seq":
        res = []
        for i in range(N_JOBS):
            r = colhe(roda("job%d" % i, arch))
            res.append(r)
            if r.get("ok"):
                log("    job%d: %.1fs  %.2f passos/s  pico %.2f GB"
                    % (i, r["segundos"], r["passos_por_s"], r["mem_pico_GB"]))
            else:
                log("    job%d: FALHOU %s" % (i, r.get("erro")))
    else:
        ps = [roda("job%d" % i, arch) for i in range(N_JOBS)]
        time.sleep(25)                      # amostra a GPU com todos rodando
        log("    nvidia-smi no meio: %s" % gpu_info())
        res = [colhe(p) for p in ps]
        for i, r in enumerate(res):
            if r.get("ok"):
                log("    job%d: %.1fs  %.2f passos/s  pico %.2f GB"
                    % (i, r["segundos"], r["passos_por_s"], r["mem_pico_GB"]))
            else:
                log("    job%d: FALHOU %s%s" % (i, r.get("erro"),
                    "  [OOM]" if r.get("oom") else ""))
    wall = time.time() - t0
    log("    WALL-CLOCK %s: %.1f s" % (modo, wall))
    return {"wall": round(wall, 2), "jobs": res}


def work():
    try:
        open(WORKER, "w").write(WORKER_SRC)
        import torch
        log("GPU: %s" % gpu_info())
        log("torch %s | cuda %s | %d jobs | %d passos por job"
            % (torch.__version__, torch.version.cuda, N_JOBS, STEPS))

        r = {}
        r["seq_media"] = bloco("arquitetura media", ARCH_MEDIA, "seq")
        r["conc_media"] = bloco("arquitetura media", ARCH_MEDIA, "conc")

        sq, cc = r["seq_media"]["wall"], r["conc_media"]["wall"]
        ganho = sq / cc if cc else 0
        r["speedup_media"] = round(ganho, 3)
        log("=" * 60)
        log("SEQUENCIAL  %.1f s" % sq)
        log("CONCORRENTE %.1f s" % cc)
        log("SPEEDUP     %.2fx  (concorrente = %.0f%% do sequencial)"
            % (ganho, 100 * cc / sq))
        if cc < 0.7 * sq:
            log("VEREDITO: PASSA — paralelismo numa GPU compensa")
        elif cc < sq:
            log("VEREDITO: MARGINAL — ganho existe mas e pequeno; decidir com o usuario")
        else:
            log("VEREDITO: REPROVA — concorrencia nao ajuda; ir para multi-sessao")

        # teste de memoria: a maior arquitetura do espaco, N contextos CUDA
        log("")
        r["conc_maior"] = bloco("MAIOR arquitetura (teste de OOM)", ARCH_MAIOR, "conc")
        ooms = sum(1 for j in r["conc_maior"]["jobs"] if j.get("oom"))
        r["oom_maior"] = ooms
        log("OOM com %s em %d contextos: %d de %d jobs"
            % (ARCH_MAIOR, N_JOBS, ooms, N_JOBS))

        json.dump(r, open(OUT, "w"), indent=1)
        log("BENCH_SINT_PRONTO")
    except Exception:
        log("ERRO:\n" + traceback.format_exc())


open(LOG, "w").close()
threading.Thread(target=work, daemon=True).start()
print("benchmark sintetico iniciado em background")
