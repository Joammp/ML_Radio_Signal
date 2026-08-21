# Mede o custo real de um passo de treino da 4L em CPU, para dimensionar o
# experimento em vez de chutar. Tensores sinteticos: nao depende do dataset.
import time, threading, traceback
import torch, torch.nn as nn, torch.optim as optim

LOGC = "/content/calib.log"


def log(m):
    with open(LOGC, "a") as f:
        f.write(str(m) + "\n")


ARCH = [{"out_channels": 32,  "kernel_size": 11, "pool": True},
        {"out_channels": 64,  "kernel_size": 7,  "pool": True},
        {"out_channels": 128, "kernel_size": 5,  "pool": True},
        {"out_channels": 256, "kernel_size": 3,  "pool": True}]


class FlexCNN(nn.Module):
    def __init__(self, num_classes, arch, classifier=(512,), dropout=0.5,
                 in_channels=2, input_length=1024):
        super().__init__()
        layers, ch = [], in_channels
        for b in arch:
            ks = b["kernel_size"]
            layers += [nn.Conv1d(ch, b["out_channels"], ks, padding=ks // 2),
                       nn.BatchNorm1d(b["out_channels"]), nn.ReLU(inplace=True)]
            if b.get("pool", True):
                layers.append(nn.MaxPool1d(2))
            ch = b["out_channels"]
        self.features = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        with torch.no_grad():
            n = self.features(torch.zeros(1, in_channels, input_length)).view(1, -1).size(1)
        head, prev = [], n
        for u in classifier:
            head += [nn.Linear(prev, u), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = u
        head.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x):
        return self.classifier(self.flatten(self.features(x)))


def job():
    try:
        log("threads torch: %d" % torch.get_num_threads())
        C = 5
        m = FlexCNN(C, ARCH)
        opt = optim.Adam(m.parameters(), lr=9e-5)
        crit = nn.CrossEntropyLoss()
        xb = torch.randn(64, 2, 1024)
        yb = torch.randint(0, C, (64,))

        for _ in range(3):
            opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()

        N = 20
        t0 = time.time()
        for _ in range(N):
            opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
        sb = (time.time() - t0) / N
        log("treino : %.3f s/batch (batch=64)" % sb)

        m.eval()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(N):
                m(xb)
        sv = (time.time() - t0) / N
        log("val    : %.3f s/batch" % sv)

        log("")
        log("dimensionamento (15 runs = 5 grupos x 3 seeds):")
        for ntr, nvl, eps in ((4000, 1500, 15), (6000, 2000, 20),
                              (6000, 2000, 15), (4000, 1500, 20),
                              (3000, 1000, 12)):
            btr, bvl = ntr // 64, nvl // 64
            por_ep = btr * sb + bvl * sv
            log("  N=%5d/%-5d %2d ep -> %5.0f s/epoca | %5.1f min/run | TOTAL %.1f h"
                % (ntr, nvl, eps, por_ep, por_ep * eps / 60, por_ep * eps * 15 / 3600))
        log("CALIB_PRONTO")
    except Exception:
        log("ERRO\n" + traceback.format_exc())


open(LOGC, "w").close()
threading.Thread(target=job, daemon=True).start()
print("calibracao iniciada em background")
