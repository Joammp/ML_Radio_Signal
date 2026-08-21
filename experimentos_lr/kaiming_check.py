# Mede a diferenca real entre a init default do PyTorch e a Kaiming usada no init_v2,
# na arquitetura 4L do projeto. Roda em CPU local, nao precisa de GPU.
import math
import torch
import torch.nn as nn

ARCH = [(2, 32, 11), (32, 64, 7), (64, 128, 5), (128, 256, 3)]

print("%-22s %10s %10s %10s %8s" % ("camada", "fan_in", "fan_out", "", ""))
print("%-22s %10s %10s %10s %8s" % ("", "", "", "std default", "std Kaiming"))
print("-" * 66)
for ci, co, k in ARCH:
    fan_in, fan_out = ci * k, co * k

    conv_def = nn.Conv1d(ci, co, k, padding=k // 2)          # init default do PyTorch
    std_def = conv_def.weight.std().item()

    conv_kai = nn.Conv1d(ci, co, k, padding=k // 2)
    nn.init.kaiming_normal_(conv_kai.weight, mode="fan_out", nonlinearity="relu")
    std_kai = conv_kai.weight.std().item()

    # formulas fechadas, para conferir a medicao
    std_def_teo = math.sqrt(1.0 / (3.0 * fan_in))            # kaiming_uniform_(a=sqrt(5))
    std_kai_teo = math.sqrt(2.0 / fan_out)                   # kaiming_normal_ relu fan_out

    print("Conv1d(%3d->%3d, k=%2d) %10d %10d   %.5f   %.5f   razao K/D = %.2fx"
          % (ci, co, k, fan_in, fan_out, std_def, std_kai, std_kai / std_def))
    print("%-22s %10s %10s   (teoria %.5f / %.5f)" % ("", "", "", std_def_teo, std_kai_teo))

print()
print("Ganho da default: kaiming_uniform_(a=sqrt(5)) -> gain = sqrt(2/(1+5)) = %.5f" % math.sqrt(2.0 / 6.0))
print("Ganho correto p/ ReLU:                          gain = sqrt(2)       = %.5f" % math.sqrt(2.0))

# efeito do ganho 0.01 na cabeca: onde nascem os logits
print("\n--- cabeca de classificacao (C=5) ---")
torch.manual_seed(0)
head = nn.Linear(512, 5)
nn.init.kaiming_normal_(head.weight, nonlinearity="relu")
nn.init.zeros_(head.bias)
feat = torch.randn(4096, 512)
logits_sem = head(feat)
with torch.no_grad():
    head.weight.mul_(0.01)
logits_com = head(feat)


def resumo(nome, lg):
    p = torch.softmax(lg, dim=1)
    loss = nn.functional.cross_entropy(lg, torch.randint(0, 5, (lg.size(0),)))
    print("%-16s |logit| medio=%7.3f | p_max medio=%.4f | loss=%.4f"
          % (nome, lg.abs().mean().item(), p.max(1).values.mean().item(), loss.item()))


resumo("sem ganho", logits_sem)
resumo("com ganho 0.01", logits_com)
print("%-16s uniforme p=%.4f | ln(5)=%.4f" % ("referencia", 1 / 5, math.log(5)))
