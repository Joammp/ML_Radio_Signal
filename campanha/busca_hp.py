# -*- coding: utf-8 -*-
"""GERADO POR gerar_busca_hp.py — NAO EDITAR A MAO.

Fonte: https://raw.githubusercontent.com/Joammp/ML_Radio_Signal/atualiza-busca-hp-estratificacao/BUSCA_HP_Classes_1cap(3).ipynb
Celulas 13 (H5PyDataset) e 18 (busca), com hiperparametros
parametrizados por linha de comando.

Diferencas em relacao ao notebook, e por que:
  LEARNING_RATE  1e-3 -> 4.5e-05   1e-3 colapsa o QAM (15/15 folds medidos em 19/08/2026)
  LR_PATIENCE    5    -> 8      pedido do usuario
  EARLY_STOP     15   -> 20     acompanha a paciencia maior
  drive_push/pull     -> no-op    o upload ao Drive e feito pelo app LOCAL, para
                                  o token de escopo `drive` nunca sair da maquina

RESSALVA: lr=4.5e-05 escolhido por medicao em 21/08/2026 (campanha/escolhe_lr.py,
QAM 4L, LR fixo sem scheduler, 3 seeds por degrau, L4 com TF32 desligado):

    9e-05    0/3 vivos   best med 20,98%   escape med ep 27,7   <- default anterior
    4.5e-05  2/3 vivos   best med 23,85%   escape med ep  7,8   <- escolhido
    2.2e-05  2/3 vivos   best med 23,23%   escape med ep  4,2
    1.1e-05  1/3 vivos   best med 22,38%   escape med ep  3,2

O comportamento e um U invertido: 9e-05 mata tudo, e descer demais tambem piora.

Nao e garantia: 1 das 3 seeds morre mesmo no melhor degrau -- e ela morreu nos
QUATRO LRs testados. Esse terco restante e colapso de inicializacao, nao problema
de LR; baixar mais nao resolve (1.1e-5 ja tentou). E o caso da camada de restart
com nova seed do train_one_fold_v2.

2.2e-05 e alternativa legitima: sua pior seed ficou a 0,07 p.p. do limiar de vivo.
Escolhido 4.5e-05 pelo teto mais alto (26,62% vs 23,57%).

PARADA ANTECIPADA: MIN_DELTA=0.1 e LR_MIN=1e-07 vieram de medicao em 21/08/2026,
sobre o history de um fold real do ASK (2L_32-64) que rodou as 120 epocas sem o
early stop disparar. Naquele fold o plato foi alcancado na ep. 27 (a 0,25 p.p.
do melhor); as 93 epocas seguintes renderam 0,25 p.p., contra um ruido de
p90=0,24 p.p. entre epocas vizinhas -- ou seja, ganho dentro do ruido.

Com os valores antigos (0.05 / 1e-9) o replay fiel do laco para so na ep. 118:
2% de economia. Com 0.1 / 1e-07 para na ep. 94: 22% de economia por 0,04 p.p.

MIN_DELTA tem de ficar ACIMA do ruido da val_acc; abaixo dele, uma oscilacao
para cima conta como melhora e zera a paciencia. Confira com
campanha/analisa_folds.py se 0.1 continua acima do ruido no SEU grupo e
arquitetura -- isto foi medido no ASK 2L_32-64, o grupo mais facil e a rede menor.
"""
import argparse
import os
import shlex
import sys

_p = argparse.ArgumentParser(description="Busca de HP de um grupo de modulacao")
_p.add_argument("--grupos", nargs="+", required=True,
                help="grupos a processar, ex.: QAM  (ASK PSK APSK QAM)")
_p.add_argument("--lr", type=float, default=4.5e-05,
                help="learning rate inicial (default: %(default)g)")
_p.add_argument("--lr-patience", type=int, default=8,
                help="epocas sem melhora ate reduzir o LR (default: %(default)d)")
_p.add_argument("--early-stop", type=int, default=20,
                help="epocas sem melhora, com LR no piso, ate encerrar (default: %(default)d)")
_p.add_argument("--min-delta", type=float, default=0.1,
                help="p.p. de val_acc que contam como melhora (default: %(default)g)")
_p.add_argument("--lr-min", type=float, default=1e-07,
                help="piso do LR; o early stop so dispara nele (default: %(default)g)")
_p.add_argument("--stagnation-patience", type=int, default=25,
                help="epocas sem avanco real ate encerrar o fold (default: %(default)d)")
_p.add_argument("--stagnation-margin", type=float, default=0.25,
                help="p.p. de val_acc que contam como avanco real (default: %(default)g)")
_p.add_argument("--modelo", choices=["cnn", "resnet", "reducao"], default="cnn",
                help="cnn = busca de arquitetura (12 candidatas); resnet = arquitetura"
                     " fixa do artigo arXiv:1712.04578 (default: %(default)s)")
_p.add_argument("--drive-base", default="/content/drive_cache/radioml_sessions",
                help="raiz dos artefatos. No Colab web, aponte para uma pasta do"
                     " Drive montado para os resultados sobreviverem a queda da"
                     " sessao (default: %(default)s)")
_p.add_argument("--epochs", type=int, default=120,
                help="teto de epocas por fold. Valor alto (ex. 1000) faz o fold"
                     " terminar SO por parada antecipada (default: %(default)d)")
_p.add_argument("--ckpt-every", type=int, default=5,
                help="epocas entre checkpoints do fold em andamento; 0 desliga"
                     " (default: %(default)d)")
_p.add_argument("--device", default="cuda",
                help="cuda, cuda:0, cuda:1, cpu (default: %(default)s)")
_p.add_argument("--base-dir", default="/content",
                help="raiz local dos artefatos (default: %(default)s)")


def _argv_bh():
    """De onde vem os parametros.

    Sob `colab exec` o script roda DENTRO do kernel IPython: sys.argv e o do
    colab_kernel_launcher (-f kernel-xxx.json), nao o que se digitou. E o
    `colab exec` 0.6.0 sequer aceita args extras -- rejeita com "No such
    option: --grupos". Como --grupos e obrigatorio, o script morria na primeira
    linha. Medido em 21/08/2026; o uso documentado no README nunca funcionou.

    Saida: variavel de ambiente, que persiste entre execs no mesmo kernel.
    Antes de rodar este script, num exec separado:
        import os; os.environ["BUSCA_HP_ARGS"] = "--grupos QAM --lr 4.5e-5"
    """
    raw = os.environ.get("BUSCA_HP_ARGS")
    if raw is not None:
        return shlex.split(raw)
    argv = sys.argv[1:]
    if any("kernel" in a and a.endswith(".json") for a in argv):
        return []                       # argv do kernel launcher: descarta
    return argv


_ARGS = _p.parse_args(_argv_bh())

# ── Fonte de dados: Kaggle publico, sem credencial ────────────────────────────
import kagglehub
path = kagglehub.dataset_download("pinxau1000/radioml2018")
modulation_classes_path = path + "/classes-fixed.json"
print("[busca_hp] dataset em %s" % path, flush=True)

# ── Drive desativado NA VM ────────────────────────────────────────────────────
# O notebook original fazia drive_push de dentro da VM, o que exige subir um
# token de escopo `drive` completo para uma maquina remota. Aqui os artefatos
# ficam apenas em disco local; quem sincroniza com o Drive e o app local.
DRIVE_FROM_VM = False


def drive_push(local_path):
    return False


def drive_pull(local_path):
    return False


from torch.utils.data import Dataset
class H5PyDataset(Dataset):
    def __init__(self, h5_filepath, data_X_name, data_Y_name, data_Z_name, indices, formats):
        self.h5_filepath = h5_filepath
        self.data_X_name = data_X_name
        self.data_Y_name = data_Y_name
        self.indices = np.array(indices)
        self.formats = formats
        self.file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Abre o arquivo apenas uma vez por worker
        if self.file is None:
            self.file = h5py.File(self.h5_filepath, 'r')
            self.X_data = self.file[self.data_X_name]
            self.Y_data = self.file[self.data_Y_name]

        # Pega o índice real baseado na sua lista de sorteados
        real_idx = self.indices[idx]

        # Busca UMA amostra (1024, 2)
        x = self.X_data[real_idx, :]
        y = self.Y_data[real_idx]

        # Normalização Individual
        # mean = x.mean()
        # std = x.std()
        # x = (x - mean) / (std + 1e-8)

        # Trata o Label
        if self.formats == 0:
            # Se y for um vetor (One-Hot), pega o índice da classe
            if hasattr(y, "__len__"):
                y = np.argmax(y)

        # Conversão para Tensor
        x_tensor = torch.from_numpy(x).float() # [1024, 2]
        x_tensor = x_tensor.permute(1, 0)      # [2, 1024] (Canais primeiro)
        y_tensor = torch.tensor(y).long()      # Valor escalar

        return x_tensor, y_tensor


# -*- coding: utf-8 -*-
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║   BUSCA DE ARQUITETURA — K-Fold CV com retomada automática entre grupos     ║
# ║                                                                              ║
# ║  Comportamento:                                                              ║
# ║    • Detecta salvamentos no Drive e continua de onde parou                  ║
# ║    • Ao terminar um grupo segue automaticamente para o próximo              ║
# ║    • LR cai pela metade a cada 2 épocas sem melhora                        ║
# ║    • Early stopping após 10 épocas sem melhora                             ║
# ║    • Divisão de dados invariante entre sessões (seed fixo + salvo no Drive) ║
# ║    • Labels remapeados automaticamente para [0, num_classes-1]              ║
# ║                                                                              ║
# ║  Pré-requisitos (células anteriores já executadas):                         ║
# ║    • kagglehub.dataset_download() → variável path                           ║
# ║    • modulation_classes_path definido                                       ║
# ║    • Classes H5PyDataset, CNN, FlexCNN definidas                            ║
# ║    • Função split() definida                                                 ║
# ║    • Drive montado                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import os, json, time, copy, random, shutil
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ResNet fiel a Tabela IV / Figura 5 de arXiv:1712.04578; ver campanha/resnet.py
import importlib.util as _ilu, os as _os


def _carrega_resnet():
    """resnet.py fica ao lado deste arquivo; na VM, em /content (o painel envia)."""
    tentativas = []
    try:
        tentativas.append(_os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)), "resnet.py"))
    except NameError:
        pass
    tentativas.append("/content/resnet.py")
    for cam in tentativas:
        if _os.path.exists(cam):
            sp = _ilu.spec_from_file_location("resnet", cam)
            mod = _ilu.module_from_spec(sp)
            sp.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("resnet.py nao encontrado em %s" % tentativas)


resnet = _carrega_resnet()
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# ▶▶  CONFIGURAÇÃO — altere apenas este bloco  ◀◀
# ══════════════════════════════════════════════════════════════════════════════

# Ordem dos grupos a processar — o script segue esta sequência automaticamente
# Grupos já completos são pulados; retoma no grupo interrompido
GRUPOS_ALVO = _ARGS.grupos
# FM tem 1 classe — não precisa de busca de arquitetura

DESIRED_SNRS     = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

K_FOLDS          = 5

# Estratificacao conjunta (classe de modulacao, SNR).
#   True  -> cada fold/particao preserva a proporcao de CADA par (classe, SNR)
#   False -> comportamento antigo: estratifica so pela classe
STRAT_BY_SNR     = True
_STRAT_TAG       = "_stratsnr" if STRAT_BY_SNR else ""

EPOCHS_PER_FOLD  = _ARGS.epochs   # teto de epocas; ver --epochs
# Medido em 22/08/2026 na busca de reducao (ASK, candidata gap1): o fold 3
# terminou com o MELHOR resultado na epoca 120, a ultima, e o bloco final de
# 20 epocas ainda ganhava +1,06 p.p. -- ou seja, o teto cortou no meio da
# subida. Com --epochs alto quem encerra e a parada por estagnacao, que mede
# se ainda ha progresso, em vez de um numero fixo que nao sabe nada do treino.
LR_PATIENCE      = _ARGS.lr_patience       # épocas sem melhora → LR cai pela metade
EARLY_STOP       = _ARGS.early_stop      # épocas sem melhora → encerra o fold
LR_MIN           = _ARGS.lr_min    # LR mínimo (não cai abaixo disso)

# Redução ESCALONADA: se uma redução não produzir melhora, a próxima é mais
# agressiva.  fator da n-ésima redução consecutiva = LR_FACTOR_BASE**n
#   0.5, 0.25, 0.125, 0.0625, ...   (volta a 0.5 assim que houver melhora)
# LR acumulado após n reduções = lr0 * BASE**(n(n+1)/2)
# Caminho ate o piso, partindo de lr0: sao n reducoes tais que
#   lr0 * BASE**(n(n+1)/2) <= LR_MIN.  Para lr0=4.5e-5 e LR_MIN=1e-7 -> n=4
#   (2.3e-5, 5.6e-6, 7.0e-7, piso).
# ATENCAO: epochs_no_improve zera A CADA REDUCAO, entao o caminho minimo ate o
#   early stop e n*LR_PATIENCE + EARLY_STOP epocas de estagnacao ININTERRUPTA.
#   Um unico falso recorde acima de MIN_DELTA reinicia a contagem inteira --
#   foi o que fez um fold do ASK rodar as 120 epocas em 21/08/2026.
LR_FACTOR_BASE   = 0.5     # fator da 1ª redução
LR_FACTOR_FLOOR  = 0.01    # redução máxima permitida num único passo
CKPT_EVERY          = _ARGS.ckpt_every           # epocas entre checkpoints
STAGNATION_PATIENCE = _ARGS.stagnation_patience  # epocas sem avanco real -> encerra
STAGNATION_MARGIN   = _ARGS.stagnation_margin    # p.p. que definem "avanco real"
MIN_DELTA        = _ARGS.min_delta    # p.p. de val_acc; abaixo disso é ruído, não melhora
SEED             = 42
DROPOUT          = 0.5
LEARNING_RATE    = _ARGS.lr
BATCH_SIZE       = 64
CLASSIFIER_HEAD  = [512]

DRIVE_BASE       = _ARGS.drive_base   # ver --drive-base
# No painel local isto e um cache em /content, espelhado para a maquina do
# usuario. No Colab web, apontar para /content/drive/MyDrive/... faz os folds e
# checkpoints sobreviverem ao teto de ~63 min sem precisar de orquestrador.
LOCAL_DIR        = "/content"
HDF5_BUILD_BATCH = 2048

# BUSCA DE REDUCAO DE PARAMETROS.
# A arquitetura convolucional fica FIXA na 2L_32-64 (a menor da busca) e o que
# varia e o que chega na densa. Medido para ASK (3 classes); ordenada da menor
# para a maior, como pedido.
_ARCH_RED = [{"out_channels": 32, "kernel_size": 7, "pool": True},
             {"out_channels": 64, "kernel_size": 5, "pool": True}]
_ARCH_RED6 = _ARCH_RED + [
    {"out_channels": 64, "kernel_size": 5, "pool": True},
    {"out_channels": 64, "kernel_size": 3, "pool": True},
    {"out_channels": 64, "kernel_size": 3, "pool": True},
    {"out_channels": 64, "kernel_size": 3, "pool": True}]

ARQ_REDUCAO = [
    {"label": "gap1",            "arch": _ARCH_RED,  "pool_saida": 1,
     "classifier": [512]},                              #     45.795
    {"label": "gap8_head128",    "arch": _ARCH_RED,  "pool_saida": 8,
     "classifier": [128]},                              #     77.027
    {"label": "gap4",            "arch": _ARCH_RED,  "pool_saida": 4,
     "classifier": [512]},                              #    144.099
    {"label": "gap8",            "arch": _ARCH_RED,  "pool_saida": 8,
     "classifier": [512]},                              #    275.171
    {"label": "6pools",          "arch": _ARCH_RED6, "pool_saida": None,
     "classifier": [512]},                              #    595.427
    {"label": "head64",          "arch": _ARCH_RED,  "pool_saida": None,
     "classifier": [64]},                               #  1.059.811
    {"label": "head128",         "arch": _ARCH_RED,  "pool_saida": None,
     "classifier": [128]},                              #  2.108.643
    {"label": "baseline_2L_512", "arch": _ARCH_RED,  "pool_saida": None,
     "classifier": [512]},                              #  8.401.635 (atual)
]

# A ResNet do artigo e uma arquitetura FIXA: nao ha o que buscar. Ela entra
# como candidata unica, e o restante do pipeline -- k-fold, estratificacao
# conjunta (classe, SNR), train_one_fold, checkpoint, parada por estagnacao --
# fica exatamente igual ao da CNN. E assim que o artigo compara VGG e ResNet.
ARQ_RESNET = [{"label": "ResNet_L%d_k%d" % (resnet.N_STACKS, resnet.KERNEL),
               "arch": None}]

ARCHITECTURES_CNN = [
    {
        "label": "2L_32-64",
        "arch": [
            {"out_channels": 32,  "kernel_size": 7, "pool": True},
            {"out_channels": 64,  "kernel_size": 5, "pool": True},
        ],
    },
    {
        "label": "2L_64-128",
        "arch": [
            {"out_channels": 64,  "kernel_size": 7, "pool": True},
            {"out_channels": 128, "kernel_size": 5, "pool": True},
        ],
    },
    {
        "label": "3L_32-64-128",
        "arch": [
            {"out_channels": 32,  "kernel_size": 7, "pool": True},
            {"out_channels": 64,  "kernel_size": 5, "pool": True},
            {"out_channels": 128, "kernel_size": 3, "pool": True},
        ],
    },
    {
        "label": "3L_64-128-256",
        "arch": [
            {"out_channels": 64,  "kernel_size": 7, "pool": True},
            {"out_channels": 128, "kernel_size": 5, "pool": True},
            {"out_channels": 256, "kernel_size": 3, "pool": True},
        ],
    },
    {
        "label": "3L_128-256-512",
        "arch": [
            {"out_channels": 128, "kernel_size": 7, "pool": True},
            {"out_channels": 256, "kernel_size": 5, "pool": True},
            {"out_channels": 512, "kernel_size": 3, "pool": True},
        ],
    },
    {
        "label": "4L_32-64-128-256",
        "arch": [
            {"out_channels": 32,  "kernel_size": 11, "pool": True},
            {"out_channels": 64,  "kernel_size": 7,  "pool": True},
            {"out_channels": 128, "kernel_size": 5,  "pool": True},
            {"out_channels": 256, "kernel_size": 3,  "pool": True},
        ],
    },
    {
        "label": "4L_64-128-256-512",
        "arch": [
            {"out_channels": 64,  "kernel_size": 11, "pool": True},
            {"out_channels": 128, "kernel_size": 7,  "pool": True},
            {"out_channels": 256, "kernel_size": 5,  "pool": True},
            {"out_channels": 512, "kernel_size": 3,  "pool": True},
        ],
    },
    {
        "label": "4L_128-256-512-512",
        "arch": [
            {"out_channels": 128, "kernel_size": 11, "pool": True},
            {"out_channels": 256, "kernel_size": 7,  "pool": True},
            {"out_channels": 512, "kernel_size": 5,  "pool": True},
            {"out_channels": 512, "kernel_size": 3,  "pool": True},
        ],
    },
    {
        "label": "5L_32-64-128-256-512",
        "arch": [
            {"out_channels": 32,   "kernel_size": 11, "pool": True},
            {"out_channels": 64,   "kernel_size": 7,  "pool": True},
            {"out_channels": 128,  "kernel_size": 5,  "pool": True},
            {"out_channels": 256,  "kernel_size": 3,  "pool": True},
            {"out_channels": 512,  "kernel_size": 3,  "pool": True},
        ],
    },
    {
        "label": "5L_64-128-256-512-1024",
        "arch": [
            {"out_channels": 64,   "kernel_size": 11, "pool": True},
            {"out_channels": 128,  "kernel_size": 7,  "pool": True},
            {"out_channels": 256,  "kernel_size": 5,  "pool": True},
            {"out_channels": 512,  "kernel_size": 3,  "pool": True},
            {"out_channels": 1024, "kernel_size": 3,  "pool": True},
        ],
    },
    {
        "label": "6L_64-64-128-128-256-512_mixpool",
        "arch": [
            {"out_channels": 64,  "kernel_size": 11, "pool": True},
            {"out_channels": 64,  "kernel_size": 7,  "pool": False},
            {"out_channels": 128, "kernel_size": 5,  "pool": True},
            {"out_channels": 128, "kernel_size": 3,  "pool": False},
            {"out_channels": 256, "kernel_size": 3,  "pool": True},
            {"out_channels": 512, "kernel_size": 3,  "pool": True},
        ],
    },
    {
        "label": "6L_32-64-64-128-256-512_mixpool",
        "arch": [
            {"out_channels": 32,  "kernel_size": 11, "pool": True},
            {"out_channels": 64,  "kernel_size": 7,  "pool": False},
            {"out_channels": 64,  "kernel_size": 5,  "pool": True},
            {"out_channels": 128, "kernel_size": 3,  "pool": False},
            {"out_channels": 256, "kernel_size": 3,  "pool": True},
            {"out_channels": 512, "kernel_size": 3,  "pool": True},
        ],
    },
]

MODELO_TIPO   = _ARGS.modelo
ARCHITECTURES = {'resnet': ARQ_RESNET,
                 'reducao': ARQ_REDUCAO}.get(MODELO_TIPO, ARCHITECTURES_CNN)

# ══════════════════════════════════════════════════════════════════════════════
# MAPA DE GRUPOS
# ══════════════════════════════════════════════════════════════════════════════

GROUP_MAP = {
    "OOK":       0, "4ASK":      0, "8ASK":      0,
    "BPSK":      1, "QPSK":      1, "8PSK":      1,
    "16PSK":     1, "32PSK":     1, "GMSK":      1, "OQPSK":     1,
    "16APSK":    2, "32APSK":    2, "64APSK":    2, "128APSK":   2,
    "16QAM":     3, "32QAM":     3, "64QAM":     3, "128QAM":    3, "256QAM":    3,
    # AM (AM-SSB-WC/SC, AM-DSB-WC/SC) e FM foram RETIRADOS do estudo.
    # Sao modulacoes analogicas; alem disso o FM tem uma unica classe, o que
    # torna a classificacao dentro do grupo degenerada. Restam 19 modulacoes
    # digitais em 4 grupos.
}
GROUP_NAMES = {0: "ASK", 1: "PSK", 2: "APSK", 3: "QAM"}
GRUPOS_DIGITAIS = ["ASK", "PSK", "APSK", "QAM"]
# alvos que nao sao um grupo isolado:
#   GROUP  -> todas as 19 digitais, rotuladas pelo GRUPO (4 classes)
#   TODAS  -> todas as 19 digitais, rotuladas pela MODULACAO (19 classes)
ALVOS_ESPECIAIS = ["GROUP", "TODAS"]

sep = "═" * 65

# ══════════════════════════════════════════════════════════════════════════════
# SEEDS
# ══════════════════════════════════════════════════════════════════════════════

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
# TF32 desligado de proposito: a T4 (Turing) nao tem TF32, entao toda a
# referencia do projeto foi medida em fp32 real. L4 (Ada) tem, e o PyTorch
# liga cudnn.allow_tf32 por padrao -> as convolucoes cairiam para 10 bits de
# mantissa (precisao de fp16) sem aviso, quebrando a comparabilidade.
torch.backends.cudnn.allow_tf32       = False
torch.backends.cuda.matmul.allow_tf32 = False

# ══════════════════════════════════════════════════════════════════════════════
# FlexCNN
# ══════════════════════════════════════════════════════════════════════════════

class FlexCNN(nn.Module):
    def __init__(self, num_classes, arch, classifier=None,
                 dropout=0.5, in_channels=2, input_length=1024,
                 pool_saida=None):
        """pool_saida: se dado, AdaptiveAvgPool1d(pool_saida) antes do flatten.

        Medido em 22/08/2026 na 2L_32-64 com 3 classes: as convolucoes sao
        10.976 parametros (0,1% do modelo) e a primeira densa e 8.390.659
        (99,9%), porque 2 poolings deixam 64 x 256 = 16.384 features. Reduzir
        esse 16.384 e o unico caminho que muda o tamanho de forma relevante;
        encurtar a cabeca ataca a dimensao errada.
        """
        super().__init__()
        if classifier is None:
            classifier = [512]

        layers = []
        ch_in  = in_channels
        for block in arch:
            ch_out = block["out_channels"]
            ks     = block["kernel_size"]
            layers += [
                nn.Conv1d(ch_in, ch_out, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(ch_out),
                nn.ReLU(inplace=True),
            ]
            if block.get("pool", True):
                layers.append(nn.MaxPool1d(2))
            ch_in = ch_out

        self.features = nn.Sequential(*layers)
        self.reduz = (nn.AdaptiveAvgPool1d(pool_saida) if pool_saida
                      else nn.Identity())
        self.flatten  = nn.Flatten()

        with torch.no_grad():
            dummy  = torch.zeros(1, in_channels, input_length)
            n_flat = self.reduz(self.features(dummy)).view(1, -1).size(1)

        head = []
        prev = n_flat
        for units in classifier:
            head += [nn.Linear(prev, units), nn.ReLU(inplace=True),
                     nn.Dropout(dropout)]
            prev  = units
        head.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x):
        return self.classifier(self.flatten(self.reduz(self.features(x))))

# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════

def _drive_dir(modelo):
    """Diretorio dos artefatos do alvo, SEPARADO por tipo de modelo.

    A CNN mantem o caminho historico <base>/<alvo>, para os folds ja
    concluidos continuarem validos. A ResNet vai para <base>/<alvo>_resnet.
    Sem isso as duas gravariam no mesmo kfold_fold_results.json e a retomada
    trataria uma busca de 12 arquiteturas e uma rede fixa como a mesma coisa.
    """
    nome = modelo if MODELO_TIPO == "cnn" else "%s_%s" % (modelo, MODELO_TIPO)
    d = os.path.join(DRIVE_BASE, nome)
    os.makedirs(d, exist_ok=True)
    return d


def _ckpt_path(modelo):
    """Checkpoint do fold EM ANDAMENTO. Nome fixo: so um fold corre por vez
    por grupo, e assim o painel sabe o que espelhar sem listar o diretorio.
    A identidade (arquitetura, fold) vai DENTRO do arquivo e e conferida."""
    return os.path.join(_drive_dir(modelo), "ckpt_atual.pt")


def _idx_path(modelo, name):
    return os.path.join(_drive_dir(modelo), f"{name}{_STRAT_TAG}.npy")


def _h5_local(modelo):
    snr_tag = '_'.join(str(s) for s in DESIRED_SNRS)
    return os.path.join(LOCAL_DIR, f"{modelo}_subset_snr_{snr_tag}.hdf5")


def _h5_drive(modelo):
    snr_tag = '_'.join(str(s) for s in DESIRED_SNRS)
    return os.path.join(_drive_dir(modelo),
                        f"{modelo}_subset_snr_{snr_tag}.hdf5")


def _results_path(modelo):
    return os.path.join(_drive_dir(modelo), "kfold_fold_results.json")


def _summary_path(modelo):
    return os.path.join(_drive_dir(modelo), "kfold_summary.json")


def _folds_path(modelo):
    return os.path.join(_drive_dir(modelo),
                        f"kfold_folds_k{K_FOLDS}{_STRAT_TAG}.json")


# ══════════════════════════════════════════════════════════════════════════════
# ESTRATIFICACAO CONJUNTA (classe, SNR)
# ══════════════════════════════════════════════════════════════════════════════

def _snr_column(h5_loc):
    """Vetor de SNR (inteiro) de todas as amostras do HDF5."""
    with h5py.File(h5_loc, 'r') as f:
        z = np.asarray(f['Z'][:])
    if z.ndim > 1:
        z = z[:, 0]
    return np.rint(z.astype(np.float64)).astype(np.int64)


def _strat_labels(y_lbl, snr_lbl, min_count, verbose=True):
    """
    Rotulo composto '<classe>_<snr>' para estratificar por classe E por SNR.

    O sklearn exige pelo menos `min_count` amostras por estrato
    (2 para train_test_split, K_FOLDS para StratifiedKFold). Pares raros
    sao fundidos num bucket '<classe>_rare'; se ainda assim for inviavel,
    cai para estratificacao apenas pela classe (com aviso).
    """
    if not STRAT_BY_SNR:
        return y_lbl.astype(str)

    y_s = np.asarray(y_lbl).astype(str)
    key = np.array([f"{c}_{s}" for c, s in zip(y_s, np.asarray(snr_lbl))])

    uniq, counts = np.unique(key, return_counts=True)
    rare = set(uniq[counts < min_count].tolist())
    if rare:
        key = np.array([f"{c}_rare" if k in rare else k
                        for k, c in zip(key, y_s)])
        if verbose:
            print(f"     ⚠ {len(rare)} par(es) (classe,SNR) com < {min_count} "
                  f"amostras → fundidos em bucket por classe")

    counts2 = np.unique(key, return_counts=True)[1]
    if counts2.min() < min_count:
        if verbose:
            print("     ⚠ estratificacao conjunta inviavel → "
                  "usando apenas a classe")
        return y_s

    if verbose:
        print(f"     Estratos (classe,SNR): {len(counts2)}  "
              f"(menor = {counts2.min():,} amostras)")
    return key


def check_snr_balance(splits, train_idx, h5_loc, max_dev_pp=0.5):
    """
    Verificacao: imprime a % de cada SNR nos folds de validacao e o desvio
    maximo (em pontos percentuais) em relacao a distribuicao global.
    """
    snr  = _snr_column(h5_loc)[train_idx]
    vals = np.unique(snr)
    ref  = np.array([100.0 * np.mean(snr == v) for v in vals])

    print("\n  Distribuicao de SNR por fold de validacao (%):")
    print("     SNR  " + " ".join(f"{v:>6}" for v in vals))
    worst = 0.0
    for i, (_, vl) in enumerate(splits):
        p = np.array([100.0 * np.mean(snr[vl] == v) for v in vals])
        worst = max(worst, float(np.abs(p - ref).max()))
        print(f"     F{i+1}   " + " ".join(f"{x:6.2f}" for x in p))
    print("     ALL  " + " ".join(f"{x:6.2f}" for x in ref))
    flag = "✅" if worst <= max_dev_pp else "⚠"
    print(f"     {flag} desvio maximo vs. global: {worst:.3f} p.p.")
    return worst


# ══════════════════════════════════════════════════════════════════════════════
# PASSO 1 — HDF5 FILTRADO COM LABELS REMAPEADOS
# ══════════════════════════════════════════════════════════════════════════════

def _alvo_info(modelo, mod_classes):
    """Indices globais, mapa global->rotulo e numero de classes de um alvo.

    Tres formas de alvo, todas alimentando o mesmo _build_hdf5:

      <grupo>  ASK/PSK/APSK/QAM -- so as modulacoes daquele grupo, rotuladas
               pela modulacao (3, 7, 4 e 5 classes respectivamente).
      GROUP    todas as 19 digitais, rotuladas pelo GRUPO -> 4 classes. E o
               classificador de primeiro estagio.
      TODAS    todas as 19 digitais, rotuladas pela MODULACAO -> 19 classes.
               E a tarefa que o artigo faz de uma vez so (la com 24, porque
               inclui as analogicas que retiramos).
    """
    digitais = sorted(i for i, m in enumerate(mod_classes) if m in GROUP_MAP)
    if modelo == "GROUP":
        idx = np.array(digitais, dtype=np.int64)
        mapa = {int(g): GROUP_MAP[mod_classes[g]] for g in digitais}
        return idx, mapa, len(GROUP_NAMES)
    if modelo == "TODAS":
        idx = np.array(digitais, dtype=np.int64)
        return idx, {int(g): l for l, g in enumerate(digitais)}, len(digitais)
    alvo = {v: k for k, v in GROUP_NAMES.items()}[modelo]
    idx = np.array(sorted(i for i, m in enumerate(mod_classes)
                          if GROUP_MAP.get(m) == alvo), dtype=np.int64)
    return idx, {int(g): l for l, g in enumerate(idx)}, len(idx)


def ensure_hdf5(modelo, mod_classes):
    """
    Garante que o HDF5 filtrado existe localmente com labels remapeados.
    Tenta carregar do Drive; se não existir, constrói do zero.
    Sempre verifica e corrige labels globais → locais.
    """
    h5_loc  = _h5_local(modelo)
    h5_drv  = _h5_drive(modelo)
    snr_set = set(DESIRED_SNRS)

    group_indices, global_to_local, num_classes = _alvo_info(modelo, mod_classes)

    # ── Carrega do Drive se disponível ───────────────────────────────────────
    if not os.path.exists(h5_loc):
        print(f"  📥 Verificando se '{modelo}' já existe no Drive...")
        if drive_pull(h5_loc):
            print(f"  📥 HDF5 baixado do Drive → {h5_loc}")
        else:
            # Constrói do zero
            print(f"  🔨 Construindo HDF5 para '{modelo}'...")
            _build_hdf5(h5_loc, mod_classes, group_indices,
                        global_to_local, num_classes, snr_set)
            # Persiste no Drive
            drive_push(h5_loc)
            print(f"  💾 HDF5 salvo no Drive: {h5_drv}")

    # ── Verifica/corrige labels ───────────────────────────────────────────────
    _fix_labels(h5_loc, global_to_local, num_classes)

    return h5_loc, num_classes, group_indices, global_to_local


def _build_hdf5(h5_out, mod_classes, group_indices,
                global_to_local, num_classes, snr_set):
    """Filtra INPUT_FILE por grupo e SNR, grava h5_out com labels locais."""
    input_file = path + '/GOLD_XYZ_OSC.0001_1024.hdf5'
    selected   = []

    with h5py.File(input_file, 'r') as src:
        N    = src['X'].shape[0]
        n_b  = int(np.ceil(N / HDF5_BUILD_BATCH))
        for i in range(n_b):
            s, e    = i * HDF5_BUILD_BATCH, min((i+1)*HDF5_BUILD_BATCH, N)
            z_batch = src['Z'][s:e, 0]
            y_batch = np.argmax(src['Y'][s:e], axis=1)
            mask    = (np.isin(z_batch, list(snr_set)) &
                       np.isin(y_batch, group_indices))
            selected.extend((np.where(mask)[0] + s).tolist())
            if (i+1) % 20 == 0 or i == n_b-1:
                print(f"    Batch {i+1}/{n_b} | selecionados: {len(selected)}")

    selected = np.array(selected, dtype=np.int64)
    M        = len(selected)

    with h5py.File(input_file, 'r') as src:
        x_shape = src['X'].shape[1:]
        z_shape = src['Z'].shape[1:]

        with h5py.File(h5_out, 'w') as out:
            out.attrs['modelo_id']      = str(group_indices[0])  # placeholder
            out.attrs['num_classes']    = num_classes
            out.attrs['group_indices']  = group_indices.tolist()
            out.attrs['snrs']           = DESIRED_SNRS
            out.attrs['total_samples']  = M
            out.attrs['labels_remapped']= True

            ds_X = out.create_dataset('X', shape=(M,)+x_shape,
                                      dtype=src['X'].dtype)
            ds_Z = out.create_dataset('Z', shape=(M,)+z_shape,
                                      dtype=src['Z'].dtype)
            ds_Y = out.create_dataset('Y', shape=(M, num_classes),
                                      dtype=np.int32)

            n_b2 = int(np.ceil(M / HDF5_BUILD_BATCH))
            for i in range(n_b2):
                s, e  = i*HDF5_BUILD_BATCH, min((i+1)*HDF5_BUILD_BATCH, M)
                idx   = selected[s:e]
                ds_X[s:e] = src['X'][idx]
                ds_Z[s:e] = src['Z'][idx]

                y_global   = np.argmax(src['Y'][idx], axis=1)
                y_local    = np.array([global_to_local[int(g)]
                                       for g in y_global])
                y_onehot   = np.zeros((len(y_local), num_classes), dtype=np.int32)
                y_onehot[np.arange(len(y_local)), y_local] = 1
                ds_Y[s:e]  = y_onehot

                if (i+1) % 10 == 0 or i == n_b2-1:
                    print(f"    Gravando batch {i+1}/{n_b2}")

    print(f"  ✅ HDF5 construído: {M} amostras, {num_classes} classes")


def _fix_labels(h5_path, global_to_local, num_classes):

    # Abre somente para leitura
    with h5py.File(h5_path, "r") as f:

        if f.attrs.get("labels_remapped", False):
            return

        y_sample = np.argmax(f["Y"][:100], axis=1)

    # <-- o arquivo já foi fechado aqui

    if y_sample.max() < num_classes:
        with h5py.File(h5_path, "a") as f:
            f.attrs["labels_remapped"] = True
        return

    print("Corrigindo labels...")

    with h5py.File(h5_path, "r") as f:
        y_global = np.argmax(f["Y"][:], axis=1)
        N = len(y_global)

    y_local = np.array([global_to_local[int(g)] for g in y_global])

    y_onehot = np.zeros((N, num_classes), dtype=np.int32)
    y_onehot[np.arange(N), y_local] = 1

    with h5py.File(h5_path, "a") as f:
        del f["Y"]
        f.create_dataset("Y", data=y_onehot)
        f.attrs["labels_remapped"] = True
        f.attrs["num_classes"] = num_classes
# ══════════════════════════════════════════════════════════════════════════════
# PASSO 2 — SPLIT DETERMINÍSTICO
# ══════════════════════════════════════════════════════════════════════════════

def ensure_split(modelo, h5_loc):
    """Carrega split do Drive ou gera e salva."""
    tp = _idx_path(modelo, "train_indices")
    vp = _idx_path(modelo, "val_indices")
    ep = _idx_path(modelo, "test_indices")

    # Tenta trazer do Drive antes de checar o cache local
    if not os.path.exists(tp):
        drive_pull(tp)
    if not os.path.exists(vp):
        drive_pull(vp)
    if not os.path.exists(ep):
        drive_pull(ep)

    if os.path.exists(tp) and os.path.exists(vp) and os.path.exists(ep):
        train_idx = np.load(tp)
        val_idx   = np.load(vp)
        test_idx  = np.load(ep)
        print(f"  ✅ Split carregado do Drive  "
              f"(treino={len(train_idx):,}  "
              f"val={len(val_idx):,}  "
              f"teste={len(test_idx):,})")
        return train_idx, val_idx, test_idx

    modo = "(classe, SNR)" if STRAT_BY_SNR else "(classe)"
    print(f"  Gerando split estratificado por {modo}  (seed={SEED})...")
    with h5py.File(h5_loc, 'r') as f:
        M       = f['X'].shape[0]
        y_lbl   = np.argmax(f['Y'][:], axis=1)

    snr_lbl = _snr_column(h5_loc)
    # min_count=5: garante >=1 amostra de cada estrato em teste, val e treino
    # apos os dois cortes sucessivos (20% e depois 25% do restante).
    strat   = _strat_labels(y_lbl, snr_lbl, min_count=5)

    indices = np.arange(M)
    tv_idx, te_idx, tv_strat, _ = train_test_split(
        indices, strat, test_size=0.2,
        random_state=SEED, stratify=strat
    )
    tr_idx, vl_idx, _, _ = train_test_split(
        tv_idx, tv_strat, test_size=0.25,
        random_state=SEED, stratify=tv_strat
    )

    train_idx = np.sort(tr_idx)
    val_idx   = np.sort(vl_idx)
    test_idx  = np.sort(te_idx)

    np.save(tp, train_idx)
    np.save(vp, val_idx)
    np.save(ep, test_idx)
    drive_push(tp)
    drive_push(vp)
    drive_push(ep)
    print(f"  ✅ Split salvo no Drive  "
          f"(treino={len(train_idx):,}  "
          f"val={len(val_idx):,}  "
          f"teste={len(test_idx):,})")
    return train_idx, val_idx, test_idx

# ══════════════════════════════════════════════════════════════════════════════
# PASSO 3 — FOLDS DETERMINÍSTICOS
# ══════════════════════════════════════════════════════════════════════════════

def ensure_folds(modelo, h5_loc, train_idx):
    """Carrega folds do Drive ou gera e salva."""
    fp = _folds_path(modelo)

    if not os.path.exists(fp):
        drive_pull(fp)

    if os.path.exists(fp):
        with open(fp) as f:
            data = json.load(f)
        assert data["k_folds"]  == K_FOLDS,          "k_folds diverge!"
        assert data["seed"]     == SEED,              "seed diverge!"
        assert data["n_train"]  == len(train_idx),    "tamanho de treino mudou!"
        assert data.get("strat", "class") == (_STRAT_TAG or "class"), \
            "estratificacao diverge! apague o JSON de folds e regenere"
        splits = [(np.array(d["train_idx"]),
                   np.array(d["val_idx"]))
                  for d in data["folds"]]
        print(f"  ✅ Folds carregados do Drive: {fp}")
        return splits

    modo = "(classe, SNR)" if STRAT_BY_SNR else "(classe)"
    print(f"  Gerando {K_FOLDS} folds estratificados por {modo}  "
          f"(seed={SEED})...")
    with h5py.File(h5_loc, 'r') as f:
        y_train = np.argmax(f['Y'][train_idx], axis=1)

    snr_train = _snr_column(h5_loc)[train_idx]
    strat     = _strat_labels(y_train, snr_train, min_count=K_FOLDS)

    skf    = StratifiedKFold(n_splits=K_FOLDS, shuffle=True,
                              random_state=SEED)
    splits = list(skf.split(np.arange(len(train_idx)), strat))

    data = {
        "modelo": modelo, "k_folds": K_FOLDS,
        "seed": SEED, "n_train": len(train_idx),
        "strat": _STRAT_TAG or "class",
        "folds": [
            {"fold": i, "n_train": len(tr), "n_val": len(vl),
             "train_idx": tr.tolist(), "val_idx": vl.tolist()}
            for i, (tr, vl) in enumerate(splits)
        ]
    }
    with open(fp, "w") as f:
        json.dump(data, f)
    drive_push(fp)

    print(f"  ✅ Folds salvos: {fp}")
    for i, (tr, vl) in enumerate(splits):
        print(f"     Fold {i+1}: treino={len(tr):,}  val={len(vl):,}")
    return splits

# ══════════════════════════════════════════════════════════════════════════════
# PASSO 4 — CARREGAR / SALVAR RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

def load_results(modelo):
    rp = _results_path(modelo)
    if not os.path.exists(rp):
        drive_pull(rp)
    if os.path.exists(rp):
        with open(rp) as f:
            data = json.load(f)
        done = {(r["label"], r["fold"]) for r in data}
        print(f"  ✅ {len(data)} resultados anteriores  "
              f"({len(done)} combinações concluídas)")
        return data, done
    return [], set()


def save_fold_result(modelo, result, all_results):
    all_results.append(result)
    rp = _results_path(modelo)
    with open(rp, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    drive_push(rp)


def update_summary(modelo, all_results, num_classes):
    from collections import defaultdict
    stats = defaultdict(list)
    meta  = {}
    for r in all_results:
        stats[r["label"]].append(r["best_val_acc"])
        if r["label"] not in meta:
            meta[r["label"]] = {
                k: r[k] for k in ["arch", "n_layers", "filters",
                                   "classifier", "dropout", "lr"]
                if k in r
            }

    summary = []
    for label, accs in stats.items():
        e = {
            "modelo"       : modelo,
            "num_classes"  : num_classes,
            "label"        : label,
            "folds_done"   : len(accs),
            "complete"     : len(accs) == K_FOLDS,
            "accs_per_fold": [round(a, 4) for a in accs],
            "mean_val_acc" : round(float(np.mean(accs)), 4),
            "std_val_acc"  : round(float(np.std(accs)),  4),
            "var_val_acc"  : round(float(np.var(accs)),  4),
            "min_val_acc"  : round(float(np.min(accs)),  4),
            "max_val_acc"  : round(float(np.max(accs)),  4),
        }
        e.update(meta.get(label, {}))
        summary.append(e)

    summary.sort(key=lambda x: (x["complete"], x["mean_val_acc"]),
                 reverse=True)
    sp = _summary_path(modelo)
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    drive_push(sp)
    return summary


def grupo_completo(modelo):
    """Retorna True se todos os folds de todas as arquiteturas estão prontos."""
    rp = _results_path(modelo)
    if not os.path.exists(rp):
        drive_pull(rp)
    if not os.path.exists(rp):
        return False
    with open(rp) as f:
        data = json.load(f)
    done = {(r["label"], r["fold"]) for r in data}
    total = len(ARCHITECTURES) * K_FOLDS
    return len(done) == total

# ══════════════════════════════════════════════════════════════════════════════
# TREINO DE UM FOLD
# ══════════════════════════════════════════════════════════════════════════════

def train_one_fold(model, tr_loader, vl_loader, device,
                   lr, epochs, lr_patience, early_stop, lr_min,
                   lr_factor_base=None, lr_factor_floor=None, min_delta=None,
                   stag_patience=None, stag_margin=None,
                   ckpt_path=None, ckpt_every=None, ckpt_id=None):
    """
    Treina um fold com:
      • Melhora só conta se superar `min_delta` p.p. (o resto é ruído)
      • Redução ESCALONADA do LR: a n-ésima redução consecutiva sem melhora
        usa fator `lr_factor_base**n` (0.5, 0.25, 0.125, ...), limitado a
        `lr_factor_floor`.  O escalonamento reinicia a cada melhora real.
      • Early stop SÓ depois de o LR atingir `lr_min` — garante que o piso
        é sempre alcançado antes de o treino terminar.

    Garantia: sob estagnação contínua, o LR vai de `lr` até `lr_min` em
    n reduções, onde lr*base**(n(n+1)/2) <= lr_min; para lr=1e-3,
    base=0.5 e lr_min=1e-9 → n=6, isto é 6*lr_patience épocas.

    Retorna (best_val_acc, history)
    """
    if lr_factor_base  is None: lr_factor_base  = LR_FACTOR_BASE
    if lr_factor_floor is None: lr_factor_floor = LR_FACTOR_FLOOR
    if min_delta       is None: min_delta       = MIN_DELTA
    if stag_patience   is None: stag_patience   = STAGNATION_PATIENCE
    if stag_margin     is None: stag_margin     = STAGNATION_MARGIN
    if ckpt_every      is None: ckpt_every      = CKPT_EVERY
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Scheduler manual — mais granular que ReduceLROnPlateau padrão
    best_val_acc      = 0.0    # melhor absoluto (checkpoint / retorno)
    ref_acc           = 0.0    # referência para a paciência (usa min_delta)
    best_state        = None
    epochs_no_improve = 0
    consec_drops      = 0      # reduções consecutivas sem melhora → escalonamento
    marco_acc         = -1e9   # último avanço REAL (supera stag_margin)
    marco_ep          = 0      # época desse avanço; base da parada absoluta
    history           = []
    model.to(device)

    # ── RETOMADA DE FOLD INTERROMPIDO ─────────────────────────────────────────
    # A VM do Colab cai a cada ~1 h. Sem isto, um fold interrompido perde tudo:
    # em 22/08/2026 o PSK morreu na epoca 93 de 120 e recomecou do zero. O
    # kfold_fold_results.json so grava folds CONCLUIDOS, entao nao ajuda aqui.
    # O `best_state` de proposito NAO entra no checkpoint: ele so alimenta um
    # load_state_dict no fim que ninguem consome (o chamador usa apenas
    # best_val_acc e history), e guarda-lo dobraria o tamanho do arquivo.
    ep_inicial = 0
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            if ckpt_id is not None and ck.get('id') != list(ckpt_id):
                # sobra de outro (arquitetura, fold): ignorar e comecar limpo
                raise ValueError('checkpoint de %r, esperado %r'
                                 % (ck.get('id'), list(ckpt_id)))

            # ATOMICIDADE. Em 22/08/2026 um fold do QAM voltou com history de
            # 185 entradas para um teto de 120: as epocas 1-65 do checkpoint
            # MAIS um treino completo 1-120. A causa era este bloco atribuir
            # direto nas variaveis e o `except` so zerar ep_inicial -- o
            # history restaurado sobrevivia e o treino recomecava por cima.
            # Agora nada e comprometido antes de TUDO dar certo.
            _model_sd = ck['model']
            _optim_sd = ck['optim']
            _novo = dict(ep_inicial=ck['epoch'], best=ck['best_val_acc'],
                         ref=ck['ref_acc'], sem=ck['epochs_no_improve'],
                         drops=ck['consec_drops'], m_acc=ck['marco_acc'],
                         m_ep=ck['marco_ep'], hist=list(ck['history']),
                         lr=ck['lr'])
            # torch.load(..., map_location=device) move TODOS os tensores do
            # checkpoint para a GPU -- inclusive os estados de RNG. Mas
            # set_rng_state e Generator.set_state exigem ByteTensor de CPU, e
            # falham com TypeError('RNG state must be a torch.ByteTensor').
            # Visto em producao em 22/08/2026 no fold 1 do TODAS/ResNet.
            def _byte_cpu(t):
                return t.detach().to('cpu', torch.uint8) if torch.is_tensor(t) else t

            if ck.get('rng_torch') is not None:
                torch.set_rng_state(_byte_cpu(ck['rng_torch']))
            if ck.get('rng_cuda') and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all([_byte_cpu(t) for t in ck['rng_cuda']])
                except Exception:
                    pass          # numero de GPUs mudou entre as VMs
            if ck.get('rng_loader') is not None and getattr(tr_loader, 'generator', None):
                tr_loader.generator.set_state(_byte_cpu(ck['rng_loader']))
            if ck.get('rng_np') is not None:
                np.random.set_state(ck['rng_np'])
            if ck.get('rng_py') is not None:
                random.setstate(ck['rng_py'])
            model.load_state_dict(_model_sd)
            optimizer.load_state_dict(_optim_sd)
            for pg in optimizer.param_groups:
                pg['lr'] = _novo['lr']

            # so aqui o estado do treino e efetivamente trocado
            ep_inicial        = _novo['ep_inicial']
            best_val_acc      = _novo['best']
            ref_acc           = _novo['ref']
            epochs_no_improve = _novo['sem']
            consec_drops      = _novo['drops']
            marco_acc         = _novo['m_acc']
            marco_ep          = _novo['m_ep']
            history           = _novo['hist']
            print(f'    ↺ Retomando da epoca {ep_inicial+1} '
                  f'(best ate aqui {best_val_acc:.2f}%, lr={_novo["lr"]:.2e})')
        except Exception as e:
            print(f'    (checkpoint descartado, comecando do zero: {e!r})')
            ep_inicial, history = 0, []
            best_val_acc = ref_acc = 0.0
            epochs_no_improve = consec_drops = 0
            marco_acc, marco_ep = -1e9, 0

    for epoch in range(ep_inicial, epochs):

        # ── Treino ────────────────────────────────────────────────────────────
        model.train()
        tr_loss = tr_correct = tr_total = 0
        pbar = tqdm(tr_loader,
                    desc=f"    Ép {epoch+1:>3}/{epochs} [tr]",
                    leave=False, ncols=80)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item()
            tr_correct += (out.argmax(1) == yb).sum().item()
            tr_total   += yb.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ── Validação ─────────────────────────────────────────────────────────
        model.eval()
        vl_loss = vl_correct = vl_total = 0
        with torch.no_grad():
            for xb, yb in vl_loader:
                xb, yb = xb.to(device), yb.to(device)
                out     = model(xb)
                loss    = criterion(out, yb)
                vl_loss    += loss.item()
                vl_correct += (out.argmax(1) == yb).sum().item()
                vl_total   += yb.size(0)

        tr_acc  = 100.0 * tr_correct / tr_total
        vl_acc  = 100.0 * vl_correct / vl_total
        tr_loss /= len(tr_loader)
        vl_loss /= len(vl_loader)
        cur_lr   = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch"     : epoch + 1,
            "train_loss": round(tr_loss, 4),
            "train_acc" : round(tr_acc,  2),
            "val_loss"  : round(vl_loss, 4),
            "val_acc"   : round(vl_acc,  2),
            "lr"        : cur_lr,
        })

        print(f"    Ép {epoch+1:>3}/{epochs} │ "
              f"tr={tr_loss:.4f}/{tr_acc:.2f}% │ "
              f"vl={vl_loss:.4f}/{vl_acc:.2f}% │ "
              f"lr={cur_lr:.2e}  "
              f"{'★' if vl_acc > best_val_acc else ''}")

        # ── Checkpoint: guarda sempre o melhor absoluto ───────────────────────
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = copy.deepcopy(model.state_dict())

        # ── Marco absoluto: base da parada por estagnação ─────────────────────
        if vl_acc > marco_acc + stag_margin:
            marco_acc, marco_ep = vl_acc, epoch + 1

        # ── Paciência: só é melhora se superar min_delta ──────────────────────
        if vl_acc > ref_acc + min_delta:
            ref_acc           = vl_acc
            epochs_no_improve = 0
            consec_drops      = 0    # a última redução funcionou → volta ao base
        else:
            epochs_no_improve += 1

        # ── Redução ESCALONADA do LR ──────────────────────────────────────────
        # Se a redução anterior não trouxe melhora, a próxima é mais forte.
        if epochs_no_improve >= lr_patience and cur_lr > lr_min:
            factor = max(lr_factor_base ** (consec_drops + 1), lr_factor_floor)
            new_lr = max(cur_lr * factor, lr_min)
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr
            consec_drops     += 1
            epochs_no_improve = 0    # janela de paciência nova para o novo LR
            print(f"    ↘  LR reduzido: {cur_lr:.2e} → {new_lr:.2e}  "
                  f"(fator={factor:.4f} │ redução consecutiva #{consec_drops})")

        # ── Early stop: só depois de o LR ter chegado ao piso ─────────────────
        elif epochs_no_improve >= early_stop and cur_lr <= lr_min * 1.001:
            print(f"    🛑 Early stopping na época {epoch+1} "
                  f"(LR no piso {lr_min:.1e} + {early_stop} épocas sem melhora)")
            break

        # ── Parada por estagnação ABSOLUTA ────────────────────────────────────
        # Independente do LR de propósito. O critério acima só dispara com o LR
        # no piso, e chegar lá depende de `epochs_no_improve`/`consec_drops`, que
        # zeram a cada oscilação para cima da val_acc. Medido em 22/08/2026: o
        # ruído p90 entre épocas é 0,33 p.p. no ASK, 0,28 no QAM e 1,33 no APSK
        # — sempre acima de MIN_DELTA. No APSK isso travou o escalonamento em
        # ×0.5, o LR parou em 3,5e-7 sem alcançar o piso de 1e-7, e o fold rodou
        # as 120 épocas inteiras. Não há valor de MIN_DELTA que resolva: a
        # varredura deu resultado não-monotônico, porque mexer no limiar muda o
        # caminho do LR. Este contador não é zerado por redução de LR nem por
        # ruído abaixo de `stag_margin`; conta desde o último avanço REAL.
        if (epoch + 1) - marco_ep >= stag_patience:
            print(f"    🛑 Parada por estagnação na época {epoch+1} "
                  f"({stag_patience} épocas sem avanço de {stag_margin} p.p.; "
                  f"último marco: {marco_acc:.2f}% na época {marco_ep})")
            break

        # ── Checkpoint periodico ──────────────────────────────────────────────
        # Custa ~1 s e salva ate uma hora de GPU quando a VM cai. Fica so o
        # arquivo mais recente (sobrescreve), gravado via .tmp + os.replace
        # para uma queda no meio da escrita nao deixar um checkpoint corrompido.
        if ckpt_path and ckpt_every and (epoch + 1) % ckpt_every == 0:
            try:
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                tmp = ckpt_path + ".tmp"
                torch.save({
                    "id"               : list(ckpt_id) if ckpt_id else None,
                    "epoch"            : epoch + 1,
                    "model"            : model.state_dict(),
                    "optim"            : optimizer.state_dict(),
                    "lr"               : optimizer.param_groups[0]["lr"],
                    "best_val_acc"     : best_val_acc,
                    "ref_acc"          : ref_acc,
                    "epochs_no_improve": epochs_no_improve,
                    "consec_drops"     : consec_drops,
                    "marco_acc"        : marco_acc,
                    "marco_ep"         : marco_ep,
                    "history"          : history,
                    # Sem estes, a retomada NAO e identica: o dropout e a ordem
                    # de embaralhamento do DataLoader continuariam de outro
                    # ponto da sequencia. O gerador do loader e o mesmo objeto
                    # que o chamador semeou com fold_seed.
                    "rng_torch"        : torch.get_rng_state(),
                    "rng_cuda"         : (torch.cuda.get_rng_state_all()
                                          if torch.cuda.is_available() else None),
                    "rng_loader"       : (tr_loader.generator.get_state()
                                          if getattr(tr_loader, "generator", None)
                                          is not None else None),
                    "rng_np"           : np.random.get_state(),
                    "rng_py"           : random.getstate(),
                }, tmp)
                os.replace(tmp, ckpt_path)
                # indice leve: o painel precisa saber a QUE fold este
                # checkpoint pertence antes de decidir se vale subir centenas
                # de MB para a VM nova. Ler o .pt so para isso seria absurdo.
                with open(ckpt_path + '.json', 'w') as _j:
                    json.dump({'id': list(ckpt_id) if ckpt_id else None,
                               'epoch': epoch + 1,
                               'best_val_acc': best_val_acc}, _j)
                print(f"    💾 checkpoint ep {epoch+1} "
                      f"({os.path.getsize(ckpt_path)/1e6:.0f} MB)")
            except Exception as e:
                print(f"    (falha ao gravar checkpoint: {e!r})")

    # Fold concluido: o checkpoint perdeu a razao de existir.
    if ckpt_path:
        for _p in (ckpt_path, ckpt_path + '.json'):
            try:
                os.remove(_p)
            except OSError:
                pass

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_acc, history

# ══════════════════════════════════════════════════════════════════════════════
# BUSCA PARA UM ÚNICO GRUPO
# ══════════════════════════════════════════════════════════════════════════════

def search_grupo(modelo, mod_classes):
    device = torch.device(_ARGS.device)

    print(f"\n{sep}")
    print(f"  🔍  GRUPO: {modelo}  │  device: {device}")
    print(sep)

    # 1. HDF5
    print(f"\n[1/4] HDF5...")
    h5_loc, num_classes, group_indices, global_to_local = \
        ensure_hdf5(modelo, mod_classes)

    # 2. Split
    print(f"\n[2/4] Split de índices...")
    train_idx, val_idx, test_idx = ensure_split(modelo, h5_loc)

    # 3. Folds
    print(f"\n[3/4] Folds...")
    fold_splits = ensure_folds(modelo, h5_loc, train_idx)
    check_snr_balance(fold_splits, train_idx, h5_loc)

    # 4. Busca
    print(f"\n[4/4] Busca de arquitetura  "
          f"({len(ARCHITECTURES)} arquiteturas × {K_FOLDS} folds)")

    all_results, done_set = load_results(modelo)

    total = len(ARCHITECTURES) * K_FOLDS
    done  = len(done_set)
    print(f"  Total jobs: {total}  │  Concluídos: {done}"
          f"  │  Restantes: {total - done}\n")

    best_mean = max(
        (r["mean_val_acc"] for r in update_summary(
            modelo, all_results, num_classes)
         if r.get("complete")),
        default=-1.0
    )

    for arch_idx, arch_entry in enumerate(ARCHITECTURES, 1):
        label = arch_entry["label"]
        arch  = arch_entry["arch"]

        folds_done = [f for f in range(K_FOLDS) if (label, f) in done_set]
        if len(folds_done) == K_FOLDS:
            accs = [r["best_val_acc"] for r in all_results
                    if r["label"] == label]
            print(f"  ⏭  [{arch_idx}/{len(ARCHITECTURES)}] "
                  f"{label}  (média={np.mean(accs):.2f}%)")
            continue

        print(f"\n{sep}")
        print(f"  [{arch_idx}/{len(ARCHITECTURES)}]  "
              f"{modelo}  │  {label}")
        if arch:
            print(f"  Filtros : " +
                  " → ".join(str(b["out_channels"]) for b in arch))
        else:
            print(f"  ResNet  : {resnet.N_STACKS} stacks × {resnet.CHANNELS} canais"
                  f"  │  kernel {resnet.KERNEL}  │  arXiv:1712.04578")
        restantes = [f+1 for f in range(K_FOLDS)
                     if (label, f) not in done_set]
        print(f"  Folds restantes: {restantes}")
        print(sep)

        t0_arch = time.time()

        for fold_i, (tr_pos, vl_pos) in enumerate(fold_splits):

            if (label, fold_i) in done_set:
                acc = next(r["best_val_acc"] for r in all_results
                           if r["label"] == label and r["fold"] == fold_i)
                print(f"\n  Fold {fold_i+1}/{K_FOLDS} — ⏭  "
                      f"(val_acc={acc:.2f}%)")
                continue

            print(f"\n  ── Fold {fold_i+1}/{K_FOLDS} "
                  f"(treino={len(tr_pos):,}  val={len(vl_pos):,}) ──")

            # Seed determinístico e único por (arch, fold)
            fold_seed = SEED + arch_idx * 100 + fold_i
            random.seed(fold_seed)
            np.random.seed(fold_seed)
            torch.manual_seed(fold_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(fold_seed)

            g = torch.Generator()
            g.manual_seed(fold_seed)

            tr_abs = train_idx[tr_pos]
            vl_abs = train_idx[vl_pos]

            tr_loader = DataLoader(
                H5PyDataset(h5_loc, 'X', 'Y', 'Z', tr_abs, formats=0),
                batch_size=BATCH_SIZE, shuffle=True,
                num_workers=0, generator=g
            )
            vl_loader = DataLoader(
                H5PyDataset(h5_loc, 'X', 'Y', 'Z', vl_abs, formats=0),
                batch_size=BATCH_SIZE, shuffle=False, num_workers=0
            )

            if MODELO_TIPO == "resnet":
                model = resnet.ResNet(num_classes=num_classes)
            else:
                # no modo reducao a candidata carrega sua propria cabeca
                model = FlexCNN(num_classes=num_classes, arch=arch,
                                classifier=arch_entry.get("classifier",
                                                          CLASSIFIER_HEAD),
                                dropout=DROPOUT,
                                pool_saida=arch_entry.get("pool_saida"))
            n_params = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)
            print(f"  Parâmetros: {n_params:,}")

            t0_fold = time.time()

            best_val_acc, history = train_one_fold(
                model      = model,
                tr_loader  = tr_loader,
                vl_loader  = vl_loader,
                device     = device,
                lr         = LEARNING_RATE,
                epochs     = EPOCHS_PER_FOLD,
                lr_patience= LR_PATIENCE,
                early_stop = EARLY_STOP,
                stag_patience = STAGNATION_PATIENCE,
                stag_margin   = STAGNATION_MARGIN,
                ckpt_path     = _ckpt_path(modelo),
                ckpt_every    = CKPT_EVERY,
                ckpt_id       = [label, fold_i],
                lr_min     = LR_MIN,
                lr_factor_base  = LR_FACTOR_BASE,
                lr_factor_floor = LR_FACTOR_FLOOR,
                min_delta       = MIN_DELTA,
            )

            elapsed = time.time() - t0_fold
            print(f"\n  ✅ Fold {fold_i+1}/{K_FOLDS}  "
                  f"val_acc={best_val_acc:.2f}%  ({elapsed:.0f}s)")

            fold_result = {
                "modelo"      : modelo,
                "num_classes" : num_classes,
                "label"       : label,
                "arch_idx"    : arch_idx,
                "fold"        : fold_i,
                "best_val_acc": round(best_val_acc, 4),
                "elapsed_s"   : round(elapsed, 1),
                "n_params"    : n_params,
                "fold_seed"   : fold_seed,
                "modelo_tipo" : MODELO_TIPO,
                "arch"        : arch,
                "n_layers"    : len(arch) if arch else resnet.N_STACKS,
                "filters"     : ([b["out_channels"] for b in arch] if arch
                                 else [resnet.CHANNELS] * resnet.N_STACKS),
                "classifier"  : CLASSIFIER_HEAD,
                "dropout"     : DROPOUT,
                "lr"          : LEARNING_RATE,
                "lr_patience" : LR_PATIENCE,
                "early_stop"  : EARLY_STOP,
                "lr_min"          : LR_MIN,
                "lr_factor_base"  : LR_FACTOR_BASE,
                "lr_factor_floor" : LR_FACTOR_FLOOR,
                "min_delta"       : MIN_DELTA,
                "stagnation_patience" : STAGNATION_PATIENCE,
                "stagnation_margin"   : STAGNATION_MARGIN,
                "history"     : history,
            }

            # Salva imediatamente — não perde progresso
            save_fold_result(modelo, fold_result, all_results)
            done_set.add((label, fold_i))
            update_summary(modelo, all_results, num_classes)

            model.cpu()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Estatísticas ao completar todos os folds da arquitetura
        accs_arch = [r["best_val_acc"] for r in all_results
                     if r["label"] == label]
        if len(accs_arch) == K_FOLDS:
            print(f"\n  📊 {modelo} │ {label}")
            print(f"     Folds     : {[round(a, 2) for a in accs_arch]}")
            print(f"     Média     : {np.mean(accs_arch):.2f}%")
            print(f"     Std       : {np.std(accs_arch):.2f}%")
            print(f"     Variância : {np.var(accs_arch):.4f}")
            print(f"     Tempo     : {time.time()-t0_arch:.0f}s")
            if np.mean(accs_arch) > best_mean:
                best_mean = np.mean(accs_arch)
                print(f"  🏆 Novo melhor: {label}  "
                      f"(média={best_mean:.2f}%)")

    # Sumário final do grupo
    summary = update_summary(modelo, all_results, num_classes)
    _print_summary(modelo, summary)
    return summary


def _print_summary(modelo, summary):
    print(f"\n{sep}")
    print(f"  🏆  RANKING — {modelo}")
    print(f"{'─'*65}")
    print(f"  {'Arquitetura':<42} {'Folds':>6} "
          f"{'Média':>8} {'Std':>7} {'Var':>8}")
    print(f"{'─'*65}")
    for s in summary[:5]:
        flag   = "✅" if s["complete"] else "⏳"
        status = f"{s['folds_done']}/{K_FOLDS}"
        print(f"  {flag} {s['label'][:40]:<40} {status:>6}"
              f" {s['mean_val_acc']:>7.2f}%"
              f" {s['std_val_acc']:>6.2f}%"
              f" {s['var_val_acc']:>8.4f}")
    print(sep)

# ══════════════════════════════════════════════════════════════════════════════
# LOOP PRINCIPAL — percorre todos os grupos automaticamente
# ══════════════════════════════════════════════════════════════════════════════

def run_all():
    mod_classes = json.load(open(modulation_classes_path))

    print(f"\n{'#'*65}")
    print(f"  BUSCA AUTOMÁTICA — {len(GRUPOS_ALVO)} grupos")
    print(f"  Grupos: {GRUPOS_ALVO}")
    print(f"  LR patience: {LR_PATIENCE} épocas  │  "
          f"Early stop: {EARLY_STOP} épocas (só com LR no piso)")
    print(f"  LR escalonado: fator {LR_FACTOR_BASE}^n  │  "
          f"piso {LR_MIN:.0e}  │  min_delta {MIN_DELTA} p.p.")
    print(f"  Parada por estagnação: {STAGNATION_PATIENCE} épocas sem avanço "
          f"de {STAGNATION_MARGIN} p.p. (independente do LR)")
    print(f"{'#'*65}")

    for gi, modelo in enumerate(GRUPOS_ALVO, 1):

        # Verifica se o grupo já está completamente processado
        if grupo_completo(modelo):
            print(f"\n  ⏭  [{gi}/{len(GRUPOS_ALVO)}] {modelo} — "
                  f"já completo, pulando")

            # Mostra sumário do grupo já concluído
            sp = _summary_path(modelo)
            if os.path.exists(sp):
                with open(sp) as f:
                    summary = json.load(f)
                if summary:
                    best = next((s for s in summary if s.get("complete")),
                                summary[0])
                    print(f"     Melhor: {best['label']}"
                          f"  (média={best['mean_val_acc']:.2f}%)")
            continue

        print(f"\n{'#'*65}")
        print(f"  [{gi}/{len(GRUPOS_ALVO)}] Iniciando grupo: {modelo}")
        print(f"{'#'*65}")

        t0 = time.time()
        search_grupo(modelo, mod_classes)
        elapsed = time.time() - t0
        print(f"\n  ⏱  Grupo '{modelo}' concluído em "
              f"{elapsed/60:.1f} min")

    # Sumário global
    print(f"\n{'#'*65}")
    print(f"  ✅  BUSCA COMPLETA — todos os grupos processados")
    print(f"{'#'*65}")
    for modelo in GRUPOS_ALVO:
        sp = _summary_path(modelo)
        if os.path.exists(sp):
            with open(sp) as f:
                summary = json.load(f)
            best = next((s for s in summary if s.get("complete")),
                        None)
            if best:
                print(f"  {modelo:<6} → {best['label']:<45}"
                      f"  {best['mean_val_acc']:.2f}% "
                      f"± {best['std_val_acc']:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# EXECUÇÃO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_all()