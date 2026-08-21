# -*- coding: utf-8 -*-
"""GERADO POR gerar_busca_hp.py — NAO EDITAR A MAO.

Fonte: https://raw.githubusercontent.com/Joammp/ML_Radio_Signal/atualiza-busca-hp-estratificacao/BUSCA_HP_Classes_1cap(3).ipynb
Celulas 13 (H5PyDataset) e 18 (busca), com hiperparametros
parametrizados por linha de comando.

Diferencas em relacao ao notebook, e por que:
  LEARNING_RATE  1e-3 -> 9e-05   1e-3 colapsa o QAM (15/15 folds medidos em 19/08/2026)
  LR_PATIENCE    5    -> 8      pedido do usuario
  EARLY_STOP     15   -> 20     acompanha a paciencia maior
  drive_push/pull     -> no-op    o upload ao Drive e feito pelo app LOCAL, para
                                  o token de escopo `drive` nunca sair da maquina

RESSALVA: lr=9e-05 nao e garantia. Em 3 seeds testadas no QAM 4L, 1 ainda colapsou
nesse valor, e o escape observado ocorreu de fato em 4.5e-5, apos a primeira
reducao de plato. Se o QAM voltar a colapsar em massa, 4.5e-5 e o proximo valor.
"""
import argparse
import sys

_p = argparse.ArgumentParser(description="Busca de HP de um grupo de modulacao")
_p.add_argument("--grupos", nargs="+", required=True,
                help="grupos a processar, ex.: QAM  (ASK PSK APSK QAM)")
_p.add_argument("--lr", type=float, default=9e-05,
                help="learning rate inicial (default: %(default)g)")
_p.add_argument("--lr-patience", type=int, default=8,
                help="epocas sem melhora ate reduzir o LR (default: %(default)d)")
_p.add_argument("--early-stop", type=int, default=20,
                help="epocas sem melhora, com LR no piso, ate encerrar (default: %(default)d)")
_p.add_argument("--device", default="cuda",
                help="cuda, cuda:0, cuda:1, cpu (default: %(default)s)")
_p.add_argument("--base-dir", default="/content",
                help="raiz local dos artefatos (default: %(default)s)")
_ARGS = _p.parse_args()

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

EPOCHS_PER_FOLD  = 120      # máximo de épocas por fold
LR_PATIENCE      = _ARGS.lr_patience       # épocas sem melhora → LR cai pela metade
EARLY_STOP       = _ARGS.early_stop      # épocas sem melhora → encerra o fold
LR_MIN           = 1e-9    # LR mínimo (não cai abaixo disso)

# Redução ESCALONADA: se uma redução não produzir melhora, a próxima é mais
# agressiva.  fator da n-ésima redução consecutiva = LR_FACTOR_BASE**n
#   0.5, 0.25, 0.125, 0.0625, ...   (volta a 0.5 assim que houver melhora)
# LR acumulado após n reduções = lr0 * BASE**(n(n+1)/2)
#   n=1 → 5.0e-04   n=3 → 1.6e-05   n=5 → 3.1e-08   n=6 → 4.8e-10
# ⇒ 6 reduções consecutivas levam 1e-3 abaixo de LR_MIN=1e-9,
#   ou seja 6*LR_PATIENCE = 30 épocas de estagnação (antes: inalcançável).
LR_FACTOR_BASE   = 0.5     # fator da 1ª redução
LR_FACTOR_FLOOR  = 0.01    # redução máxima permitida num único passo
MIN_DELTA        = 0.05    # p.p. de val_acc; abaixo disso é ruído, não melhora
SEED             = 42
DROPOUT          = 0.5
LEARNING_RATE    = _ARGS.lr
BATCH_SIZE       = 64
CLASSIFIER_HEAD  = [512]

DRIVE_BASE       = "/content/drive_cache/radioml_sessions"  # cache local; sincronizado com o Drive via drive_push/drive_pull
LOCAL_DIR        = "/content"
HDF5_BUILD_BATCH = 2048

ARCHITECTURES = [
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

# ══════════════════════════════════════════════════════════════════════════════
# MAPA DE GRUPOS
# ══════════════════════════════════════════════════════════════════════════════

GROUP_MAP = {
    "OOK":       0, "4ASK":      0, "8ASK":      0,
    "BPSK":      1, "QPSK":      1, "8PSK":      1,
    "16PSK":     1, "32PSK":     1, "GMSK":      1, "OQPSK":     1,
    "16APSK":    2, "32APSK":    2, "64APSK":    2, "128APSK":   2,
    "16QAM":     3, "32QAM":     3, "64QAM":     3, "128QAM":    3, "256QAM":    3,
    "AM-SSB-WC": 4, "AM-SSB-SC": 4, "AM-DSB-WC": 4, "AM-DSB-SC": 4,
    "FM":        5,
}
GROUP_NAMES = {0: "ASK", 1: "PSK", 2: "APSK", 3: "QAM", 4: "AM", 5: "FM"}

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

# ══════════════════════════════════════════════════════════════════════════════
# FlexCNN
# ══════════════════════════════════════════════════════════════════════════════

class FlexCNN(nn.Module):
    def __init__(self, num_classes, arch, classifier=None,
                 dropout=0.5, in_channels=2, input_length=1024):
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
        self.flatten  = nn.Flatten()

        with torch.no_grad():
            dummy  = torch.zeros(1, in_channels, input_length)
            n_flat = self.features(dummy).view(1, -1).size(1)

        head = []
        prev = n_flat
        for units in classifier:
            head += [nn.Linear(prev, units), nn.ReLU(inplace=True),
                     nn.Dropout(dropout)]
            prev  = units
        head.append(nn.Linear(prev, num_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x):
        return self.classifier(self.flatten(self.features(x)))

# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════

def _drive_dir(modelo):
    """Retorna e cria o diretório do modelo no Drive."""
    d = os.path.join(DRIVE_BASE, modelo)
    os.makedirs(d, exist_ok=True)
    return d


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

def ensure_hdf5(modelo, mod_classes):
    """
    Garante que o HDF5 filtrado existe localmente com labels remapeados.
    Tenta carregar do Drive; se não existir, constrói do zero.
    Sempre verifica e corrige labels globais → locais.
    """
    h5_loc  = _h5_local(modelo)
    h5_drv  = _h5_drive(modelo)
    snr_set = set(DESIRED_SNRS)

    target_id       = {v: k for k, v in GROUP_NAMES.items()}[modelo]
    group_indices   = np.array(
        sorted([i for i, m in enumerate(mod_classes)
                if GROUP_MAP[m] == target_id]),
        dtype=np.int64
    )
    num_classes     = len(group_indices)
    global_to_local = {int(g): l for l, g in enumerate(group_indices)}

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
                   lr_factor_base=None, lr_factor_floor=None, min_delta=None):
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Scheduler manual — mais granular que ReduceLROnPlateau padrão
    best_val_acc      = 0.0    # melhor absoluto (checkpoint / retorno)
    ref_acc           = 0.0    # referência para a paciência (usa min_delta)
    best_state        = None
    epochs_no_improve = 0
    consec_drops      = 0      # reduções consecutivas sem melhora → escalonamento
    history           = []
    model.to(device)

    for epoch in range(epochs):

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
        print(f"  Filtros : " +
              " → ".join(str(b["out_channels"]) for b in arch))
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

            model    = FlexCNN(num_classes=num_classes, arch=arch,
                               classifier=CLASSIFIER_HEAD, dropout=DROPOUT)
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
                "arch"        : arch,
                "n_layers"    : len(arch),
                "filters"     : [b["out_channels"] for b in arch],
                "classifier"  : CLASSIFIER_HEAD,
                "dropout"     : DROPOUT,
                "lr"          : LEARNING_RATE,
                "lr_patience" : LR_PATIENCE,
                "early_stop"  : EARLY_STOP,
                "lr_min"          : LR_MIN,
                "lr_factor_base"  : LR_FACTOR_BASE,
                "lr_factor_floor" : LR_FACTOR_FLOOR,
                "min_delta"       : MIN_DELTA,
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