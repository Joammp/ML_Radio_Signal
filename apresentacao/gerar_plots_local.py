# -*- coding: utf-8 -*-
"""
Versão local (Windows) da célula de ANÁLISE DE PROGRESSO do notebook
PLOTS_HPS_CNN_nested.ipynb — somente leitura: baixa os kfold_fold_results.json
do Google Drive e gera os gráficos/tabela em ./kfold_analysis.
"""

import io
import json
import os
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")  # sem display — só salva PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "serif"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DRIVE_FOLDER_ID = "1E2bJyP18S4xq4OhBbvgryJ0Oc5Hm_tv2"
SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN_FILE = os.path.join(BASE_DIR, "token.json")

GRUPOS_ALVO = ["ASK", "PSK", "APSK", "QAM", "AM", "FM"]

# Cache local dos JSONs baixados e pasta de saída dos gráficos
DRIVE_BASE = os.path.join(BASE_DIR, "drive_cache", "radioml_sessions")
OUT_DIR = os.path.join(BASE_DIR, "kfold_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# ── AUTENTICAÇÃO ──────────────────────────────────────────────────────────────
_drive_creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
if _drive_creds.expired and _drive_creds.refresh_token:
    _drive_creds.refresh(Request())
    with open(TOKEN_FILE, "w") as f:
        f.write(_drive_creds.to_json())

_drive_service = build("drive", "v3", credentials=_drive_creds)
_about = _drive_service.about().get(fields="user").execute()
print(f"Conectado ao Google Drive como: {_about['user']['emailAddress']}")

# ── ACESSO SOMENTE LEITURA AO DRIVE (nunca cria pastas nem sobe arquivos) ────
_drive_folder_cache = {}


def _drive_find_folder(rel_dir):
    """Navega a árvore de pastas no Drive; retorna o ID ou None se não existir."""
    rel_dir = rel_dir.replace("\\", "/")
    if rel_dir in ("", "."):
        return DRIVE_FOLDER_ID
    if rel_dir in _drive_folder_cache:
        return _drive_folder_cache[rel_dir]

    parent_id = DRIVE_FOLDER_ID
    partial = ""
    for part in rel_dir.split("/"):
        if not part:
            continue
        partial = f"{partial}/{part}" if partial else part
        if partial in _drive_folder_cache:
            parent_id = _drive_folder_cache[partial]
            continue
        query = (
            f"'{parent_id}' in parents and name = '{part}' "
            f"and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        )
        resp = _drive_service.files().list(q=query, fields="files(id, name)").execute()
        files = resp.get("files", [])
        if not files:
            return None
        parent_id = files[0]["id"]
        _drive_folder_cache[partial] = parent_id
    return parent_id


def _drive_find_file(rel_dir, filename):
    parent_id = _drive_find_folder(rel_dir)
    if parent_id is None:
        return None
    query = f"'{parent_id}' in parents and name = '{filename}' and trashed = false"
    resp = _drive_service.files().list(q=query, fields="files(id, name)").execute()
    files = resp.get("files", [])
    return files[0]["id"] if files else None


def drive_pull(local_path):
    rel_path = os.path.relpath(local_path, DRIVE_BASE)
    file_id = _drive_find_file(os.path.dirname(rel_path), os.path.basename(rel_path))
    if file_id is None:
        return False
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    request = _drive_service.files().get_media(fileId=file_id)
    with io.FileIO(local_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return True


def _read_only_json(local_path):
    if not os.path.exists(local_path):
        drive_pull(local_path)
    if not os.path.exists(local_path):
        return None
    with open(local_path, "r", encoding="utf-8") as f:
        return json.load(f)


def carregar_resultados(modelo):
    local_path = os.path.join(DRIVE_BASE, modelo, "kfold_fold_results.json")
    data = _read_only_json(local_path)
    return data or []


def agrupar_por_label(results):
    by_label = defaultdict(list)
    for r in results:
        by_label[r["label"]].append(r)
    return by_label


def moda_int(values):
    """Moda desconsiderando as casas decimais (trunca para inteiro)."""
    if not values:
        return np.nan
    ints = [int(v) for v in values]
    counts = Counter(ints)
    maxc = max(counts.values())
    return sorted(v for v, c in counts.items() if c == maxc)[0]


def plot_curvas_epocas(by_label, modelo):
    """Uma curva por arquitetura tentada: gráfico do melhor fold e do pior fold."""
    if not by_label:
        return
    for kind, titulo in (("best", "melhor fold"), ("worst", "pior fold")):
        fig, ax = plt.subplots(figsize=(11, 6.5))
        plotou = False
        for label, folds in sorted(by_label.items()):
            com_hist = [r for r in folds if r.get("history")]
            if not com_hist:
                continue
            ordenado = sorted(com_hist, key=lambda r: r["best_val_acc"])
            escolhido = ordenado[-1] if kind == "best" else ordenado[0]
            ep = [h["epoch"] for h in escolhido["history"]]
            acc = [h["val_acc"] for h in escolhido["history"]]
            ax.plot(ep, acc, label=f"{label} (fold {escolhido['fold']+1}, "
                                   f"{escolhido['best_val_acc']:.2f}%)")
            plotou = True
        if not plotou:
            plt.close(fig)
            continue
        ax.set_xlabel("Época")
        ax.set_ylabel("Acurácia de validação (%)")
        ax.set_title(f"{modelo} — acurácia por época ({titulo} de cada arquitetura)")
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fname = os.path.join(OUT_DIR, f"{modelo}_epochs_{kind}_fold.png")
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  [img] {fname}")


def plot_params_vs_acc(by_label, modelo):
    labels, nparams, medias = [], [], []
    for label, folds in by_label.items():
        accs = [r["best_val_acc"] for r in folds]
        if not accs:
            continue
        labels.append(label)
        nparams.append(folds[0]["n_params"])
        medias.append(float(np.mean(accs)))
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(nparams, medias, s=70)
    for x, y, lab in zip(nparams, medias, labels):
        ax.annotate(lab, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Número de parâmetros (escala log)")
    ax.set_ylabel("Acurácia média de validação (%)")
    ax.set_title(f"{modelo} — parâmetros x acurácia média")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, f"{modelo}_params_vs_acc.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  [img] {fname}")


def montar_tabela(by_label, modelo):
    linhas = []
    for label, folds in by_label.items():
        accs = [r["best_val_acc"] for r in folds]
        if not accs:
            continue
        linhas.append({
            "modelo": modelo,
            "arquitetura": label,
            "n_params": folds[0]["n_params"],
            "folds_concluidos": len(accs),
            "media_acc": round(float(np.mean(accs)), 2),
            "max_acc": round(float(np.max(accs)), 2),
            "min_acc": round(float(np.min(accs)), 2),
            "moda_acc_int": moda_int(accs),
            "std_acc": round(float(np.std(accs)), 2),
        })
    return linhas


tabela_geral = []
for modelo in GRUPOS_ALVO:
    print(f"\n=== {modelo} ===")
    resultados = carregar_resultados(modelo)
    if not resultados:
        print(f"  [aviso] nenhum resultado encontrado para '{modelo}', pulando")
        continue
    by_label = agrupar_por_label(resultados)
    plot_curvas_epocas(by_label, modelo)
    plot_params_vs_acc(by_label, modelo)
    tabela_geral.extend(montar_tabela(by_label, modelo))

if tabela_geral:
    df = pd.DataFrame(tabela_geral).sort_values(["modelo", "media_acc"],
                                                ascending=[True, False])
    csv_path = os.path.join(OUT_DIR, "tabela_resultados.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nTabela salva em: {csv_path}\n")
    print(df.to_string(index=False))
else:
    print("\nNenhum dado encontrado em nenhum grupo.")
