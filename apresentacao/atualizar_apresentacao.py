# -*- coding: utf-8 -*-
"""
Atualiza a apresentação de resultados de ponta a ponta. A cada execução:

  1. Baixa do Google Drive os kfold_fold_results.json mais recentes de cada
     grupo (sempre re-baixa — não usa cache local);
  2. Regenera os gráficos em ./kfold_analysis;
  3. Gera o arquivo apresentacao_kfold_cnn.tex dinamicamente a partir dos
     dados (slides de um grupo só existem se houver resultados; a tabela e os
     números citados são todos calculados — nada de conclusão qualitativa
     escrita à mão);
  4. Compila o PDF com pdflatex (2 passadas).

Somente leitura no Drive: nunca cria pastas nem sobe arquivos.
Uso:  python atualizar_apresentacao.py
"""

import datetime
import io
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Estilo escuro (Variação 4B): fundo grafite igual ao dos slides
GRAFITE = "#1A1A1A"
CLARO = "#E6E6E6"
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "figure.facecolor": GRAFITE,
    "axes.facecolor": GRAFITE,
    "savefig.facecolor": GRAFITE,
    "text.color": CLARO,
    "axes.labelcolor": CLARO,
    "axes.edgecolor": CLARO,
    "xtick.color": CLARO,
    "ytick.color": CLARO,
    "legend.facecolor": GRAFITE,
    "legend.edgecolor": CLARO,
    "legend.labelcolor": CLARO,
    "grid.color": "#777777",
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DRIVE_FOLDER_ID = "1E2bJyP18S4xq4OhBbvgryJ0Oc5Hm_tv2"
SCOPES = ["https://www.googleapis.com/auth/drive"]
TOKEN_FILE = os.path.join(BASE_DIR, "token.json")

GRUPOS_ALVO = ["ASK", "PSK", "APSK", "QAM", "AM", "FM"]

DRIVE_BASE = os.path.join(BASE_DIR, "drive_cache", "radioml_sessions")
OUT_DIR = os.path.join(BASE_DIR, "kfold_analysis")
TEX_PATH = os.path.join(BASE_DIR, "apresentacao_kfold_cnn.tex")
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. DRIVE — acesso somente leitura
# ══════════════════════════════════════════════════════════════════════════════
try:
    _drive_creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if _drive_creds.expired and _drive_creds.refresh_token:
        _drive_creds.refresh(Request())
        with open(TOKEN_FILE, "w") as f:
            f.write(_drive_creds.to_json())
    _drive_service = build("drive", "v3", credentials=_drive_creds)
    _about = _drive_service.about().get(fields="user").execute()
    print(f"Conectado ao Google Drive como: {_about['user']['emailAddress']}")
except Exception as exc:
    _drive_service = None
    print(f"[aviso] Google Drive indisponível ({exc}).")
    print("[aviso] Usando apenas o cache local em drive_cache\\ — reautorize o "
          "token.json para baixar resultados novos.")

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
    if _drive_service is None:
        return False
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


def carregar_resultados(modelo):
    """SEMPRE re-baixa do Drive, para pegar resultados novos a cada execução."""
    local_path = os.path.join(DRIVE_BASE, modelo, "kfold_fold_results.json")
    if not drive_pull(local_path) and not os.path.exists(local_path):
        return []
    with open(local_path, "r", encoding="utf-8") as f:
        return json.load(f) or []


# ══════════════════════════════════════════════════════════════════════════════
# 2. GRÁFICOS
# ══════════════════════════════════════════════════════════════════════════════
def agrupar_por_label(results):
    by_label = defaultdict(list)
    for r in results:
        by_label[r["label"]].append(r)
    return by_label


def moda_int(values):
    if not values:
        return np.nan
    ints = [int(v) for v in values]
    counts = Counter(ints)
    maxc = max(counts.values())
    return sorted(v for v, c in counts.items() if c == maxc)[0]


def plot_curvas_epocas(by_label, modelo):
    gerados = {}
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
        gerados[kind] = fname
        print(f"  [img] {fname}")
    return gerados


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
        return None
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
    return fname


# Acurácias das etapas anteriores do projeto (relatório inicial de IC):
# treino com 24 classes, SNR 20/30 dB.
MODELOS_ETAPA_ANTERIOR = [("MLP", 29.0), ("LightGBM", 44.0), ("CNN 1D", 78.6)]

# Resultados de ASK e PSK publicados no relatório (recuperados de backups
# locais; os JSONs não existem no Drive). Usados APENAS como fallback quando
# o grupo não tem kfold_fold_results.json disponível — se os dados voltarem
# ao Drive, eles têm prioridade e estas linhas são ignoradas.
# (arquitetura, n_params, folds, média, máx, mín, desvio)
RESULTADOS_RELATORIO = {
    "ASK": [
        ("6L_64-64-128-128-256-512_mixpool", 17394435, 5, 97.80, 97.93, 97.73, 0.07),
        ("6L_32-64-64-128-256-512_mixpool", 17334051, 5, 97.71, 97.76, 97.64, 0.04),
        ("5L_32-64-128-256-512", 8941155, 5, 97.70, 97.81, 97.57, 0.08),
        ("5L_64-128-256-512-1024", 18973891, 5, 97.66, 97.82, 97.45, 0.15),
        ("4L_32-64-128-256", 8546403, 5, 97.62, 97.73, 97.57, 0.07),
        ("4L_64-128-256-512", 17397955, 5, 97.62, 97.69, 97.56, 0.05),
        ("4L_128-256-512-512", 18457475, 5, 97.62, 97.69, 97.58, 0.04),
        ("3L_64-128-256", 16920771, 5, 97.09, 97.22, 96.96, 0.10),
        ("3L_128-256-512", 34118019, 5, 97.07, 97.35, 96.90, 0.16),
        ("3L_32-64-128", 8426595, 5, 96.95, 97.19, 96.81, 0.14),
        ("2L_64-128", 16821699, 5, 93.74, 94.30, 93.03, 0.42),
        ("2L_32-64", 8401635, 5, 93.28, 93.95, 92.99, 0.36),
    ],
    "PSK": [
        ("2L_32-64", 8403687, 5, 68.21, 83.51, 57.15, 9.61),
        ("3L_32-64-128", 8428647, 5, 53.26, 84.92, 14.29, 26.55),
        ("4L_32-64-128-256", 8548455, 5, 46.89, 96.00, 14.29, 39.94),
        ("6L_32-64-64-128-256-512_mixpool", 17336103, 5, 42.10, 96.22, 14.29, 34.95),
        ("2L_64-128", 16823751, 5, 36.91, 42.90, 14.29, 11.31),
        ("5L_32-64-128-256-512", 8943207, 5, 30.61, 95.91, 14.29, 32.65),
        ("6L_64-64-128-128-256-512_mixpool", 17396487, 5, 25.71, 71.39, 14.29, 22.84),
        ("3L_128-256-512", 34120071, 5, 20.00, 42.85, 14.29, 11.42),
        ("3L_64-128-256", 16922823, 5, 14.29, 14.29, 14.29, 0.00),
        ("4L_64-128-256-512", 17400007, 5, 14.29, 14.29, 14.29, 0.00),
        ("4L_128-256-512-512", 18459527, 5, 14.29, 14.29, 14.29, 0.00),
        ("5L_64-128-256-512-1024", 18975943, 5, 14.29, 14.29, 14.29, 0.00),
    ],
}

AMARELO = "#FFC828"


def gerar_grafico_comparacao_modelos():
    nomes = [m for m, _ in MODELOS_ETAPA_ANTERIOR]
    accs = [a for _, a in MODELOS_ETAPA_ANTERIOR]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(nomes, accs, height=0.55, color=AMARELO)
    for i, v in enumerate(accs):
        ax.text(v + 1.5, i, f"{v:.1f}%".replace(".", ","),
                va="center", color=CLARO, fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Acurácia no conjunto de teste (%)")
    ax.set_title("Comparação entre modelos — 24 classes, SNR 20/30 dB")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    fname = os.path.join(OUT_DIR, "comparacao_modelos.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  [img] {fname}")
    return fname


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


# ══════════════════════════════════════════════════════════════════════════════
# 3. GERAÇÃO DO .TEX (100% derivado dos dados — sem análise qualitativa)
# ══════════════════════════════════════════════════════════════════════════════
def tex_escape(s):
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")):
        s = s.replace(a, b)
    return s


def fmt_num(v):
    """77.99 -> '77,99' (padrão pt-BR)."""
    return f"{v:.2f}".replace(".", ",")


def fmt_int(v):
    """8402148 -> '8.402.148'."""
    return f"{int(v):,}".replace(",", ".")


def slide_imagem(titulo, img_path):
    rel = os.path.relpath(img_path, BASE_DIR).replace("\\", "/")
    return (
        f"\\begin{{frame}}{{{titulo}}}\n"
        f"  \\centering\n"
        f"  \\includegraphics[width=0.96\\textwidth,height=0.82\\textheight,"
        f"keepaspectratio]{{{rel}}}\n"
        f"\\end{{frame}}\n"
    )


def gerar_tex(status_grupos, imagens_por_grupo, df):
    hoje = datetime.date.today()
    meses = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho",
             "agosto", "setembro", "outubro", "novembro", "dezembro"]
    data_str = f"{hoje.day} de {meses[hoje.month - 1]} de {hoje.year}"

    partes = [r"""% Gerado automaticamente por atualizar_apresentacao.py — NÃO editar à mão.
\documentclass[aspectratio=169,11pt]{beamer}

% ------------------------------------------------
% TEMA MINIMALISTA AMARELO — modo escuro (Variação 4B)
% ------------------------------------------------
\usepackage{xcolor}
\usepackage{colortbl}

\definecolor{myyellow}{RGB}{255,200,40}
\definecolor{grafite}{RGB}{26,26,26}
\definecolor{textoclaro}{RGB}{230,230,230}

\setbeamercolor{background canvas}{bg=grafite}
\setbeamercolor{normal text}{fg=textoclaro}
\setbeamercolor{title}{fg=myyellow}
\setbeamercolor{frametitle}{fg=myyellow}
\setbeamercolor{structure}{fg=myyellow}

\setbeamertemplate{navigation symbols}{}

% Linhas da tabela (booktabs) claras, visíveis no fundo grafite
\arrayrulecolor{textoclaro}

% ------------------------------------------------
% PACOTES
% ------------------------------------------------
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[brazil]{babel}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{adjustbox}

% ------------------------------------------------
% INFO
% ------------------------------------------------
\title{Classificação Automática de Modulações}
\subtitle{Aplicações em Internet das Coisas}
\author{João Marco Pereira}
\institute{CEFET/RJ}
\date{Resultados parciais por grupo de modulação --- """ + data_str + r"""}

\begin{document}

\begin{frame}
  \titlepage
\end{frame}

\begin{frame}{Contexto e motivação}
  \begin{itemize}\setlength{\itemsep}{2mm}
    \item \textbf{Internet das Coisas (IoT):} dispositivos heterogêneos compartilham o espectro de RF usando diferentes esquemas de modulação.
    \item \textbf{Reconhecimento automático de modulação (AMR):} identificar a modulação do sinal recebido --- base para monitoramento espectral, comunicações cognitivas, segurança e gerenciamento de redes.
    \item Abordagens clássicas dependem de extração manual de características e de pressupostos sobre o canal, limitando a generalização.
    \item \textbf{Aprendizado profundo:} extração de características e classificação aprendidas de forma conjunta (fim-a-fim), diretamente das amostras I/Q.
  \end{itemize}
\end{frame}

\begin{frame}{Objetivos}
  \textbf{Objetivo geral:} desenvolver e avaliar modelos de aprendizado de máquina para classificar automaticamente o tipo de modulação de um sinal recebido em ambiente IoT.
  \vspace{3mm}

  \textbf{Objetivos específicos:}
  \begin{itemize}\setlength{\itemsep}{2mm}
    \item estudar técnicas de aprendizado de máquina aplicadas à classificação de sinais de comunicação;
    \item implementar e treinar modelos de aprendizado supervisionado para identificação de modulações;
    \item avaliar o desempenho dos modelos em cenários com ruído e diferentes condições de canal.
  \end{itemize}
\end{frame}

\begin{frame}{Bases de dados consideradas}
  \begin{itemize}\setlength{\itemsep}{3mm}
    \item \textbf{RadioModRec-1:} 15 modulações digitais; canais AWGN, Rayleigh e Rician; SNR de $-20$ a $+20$~dB.
    \item \textbf{AugMod (Pythagore-Mod-Reco):} amostras I/Q em HDF5; 7 classes de modulação e 5 níveis de SNR.
    \item \textbf{DeepSig RadioML 2018.01A} (\emph{selecionada}): $\approx$2,5 milhões de exemplos; 24 modulações analógicas e digitais; séries de 1024 amostras I/Q; SNR de $-20$ a $+30$~dB com efeitos realistas de canal.
  \end{itemize}
\end{frame}

\begin{frame}{Etapas anteriores --- modelos avaliados}
  Treinamento inicial com amostras de SNR 20 e 30~dB (24 classes):
  \vspace{2mm}
  \begin{itemize}\setlength{\itemsep}{2mm}
    \item \textbf{MLP} (vetores 1D das amostras I/Q; \emph{Grid Search}): acurácia $\approx$29\%.
    \item \textbf{LightGBM} (\emph{gradient boosting}; \emph{Randomized Search}): acurácia $\approx$44\%.
    \item \textbf{CNN 1D básica} (séries temporais I/Q): acurácia $\approx$78,6\%.
  \end{itemize}
  \vspace{3mm}
  A CNN motivou a etapa seguinte: análise por grupos de modulação e busca de arquiteturas.
\end{frame}

\begin{frame}{Arquitetura da CNN de referência}
  \centering
  \scriptsize
  \begin{adjustbox}{max width=\textwidth, max totalheight=0.8\textheight}
  \begin{tabular}{lcr}
    \toprule
    \textbf{Camada (tipo)} & \textbf{Dimensão de saída} & \textbf{Parâmetros} \\
    \midrule
    Conv1D (64 filtros)   & (None, 1022, 64) & 448 \\
    MaxPooling1D          & (None, 511, 64)  & 0 \\
    Dropout               & (None, 511, 64)  & 0 \\
    Conv1D (128 filtros)  & (None, 509, 128) & 24.704 \\
    MaxPooling1D          & (None, 254, 128) & 0 \\
    Dropout               & (None, 254, 128) & 0 \\
    Conv1D (256 filtros)  & (None, 252, 256) & 98.560 \\
    MaxPooling1D          & (None, 126, 256) & 0 \\
    Dropout               & (None, 126, 256) & 0 \\
    Flatten               & (None, 32.256)   & 0 \\
    Dense (512 neurônios) & (None, 512)      & 16.515.584 \\
    Dropout               & (None, 512)      & 0 \\
    Dense (24 classes)    & (None, 24)       & 12.312 \\
    \midrule
    \multicolumn{2}{r}{\textbf{Total de parâmetros treináveis}} & \textbf{16.651.608} \\
    \bottomrule
  \end{tabular}
  \end{adjustbox}
\end{frame}
"""]

    # ── Figuras do relatório: o slide só entra se o PNG existir localmente ──
    def figura_relatorio(fname, titulo):
        for d in (BASE_DIR, os.path.join(BASE_DIR, "report_figs")):
            p = os.path.join(d, fname)
            if os.path.exists(p):
                return slide_imagem(titulo, p)
        print(f"  [aviso] figura do relatório ausente: {fname} (slide omitido — "
              f"coloque o PNG em report_figs\\ para incluí-lo)")
        return None

    for fname, titulo in [
        ("CM_MLP.png", "MLP --- matriz de confusão"),
        ("CM_LGBM.png", "LightGBM --- matriz de confusão"),
        ("CM_CNN.png", "CNN --- matriz de confusão"),
    ]:
        s = figura_relatorio(fname, titulo)
        if s:
            partes.append(s)

    partes.append(slide_imagem(
        "Comparação entre modelos (etapas anteriores)",
        os.path.join(OUT_DIR, "comparacao_modelos.png")))

    partes.append(r"""
\begin{frame}{Grupos de modulação (RadioML 2018.01A)}
  \centering
  \begin{tabular}{ll}
    \toprule
    \textbf{Grupo} & \textbf{Modulações} \\
    \midrule
    ASK  & OOK, 4ASK, 8ASK \\
    PSK  & BPSK, QPSK, 8PSK, 16PSK, 32PSK, GMSK, OQPSK \\
    APSK & 16APSK, 32APSK, 64APSK, 128APSK \\
    QAM  & 16QAM, 32QAM, 64QAM, 128QAM, 256QAM \\
    AM   & AM-SSB-WC, AM-SSB-SC, AM-DSB-WC, AM-DSB-SC \\
    FM   & FM \\
    \bottomrule
  \end{tabular}
  \vspace{3mm}

  Filtragem por SNR (0 a $+30$~dB); divisão 60/20/20 (treino/validação/teste) estratificada; \emph{seed} fixa (42).
\end{frame}
""")

    figs_grupos = [
        ("accuracy_vs_snr_individual.png",
         "Modelo combinado (2 estágios) --- acurácia $\\times$ SNR (24 classes)"),
        ("confusion_matrices_all.png",
         "Modelo combinado (2 estágios) --- matriz de confusão (24 classes)"),
    ]
    for g in GRUPOS_ALVO:
        figs_grupos.append((f"avaliacao_2estagios_{g}.png",
                            f"Modelo de 2 estágios --- avaliação do grupo {g}"))
    figs_grupos.append(("avaliacao_2estagios_GERAL.png",
                        "Modelo de 2 estágios --- avaliação geral (6 grupos)"))
    for fname, titulo in figs_grupos:
        s = figura_relatorio(fname, titulo)
        if s:
            partes.append(s)

    partes.append(r"""
\begin{frame}{Metodologia}
  \begin{itemize}\setlength{\itemsep}{2mm}
    \item \textbf{Dataset:} RadioML 2018.01A (sinais I/Q de 1024 amostras, 24 modulações, vários SNRs).
    \item \textbf{Modelo base:} CNN 1D (blocos Conv1d + BatchNorm + ReLU + MaxPool) seguida de classificador denso.
    \item \textbf{Busca de arquitetura:} variação do número de camadas convolucionais e de filtros por camada (ex.: \texttt{2L\_32-64} = 2 camadas com 32 e 64 filtros).
    \item \textbf{Validação:} k-fold com 5 folds por arquitetura; métrica = melhor acurácia de validação de cada fold.
    \item Treino com \emph{early stopping} baseado na paciência do \emph{scheduler} (ReduceLROnPlateau).
    \item Resultados sincronizados via Google Drive e analisados em modo somente leitura.
  \end{itemize}
\end{frame}
""")

    # ── Status geral (tabela calculada a partir dos dados) ──
    linhas_status = []
    tem_fallback = any(st.get("relatorio") for st in status_grupos.values())
    for modelo in GRUPOS_ALVO:
        st = status_grupos.get(modelo)
        if st is None:
            linhas_status.append(
                f"    {modelo} & --- & --- & "
                f"\\multicolumn{{3}}{{l}}{{sem resultados até o momento}} \\\\"
            )
        else:
            nome = modelo + ("*" if st.get("relatorio") else "")
            melhor = st["melhor"]
            linhas_status.append(
                f"    {nome} & {st['n_arq']} & {st['n_folds']} & "
                f"\\texttt{{{tex_escape(melhor['arquitetura'])}}} & "
                f"{fmt_num(melhor['media_acc'])} & {fmt_num(melhor['max_acc'])} \\\\"
            )
    nota = ("  \\par\\vspace{2mm}{\\scriptsize * dados do relatório "
            "(backups locais); não disponíveis no Drive.}\n"
            if tem_fallback else "")
    partes.append(
        "\n\\begin{frame}{Status geral da busca}\n"
        "  \\centering\n  \\small\n"
        "  \\begin{adjustbox}{max width=\\textwidth}\n"
        "  \\begin{tabular}{lrrlrr}\n    \\toprule\n"
        "    \\textbf{Grupo} & \\textbf{Arqs.} & \\textbf{Folds} & "
        "\\textbf{Melhor arquitetura} & \\textbf{Média (\\%)} & \\textbf{Máx (\\%)} \\\\\n"
        "    \\midrule\n"
        + "\n".join(linhas_status)
        + "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\end{adjustbox}\n" + nota + "\\end{frame}\n"
    )

    # ── Slides de gráficos, por grupo (só os que existem) ──
    for modelo in GRUPOS_ALVO:
        imgs = imagens_por_grupo.get(modelo, {})
        marca = (" (figura do relatório)"
                 if any("report_figs" in p for p in imgs.values()) else "")
        if imgs.get("best"):
            partes.append(slide_imagem(
                f"{modelo} --- acurácia por época "
                f"(melhor fold de cada arquitetura){marca}",
                imgs["best"]))
        if imgs.get("worst"):
            partes.append(slide_imagem(
                f"{modelo} --- acurácia por época "
                f"(pior fold de cada arquitetura){marca}",
                imgs["worst"]))
        if imgs.get("params"):
            partes.append(slide_imagem(
                f"{modelo} --- número de parâmetros $\\times$ acurácia média{marca}",
                imgs["params"]))

    # ── Tabela resumo (paginada se crescer) ──
    if not df.empty:
        LINHAS_POR_SLIDE = 12
        blocos = [df.iloc[i:i + LINHAS_POR_SLIDE]
                  for i in range(0, len(df), LINHAS_POR_SLIDE)]
        for bi, bloco in enumerate(blocos):
            sufixo = f" ({bi + 1}/{len(blocos)})" if len(blocos) > 1 else ""
            linhas_tex = []
            for _, ln in bloco.iterrows():
                linhas_tex.append(
                    f"    {ln['modelo']} & \\texttt{{{tex_escape(str(ln['arquitetura']))}}} & "
                    f"{fmt_int(ln['n_params'])} & {int(ln['folds_concluidos'])} & "
                    f"{fmt_num(ln['media_acc'])} & {fmt_num(ln['max_acc'])} & "
                    f"{fmt_num(ln['min_acc'])} & {fmt_num(ln['std_acc'])} \\\\"
                )
            partes.append(
                f"\n\\begin{{frame}}{{Resumo dos resultados por arquitetura{sufixo}}}\n"
                "  \\centering\n  \\small\n"
                "  \\begin{adjustbox}{max width=\\textwidth}\n"
                "  \\begin{tabular}{llrrrrrr}\n    \\toprule\n"
                "    Grupo & Arquitetura & Parâmetros & Folds & Média (\\%) & "
                "Máx (\\%) & Mín (\\%) & Desv.\\ (p.p.) \\\\\n    \\midrule\n"
                + "\n".join(linhas_tex)
                + "\n    \\bottomrule\n  \\end{tabular}\n"
                "  \\end{adjustbox}\n"
                + ("  \\par\\vspace{2mm}{\\scriptsize * dados do relatório "
                   "(backups locais); não disponíveis no Drive.}\n"
                   if bloco["modelo"].str.contains("*", regex=False).any() else "")
                + "\\end{frame}\n"
            )

    partes.append("\n\\end{document}\n")

    with open(TEX_PATH, "w", encoding="utf-8") as f:
        f.write("".join(partes))
    print(f"\n[tex] {TEX_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. EXECUÇÃO
# ══════════════════════════════════════════════════════════════════════════════
tabela_geral = []
imagens_por_grupo = {}
status_grupos = {}

print("\n=== Figuras estáticas ===")
gerar_grafico_comparacao_modelos()

for modelo in GRUPOS_ALVO:
    print(f"\n=== {modelo} ===")
    resultados = carregar_resultados(modelo)
    if not resultados:
        # Fallback: figuras extraídas do relatório + números publicados nele
        fb = RESULTADOS_RELATORIO.get(modelo)
        imgs_fb = {}
        for kind, fname in (("best", f"{modelo}_epochs_best_fold_light.png"),
                            ("worst", f"{modelo}_epochs_worst_fold_light.png"),
                            ("params", f"{modelo}_params_vs_acc_light.png")):
            p = os.path.join(BASE_DIR, "report_figs", fname)
            if os.path.exists(p):
                imgs_fb[kind] = p
        if not fb and not imgs_fb:
            print(f"  [aviso] nenhum resultado encontrado para '{modelo}', pulando")
            continue
        print(f"  [fallback] '{modelo}' sem dados no Drive; usando "
              f"figuras/números do relatório")
        if imgs_fb:
            imagens_por_grupo[modelo] = imgs_fb
        if fb:
            linhas = [{
                "modelo": modelo + "*",
                "arquitetura": arq, "n_params": npar, "folds_concluidos": fl,
                "media_acc": me, "max_acc": mx, "min_acc": mn,
                "moda_acc_int": "", "std_acc": sd,
            } for (arq, npar, fl, me, mx, mn, sd) in fb]
            tabela_geral.extend(linhas)
            melhor = max(linhas, key=lambda l: l["media_acc"])
            status_grupos[modelo] = {
                "n_arq": len(linhas),
                "n_folds": sum(l["folds_concluidos"] for l in linhas),
                "melhor": melhor,
                "relatorio": True,
            }
        continue
    by_label = agrupar_por_label(resultados)
    imgs = plot_curvas_epocas(by_label, modelo)
    params_img = plot_params_vs_acc(by_label, modelo)
    if params_img:
        imgs["params"] = params_img
    imagens_por_grupo[modelo] = imgs

    linhas = montar_tabela(by_label, modelo)
    tabela_geral.extend(linhas)
    if linhas:
        melhor = max(linhas, key=lambda l: l["media_acc"])
        status_grupos[modelo] = {
            "n_arq": len(linhas),
            "n_folds": sum(l["folds_concluidos"] for l in linhas),
            "melhor": melhor,
        }

if tabela_geral:
    df = pd.DataFrame(tabela_geral).sort_values(["modelo", "media_acc"],
                                                ascending=[True, False])
    csv_path = os.path.join(OUT_DIR, "tabela_resultados.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[csv] {csv_path}")
else:
    df = pd.DataFrame()
    print("\nNenhum dado encontrado em nenhum grupo.")

gerar_tex(status_grupos, imagens_por_grupo, df)

# ── Compilação (2 passadas) ──
for passada in (1, 2):
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-enable-installer",
         os.path.basename(TEX_PATH)],
        cwd=BASE_DIR, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        print(f"\n[erro] pdflatex falhou na passada {passada}:")
        print("\n".join(proc.stdout.splitlines()[-30:]))
        sys.exit(1)

pdf_path = TEX_PATH.replace(".tex", ".pdf")
print(f"[pdf] {pdf_path}")
print("\nApresentação atualizada com sucesso.")
