"""Sincroniza os resultados da campanha com o Google Drive, RODANDO LOCAL.

POR QUE ISSO NAO ROda NA VM
    Nao e limitacao do Drive -- varias VMs conseguiriam se conectar. O
    problema e o que isso exigiria: subir um token de escopo
    "https://www.googleapis.com/auth/drive" (leitura E escrita em TODO o
    Drive da conta) para 4 maquinas remotas de terceiros, que caem sozinhas e
    cujo disco nao controlamos. O token fica nesta maquina, ponto. As VMs
    gravam local, o painel espelha para ca, e este script -- e so ele --
    fala com o Drive.

LAYOUT, IDENTICO AO DO drive_push DO NOTEBOOK
    O notebook enviava DRIVE_BASE/<relativo> para DRIVE_FOLDER_ID/<relativo>.
    Aqui a raiz local passa a ser campanha/resultados, entao:

        campanha/resultados/QAM/kfold_fold_results.json
            -> Drive: <DRIVE_FOLDER_ID>/QAM/kfold_fold_results.json

    Mesma semantica de criar-ou-atualizar, mesma arvore de pastas. Um arquivo
    enviado por aqui e indistinguivel de um enviado pelo notebook.

USO
    python campanha/sincroniza_drive.py --once        # uma passada e sai
    python campanha/sincroniza_drive.py               # vigia a cada 5 min
    python campanha/sincroniza_drive.py --seco        # mostra sem enviar

Se o token expirar sem refresh possivel, regenere com:
    python C:/Users/Administrador/Documents/coding/ml_teste/gerar_token_drive.py
"""
import argparse
import hashlib
import json
import os
import sys
import time

DRIVE_FOLDER_ID = "1E2bJyP18S4xq4OhBbvgryJ0Oc5Hm_tv2"   # mesmo do notebook
SCOPES = ["https://www.googleapis.com/auth/drive"]

AQUI = os.path.dirname(os.path.abspath(__file__))
RESULTADOS = os.path.join(AQUI, "resultados")
CRED_PADRAO = r"C:/Users/Administrador/Documents/coding/ml_teste"
ESTADO = os.path.join(RESULTADOS, ".sincronizado.json")

# artefatos que valem o upload. _env.py e ruido de orquestracao; nada de
# credencial entra nesta lista por construcao.
PADROES = ("kfold_", "escolhe_lr", "runner_", "prep.log")
EXT_OK = (".json", ".npy", ".log", ".csv")

_p = argparse.ArgumentParser(description="Espelha campanha/resultados no Drive")
_p.add_argument("--once", action="store_true", help="uma passada e encerra")
_p.add_argument("--intervalo", type=int, default=300, help="segundos entre passadas")
_p.add_argument("--seco", action="store_true", help="mostra o que faria, sem enviar")
_p.add_argument("--credenciais", default=CRED_PADRAO,
                help="pasta com token.json e client_secret.json")
_p.add_argument("--raiz", default=RESULTADOS, help="raiz local a espelhar")
_A = _p.parse_args()


def _servico():
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    tok = os.path.join(_A.credenciais, "token.json")
    if not os.path.exists(tok):
        sys.exit("token.json nao encontrado em %s\nGere com gerar_token_drive.py"
                 % _A.credenciais)
    creds = Credentials.from_authorized_user_file(tok, SCOPES)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(tok, "w") as f:
                f.write(creds.to_json())
            print("[drive] token renovado pelo refresh_token")
        else:
            sys.exit("token invalido e sem refresh. Rode gerar_token_drive.py de novo.")
    return build("drive", "v3", credentials=creds)


_cache_pastas = {}


def _pasta(svc, rel_dir):
    """Resolve (criando se preciso) a pasta relativa. Mesma logica do notebook."""
    parent = DRIVE_FOLDER_ID
    if not rel_dir or rel_dir == ".":
        return parent
    parcial = ""
    for parte in rel_dir.replace("\\", "/").split("/"):
        if not parte:
            continue
        parcial = "%s/%s" % (parcial, parte) if parcial else parte
        if parcial in _cache_pastas:
            parent = _cache_pastas[parcial]
            continue
        q = ("'%s' in parents and name = '%s' and "
             "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
             % (parent, parte))
        achados = svc.files().list(q=q, fields="files(id, name)").execute().get("files", [])
        if achados:
            fid = achados[0]["id"]
        else:
            fid = svc.files().create(
                body={"name": parte, "mimeType": "application/vnd.google-apps.folder",
                      "parents": [parent]}, fields="id").execute()["id"]
        _cache_pastas[parcial] = fid
        parent = fid
    return parent


def _envia(svc, caminho, rel):
    from googleapiclient.http import MediaFileUpload
    rel_dir, nome = os.path.dirname(rel), os.path.basename(rel)
    parent = _pasta(svc, rel_dir)
    q = "'%s' in parents and name = '%s' and trashed = false" % (parent, nome)
    achados = svc.files().list(q=q, fields="files(id)").execute().get("files", [])
    media = MediaFileUpload(caminho, resumable=True)
    if achados:
        svc.files().update(fileId=achados[0]["id"], media_body=media).execute()
        return "atualizado"
    svc.files().create(body={"name": nome, "parents": [parent]},
                       media_body=media, fields="id").execute()
    return "criado"


def _digest(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _candidatos():
    for raiz, _dirs, arqs in os.walk(_A.raiz):
        for a in sorted(arqs):
            if not a.endswith(EXT_OK) or not any(a.startswith(p) for p in PADROES):
                continue
            caminho = os.path.join(raiz, a)
            yield caminho, os.path.relpath(caminho, _A.raiz).replace("\\", "/")


def passada(svc, estado):
    enviados = pulados = 0
    for caminho, rel in _candidatos():
        d = _digest(caminho)
        if estado.get(rel) == d:
            pulados += 1
            continue
        if _A.seco:
            print("  [seco] enviaria %s (%d bytes)" % (rel, os.path.getsize(caminho)))
            enviados += 1
            continue
        try:
            acao = _envia(svc, caminho, rel)
            estado[rel] = d
            print("  %-9s %s" % (acao, rel))
            enviados += 1
        except Exception as e:
            print("  FALHOU   %s -> %s: %s" % (rel, type(e).__name__, e))
    if not _A.seco:
        os.makedirs(os.path.dirname(ESTADO), exist_ok=True)
        json.dump(estado, open(ESTADO, "w"), indent=1)
    print("[drive] %d enviado(s), %d ja em dia  (%s)"
          % (enviados, pulados, time.strftime("%H:%M:%S")))
    return enviados


def main():
    if not os.path.isdir(_A.raiz):
        sys.exit("raiz local nao existe: %s" % _A.raiz)
    estado = {}
    if os.path.exists(ESTADO):
        try:
            estado = json.load(open(ESTADO))
        except Exception:
            estado = {}
    svc = None if _A.seco else _servico()
    print("[drive] raiz local : %s" % _A.raiz)
    print("[drive] destino    : Drive/%s/<grupo>/<arquivo>" % DRIVE_FOLDER_ID)
    print("[drive] modo       : %s" % ("passada unica" if _A.once else
                                       "vigia a cada %ds" % _A.intervalo))
    while True:
        passada(svc, estado)
        if _A.once:
            return
        try:
            time.sleep(_A.intervalo)
        except KeyboardInterrupt:
            print("\n[drive] encerrado pelo usuario")
            return


if __name__ == "__main__":
    main()
