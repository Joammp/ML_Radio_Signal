"""Painel da campanha de HP -- orquestra as VMs do Colab a partir DESTA maquina.

Roda local: `python campanha/painel.py`

Um job por grupo de modulacao, cada um na sua VM do Colab, todos em paralelo.
Log ao vivo, progresso por fold, selecao de GPU e espelho local do resultado.

LIMITES DE CONCORRENCIA, MEDIDOS EM 21/08/2026
    L4  -> no maximo 2 sessoes simultaneas (a 3a da "Precondition Failed")
    T4  -> no maximo 2 sessoes simultaneas
    O limite e POR ACELERADOR, nao global: 2xL4 + 2xT4 convivem numa conta.
Por isso a campanha roda 4 grupos em paralelo, e nao 1 como o README supunha.
O painel recusa passar do teto para nao queimar tentativas.

ATRIBUICAO PADRAO DE GPU
    Os grupos pesados vao para as L4 (mais rapidas) e os leves para as T4,
    equilibrando o relogio. Custo estimado por grupo, do README da campanha:
        PSK ~94 h  >  QAM ~67 h  >  APSK ~53 h  >  ASK ~40 h

O QUE O PAINEL NAO FAZ
    Upload ao Drive. A pasta do projeto tem client_secret.json e token.json,
    e ligar credencial de escopo `drive` numa app sem o dono pedir seria
    passar do combinado. O espelho local ja deixa tudo em disco; o envio ao
    Drive fica como proximo passo, deliberadamente.
"""
import json
import os
import queue
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk

# ── caminhos ──────────────────────────────────────────────────────────────────
AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
COLAB = os.path.join(os.path.expanduser("~"), ".colab-cli", "Scripts", "colab.exe")
RESULTADOS = os.path.join(AQUI, "resultados")
RUNNER = os.path.join(AQUI, "runner_vm.py")
RESNET_PY = os.path.join(AQUI, "resnet.py")
# onde o busca_hp grava/le os artefatos DENTRO da VM (o DRIVE_BASE dele)
_DRIVE_VM = "/content/drive_cache/radioml_sessions/%s"


# perfil -> sufixo dos artefatos. A CNN fica sem sufixo (caminho historico,
# para os folds ja concluidos continuarem validos); os demais se separam.
SUFIXO = {"campanha": "", "teste": "", "resnet": "_resnet", "reducao": "_reducao"}


def _dir_vm(grupo, perfil):
    """Mesma regra de _drive_dir no busca_hp: cada perfil tem seu diretorio,
    para buscas diferentes nunca compartilharem kfold_fold_results.json."""
    return _DRIVE_VM % (grupo + SUFIXO.get(perfil or "campanha", ""))

# ── limites medidos ───────────────────────────────────────────────────────────
# Teto de sessoes simultaneas POR ACELERADOR (nao e global: 2xL4 + 2xT4
# convivem). L4 e T4 foram MEDIDOS em 21/08/2026 -- a 3a sessao da
# "Precondition Failed". Os demais nao foram medidos e usam o mesmo palpite;
# se o teto real for menor, o painel mostra "sem cota de X" ao tentar, que e
# a resposta honesta do servidor.
TETO_GPU = {"L4": 2, "T4": 2, "A100": 2, "H100": 2, "G4": 2}
MEDIDOS = ("L4", "T4")
# Ciclos de ~60 s entre espelhamentos do checkpoint. O valor existe porque o
# checkpoint da CNN e enorme (200 MB na 2L_32-64: 8,4 M de parametros, quase
# todos na densa). Medidos em 22/08/2026: ResNet 3 MB, busca de reducao 0,6 MB
# -- nesses perfis esperar 5 ciclos so aumenta o que se perde numa queda.
CICLOS_CKPT = {"resnet": 1, "reducao": 1}
CICLOS_CKPT_PADRAO = 5
# Execucao perene. O Colab encerra a sessao num teto RIGIDO de ~63 min --
# medido em 77 sessoes deste log, agrupadas em 62-64 min, independente de GPU,
# regiao, credito pago ou atividade no kernel. Nao ha como evitar; da para
# tornar barato. Com checkpoint a cada 5 epocas, cada queda custa <= 5 epocas.
PERENE_ESPERA_S = 45     # respiro antes de pedir VM nova
PERENE_MIN_VIDA_S = 600  # abaixo disto a tentativa conta como fracasso rapido
PERENE_MAX_SEGUIDAS = 5  # fracassos rapidos seguidos -> desiste e avisa
GPUS = ["L4", "T4", "A100", "H100", "G4"]

# ── perfis de execucao ────────────────────────────────────────────────────────
# 'campanha' e o trabalho real. 'teste' exercita exatamente o mesmo caminho
# (sessao, upload, runner, log, espelho) com um payload de ~2 min, para validar
# a orquestracao sem esperar um fold de 40-90 min.
PERFIS = {
    "campanha": {
        "alvo": os.path.join(AQUI, "busca_hp.py"),
        "envvar": "BUSCA_HP_ARGS",
        "args": "--grupos {grupo} --lr 4.5e-5",
        "prep": False,          # busca_hp baixa o dado sozinho via kagglehub
        "total_folds": 60,      # 12 arquiteturas x 5 folds
    },
    # Busca de REDUCAO: arquitetura convolucional fixa (2L_32-64) e o que varia
    # e o que chega na densa -- onde estao 99,9% dos parametros. 8 candidatas,
    # da menor (45.795) para a maior (8.401.635, o baseline atual).
    "reducao": {
        "alvo": os.path.join(AQUI, "busca_hp.py"),
        "envvar": "BUSCA_HP_ARGS",
        "args": "--grupos {grupo} --lr 4.5e-4 --modelo reducao"
                "  --epochs 1000 --ckpt-every 1",
        "prep": False,
        "total_folds": 40,     # 8 candidatas x 5 folds
    },
    # A ResNet do artigo e arquitetura FIXA: 5 folds por alvo, sem busca.
    "resnet": {
        "alvo": os.path.join(AQUI, "busca_hp.py"),
        "envvar": "BUSCA_HP_ARGS",
        "args": "--grupos {grupo} --lr 4.5e-4 --modelo resnet"
                "  --epochs 1000 --ckpt-every 1",
        "prep": False,
        "total_folds": 5,      # 1 arquitetura x 5 folds
    },
    # 'teste' usa um payload sintetico com a MESMA forma do busca_hp (sincrono,
    # so print, grava kfold_fold_results.json fold a fold). Nao serve para
    # escolhe_lr.py, que sobe a propria thread e retornaria na hora -- o runner
    # marcaria "concluido" antes de o treino comecar.
    "teste": {
        "alvo": os.path.join(AQUI, "teste_payload.py"),
        "envvar": "BUSCA_HP_ARGS",
        "args": "--grupos {grupo} --folds 5 --seg-por-fold 15",
        "prep": False,
        "total_folds": 5,
    },
}

# Alvos. AM e FM sairam do estudo (analogicas; o FM ainda tem classe unica).
#   GROUP  -> 19 digitais rotuladas pelo grupo   (4 classes, 1o estagio)
#   TODAS  -> 19 digitais rotuladas pela modulacao (19 classes) -- so ResNet
GRUPOS = [("PSK", "L4"), ("QAM", "L4"), ("APSK", "T4"), ("ASK", "T4"),
          ("GROUP", "L4"), ("TODAS", "L4")]
SO_RESNET = ("TODAS",)   # a CNN nao roda a tarefa completa; ver perfil resnet


def _cli(*args, timeout=900):
    """Chama o colab-cli. Devolve (ok, saida).

    Duas armadilhas ja pagas:
      - `colab download` retorna exit 0 MESMO FALHANDO; so da para saber pelo
        texto da saida ou pelo arquivo. Nunca confie no codigo de retorno.
      - MSYS_NO_PATHCONV=1 protege contra o Git Bash reescrever "/content/x"
        como "C:/Program Files/Git/content/x". Chamando por subprocess sem
        shell isso nao ocorre, mas fica por seguranca se alguem embrulhar.
    """
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    try:
        p = subprocess.run([COLAB] + list(args), capture_output=True, text=True,
                           timeout=timeout, env=env,
                           encoding="utf-8", errors="replace",
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        return False, "[painel] timeout em: colab %s" % " ".join(args)
    saida = (p.stdout or "") + (p.stderr or "")
    low = saida.lower()
    # NAO confundir "File or directory not found" (arquivo ainda nao existe na
    # VM, situacao normal antes do 1o fold) com a sessao ter morrido. Perda de
    # sessao sempre nomeia a sessao; falta de arquivo, nunca.
    perdida = ("appears to be lost" in low
               or ("session" in low and "not found" in low)
               or "404/401" in low)
    return (not perdida and p.returncode == 0), saida, perdida


def sessoes_ativas():
    ok, saida, _ = _cli("sessions", timeout=180)
    linhas = [l for l in saida.splitlines() if l.strip().startswith("[")
              and "No active sessions" not in l]
    return linhas


class Job(object):
    def __init__(self, grupo, gpu):
        self.grupo = grupo
        self.gpu = gpu
        # O nome da sessao inclui o perfil: sem isso um job ResNet do ASK e um
        # job CNN do ASK disputariam a MESMA sessao no Colab, e o segundo
        # sequestraria a VM do primeiro. Ver sessao(), que le job.perfil.
        self._sessao_base = "cmp_%s" % grupo.lower()
        self.estado = "parado"
        self.folds = 0
        self.total = 0
        self.inicio = None
        self.thread = None
        self.perfil = None      # capturado na thread da UI; ver alterna()
        self.tentativas = 0     # fracassos RAPIDOS seguidos (execucao perene)
        self.parar = threading.Event()
        self.log = ""

    @property
    def sessao(self):
        suf = SUFIXO.get(self.perfil or "campanha", "")
        return self._sessao_base + suf.replace("_resnet", "_rn").replace(
            "_reducao", "_rd")

    @property
    def dir_local(self):
        return os.path.join(RESULTADOS, self.grupo + SUFIXO.get(self.perfil or "campanha", ""))


class Painel(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Campanha de HP — RadioML 2018.01A")
        self.geometry("1080x680")
        self.minsize(900, 560)
        self.fila = queue.Queue()
        self.jobs = {g: Job(g, gpu) for g, gpu in GRUPOS}
        self.perfil = tk.StringVar(value="campanha")
        self.perene = tk.BooleanVar(value=False)
        self.grupo_log = tk.StringVar(value=GRUPOS[0][0])
        self._monta()
        self.after(300, self._drena)
        self.protocol("WM_DELETE_WINDOW", self._fechar)

    # ── interface ─────────────────────────────────────────────────────────────
    def _monta(self):
        topo = ttk.Frame(self, padding=8)
        topo.pack(fill="x")
        ttk.Label(topo, text="Perfil:").pack(side="left")
        cb = ttk.Combobox(topo, textvariable=self.perfil, values=list(PERFIS),
                          state="readonly", width=11)
        cb.pack(side="left", padx=(4, 14))
        ttk.Button(topo, text="Iniciar todos", command=self.iniciar_todos).pack(side="left")
        ttk.Button(topo, text="Parar todos", command=self.parar_todos).pack(side="left", padx=6)
        ttk.Button(topo, text="Limpar GPUs orfas", command=self.limpar_orfas).pack(side="left")
        ttk.Button(topo, text="Abrir resultados", command=self.abrir_resultados).pack(side="left", padx=6)
        ttk.Checkbutton(topo, text="Execução perene",
                        variable=self.perene).pack(side="left", padx=(10, 0))
        self.lbl_teto = ttk.Label(topo, text="")
        self.lbl_teto.pack(side="right")

        grade = ttk.LabelFrame(self, text="Jobs", padding=8)
        grade.pack(fill="x", padx=8)
        cols = ("Grupo", "GPU", "Estado", "Sessao", "Progresso", "Folds", "Tempo", "")
        for c, t in enumerate(cols):
            ttk.Label(grade, text=t, font=("TkDefaultFont", 9, "bold")).grid(
                row=0, column=c, sticky="w", padx=6, pady=(0, 4))
        self.w = {}
        for r, (g, _) in enumerate(GRUPOS, start=1):
            job = self.jobs[g]
            ttk.Label(grade, text=g, width=7).grid(row=r, column=0, sticky="w", padx=6)
            var = tk.StringVar(value=job.gpu)
            cbg = ttk.Combobox(grade, textvariable=var, values=GPUS,
                               state="readonly", width=5)
            cbg.grid(row=r, column=1, padx=6)
            cbg.bind("<<ComboboxSelected>>",
                     lambda e, gg=g, vv=var: self._muda_gpu(gg, vv.get()))
            est = ttk.Label(grade, text="parado", width=16)
            est.grid(row=r, column=2, sticky="w", padx=6)
            ses = ttk.Label(grade, text="-", width=14)
            ses.grid(row=r, column=3, sticky="w", padx=6)
            pb = ttk.Progressbar(grade, length=130, mode="determinate")
            pb.grid(row=r, column=4, padx=6)
            fol = ttk.Label(grade, text="0", width=9)
            fol.grid(row=r, column=5, sticky="w", padx=6)
            tmp = ttk.Label(grade, text="-", width=9)
            tmp.grid(row=r, column=6, sticky="w", padx=6)
            bt = ttk.Button(grade, text="Iniciar", width=9,
                            command=lambda gg=g: self.alterna(gg))
            bt.grid(row=r, column=7, padx=6, pady=2)
            self.w[g] = {"gpu": var, "cbg": cbg, "estado": est, "sessao": ses,
                         "pb": pb, "folds": fol, "tempo": tmp, "bt": bt}

        sel = ttk.Frame(self, padding=(8, 6, 8, 0))
        sel.pack(fill="x")
        ttk.Label(sel, text="Log ao vivo:").pack(side="left")
        for g, _ in GRUPOS:
            ttk.Radiobutton(sel, text=g, value=g, variable=self.grupo_log,
                            command=self._mostra_log).pack(side="left", padx=3)

        quadro = ttk.Frame(self, padding=8)
        quadro.pack(fill="both", expand=True)
        self.txt = tk.Text(quadro, wrap="none", font=("Consolas", 9),
                           bg="#101418", fg="#d8e0e8", insertbackground="#d8e0e8")
        sb = ttk.Scrollbar(quadro, command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.txt.pack(side="left", fill="both", expand=True)

        self.status = ttk.Label(self, text="pronto", relief="sunken", anchor="w")
        self.status.pack(fill="x", side="bottom")
        self._atualiza_teto()

    def _muda_gpu(self, grupo, gpu):
        job = self.jobs[grupo]
        if job.estado != "parado":
            self.w[grupo]["gpu"].set(job.gpu)
            self._status("%s esta rodando; pare antes de trocar a GPU." % grupo)
            return
        job.gpu = gpu
        self._atualiza_teto()

    def _atualiza_teto(self):
        # conta apenas o que esta NO AR: sao 6 alvos para 4 vagas, entao
        # contar linhas da tabela acusaria estouro permanente.
        cont = {}
        for g, _ in GRUPOS:
            j = self.jobs[g]
            if j.estado != "parado":
                cont[j.gpu] = cont.get(j.gpu, 0) + 1
        partes, estouro = [], False
        for gpu in GPUS:
            n, teto = cont.get(gpu, 0), TETO_GPU[gpu]
            if n:                       # so o que esta em uso, senao polui
                partes.append("%s %d/%d%s" % (gpu, n, teto,
                                              "" if gpu in MEDIDOS else "?"))
            if n > teto:
                estouro = True
        self.lbl_teto.config(text="teto de sessoes:  " + "   ".join(partes),
                             foreground="#b00020" if estouro else "")
        return not estouro

    def _status(self, msg):
        self.status.config(text=msg)

    def _mostra_log(self):
        job = self.jobs[self.grupo_log.get()]
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", job.log or "(sem log ainda)")
        self.txt.see("end")

    def abrir_resultados(self):
        os.makedirs(RESULTADOS, exist_ok=True)
        os.startfile(RESULTADOS)

    # ── controle ──────────────────────────────────────────────────────────────
    def iniciar_todos(self):
        if not self._atualiza_teto():
            self._status("Acima do teto de sessoes. Redistribua as GPUs antes.")
            return
        for g, _ in GRUPOS:
            if self.jobs[g].estado == "parado":
                self.alterna(g)
                time.sleep(0.4)     # nao dispara 4 `colab new` no mesmo instante

    def parar_todos(self):
        for g, _ in GRUPOS:
            if self.jobs[g].estado != "parado":
                self.alterna(g)

    def alterna(self, grupo):
        job = self.jobs[grupo]
        if job.estado == "parado":
            perfil = self.perfil.get()
            if grupo in SO_RESNET and perfil != "resnet":
                self._status("%s tem 19 classes e so roda no perfil resnet."
                             % grupo)
                return
            ativos = sum(1 for g, _ in GRUPOS
                         if self.jobs[g].estado != "parado"
                         and self.jobs[g].gpu == job.gpu)
            if ativos >= TETO_GPU[job.gpu]:
                self._status("%s ja tem %d sessao(oes) no ar (teto %d)."
                             % (job.gpu, ativos, TETO_GPU[job.gpu]))
                return
            job.parar.clear()
            job.inicio = time.time()
            # ler uma StringVar do Tk fora da thread principal levanta
            # RuntimeError; captura aqui, onde ainda estamos na thread da UI.
            job.perfil = self.perfil.get()
            job.thread = threading.Thread(target=self._worker, args=(job,),
                                          daemon=True, name="painel_%s" % grupo)
            job.thread.start()
            self.w[grupo]["bt"].config(text="Parar")
        else:
            job.parar.set()
            self.w[grupo]["bt"].config(text="Iniciar")
            self._evento(job, estado="parando...")

    def limpar_orfas(self):
        def _t():
            py = os.path.join(os.path.expanduser("~"), ".colab-cli", "Scripts", "python.exe")
            ferr = os.path.join(AQUI, "ferramentas", "unassign_orfa.py")
            try:
                p = subprocess.run([py, ferr], capture_output=True, text=True, timeout=300)
                self.fila.put(("status", (p.stdout or p.stderr or "").strip().splitlines()[-1:]))
            except Exception as e:
                self.fila.put(("status", ["falha ao limpar orfas: %s" % e]))
        threading.Thread(target=_t, daemon=True).start()
        self._status("liberando atribuicoes de GPU orfas...")

    # ── worker ────────────────────────────────────────────────────────────────
    def _evento(self, job, **kw):
        self.fila.put(("job", (job.grupo, kw)))

    def _worker(self, job):
        perfil = PERFIS[job.perfil]
        job.total = perfil["total_folds"]
        os.makedirs(job.dir_local, exist_ok=True)
        log_vm = "/content/runner_%s.log" % job.grupo
        try:
            self._evento(job, estado="conferindo cota")
            no_ar = [l for l in sessoes_ativas() if job.gpu.lower() in l.lower()]
            if len(no_ar) >= TETO_GPU[job.gpu]:
                self._append(job, "[painel] %s ocupada no SERVIDOR (%d/%d):%s%s"
                             % (job.gpu, len(no_ar), TETO_GPU[job.gpu],
                                chr(10), chr(10).join(no_ar)))
                self._evento(job, estado="sem cota de %s" % job.gpu, fim=True)
                return
            self._evento(job, estado="criando sessao", sessao=job.sessao)
            ok, s, _ = _cli("new", "-s", job.sessao, "--gpu", job.gpu, timeout=600)
            if not ok:
                if "Precondition Failed" in s:
                    self._evento(job, estado="sem cota de %s" % job.gpu, fim=True)
                else:
                    self._evento(job, estado="falha ao criar", fim=True)
                self._append(job, s)
                return
            if job.parar.is_set():
                return self._encerra(job)

            self._evento(job, estado="enviando scripts")
            # resnet.py vai junto: o busca_hp o carrega de /content quando
            # roda no perfil resnet.
            envios = [(perfil["alvo"], "/content/" + os.path.basename(perfil["alvo"])),
                      (RUNNER, "/content/runner_vm.py")]
            if os.path.exists(RESNET_PY):
                envios.append((RESNET_PY, "/content/resnet.py"))
            for local, remoto in envios:
                ok, s, _ = _cli("upload", "-s", job.sessao, local, remoto, timeout=600)
                if not ok:
                    self._evento(job, estado="falha no upload", fim=True)
                    self._append(job, s)
                    return

            if perfil["prep"]:
                self._evento(job, estado="preparando dado")
                ok, s, _ = _cli("exec", "-s", job.sessao, "-f",
                                os.path.join(AQUI, "prep_all.py"), timeout=900)
                self._append(job, s)
                if not self._espera(job, "/content/prep.log", "PREP_ALL_PRONTO", 40):
                    self._evento(job, estado="prep falhou", fim=True)
                    return

            if job.parar.is_set():
                return self._encerra(job)

            self._evento(job, estado="lancando job")
            env_py = os.path.join(job.dir_local, "_env.py")
            with open(env_py, "w", encoding="utf-8") as f:
                f.write("import os as _o\n")
                for k, v in (("RUNNER_GRUPO", job.grupo),
                             ("RUNNER_ALVO", "/content/" + os.path.basename(perfil["alvo"])),
                             ("RUNNER_ARGS", perfil["args"].format(grupo=job.grupo)),
                             ("RUNNER_ENVVAR", perfil["envvar"])):
                    f.write('_o.environ["%s"] = %r\n' % (k, v))
                f.write("_o.makedirs(%r, exist_ok=True)\n" % _dir_vm(job.grupo, job.perfil))
                f.write('print("[env] pronto para %s")\n' % job.grupo)
            ok, s, _ = _cli("exec", "-s", job.sessao, "-f", env_py, timeout=600)
            self._append(job, s)

            # RETOMADA ENTRE VMs. O busca_hp le os folds concluidos de
            # /content/drive_cache/radioml_sessions/<G>/kfold_fold_results.json.
            # Numa VM nova esse caminho esta vazio, e a unica recuperacao
            # prevista era drive_pull() -- desativado de proposito, para o
            # token de escopo `drive` nunca subir a maquina remota. Sem isto
            # TODA VM nova recomeca do fold 1, e o espelho local acaba
            # sobrescrito por uma versao mais curta. Visto em 22/08/2026.
            anterior = os.path.join(job.dir_local, "kfold_fold_results.json")
            if os.path.exists(anterior):
                try:
                    n_ant = len(json.load(open(anterior, encoding="utf-8")))
                except Exception:
                    n_ant = 0
                if n_ant:
                    destino = _dir_vm(job.grupo, job.perfil) + "/kfold_fold_results.json"
                    ok, s, _ = _cli("upload", "-s", job.sessao, anterior,
                                    destino, timeout=900)
                    self._append(job, "[painel] retomada: %d fold(s) anteriores %s"
                                 % (n_ant, "enviados" if ok else "FALHARAM ao subir"))
            # o fold EM ANDAMENTO tem seu proprio checkpoint (ckpt_atual.pt).
            # Sem ele, uma queda a meio caminho perde tudo: em 22/08/2026 o
            # PSK morreu na epoca 93 de 120 e recomecou do zero.
            ck_local = os.path.join(job.dir_local, "ckpt_atual.pt")
            if os.path.exists(ck_local) and os.path.getsize(ck_local) > 0:
                # Um checkpoint cujo fold JA concluiu e sobra: a VM o apaga ao
                # terminar, o espelho nao consegue mais baixa-lo e a copia velha
                # fica para tras. Subir 200 MB para o busca_hp descartar por
                # divergencia de identidade seria desperdicio puro.
                ident = None
                try:
                    ident = json.load(open(ck_local + ".json", encoding="utf-8")).get("id")
                except Exception:
                    pass
                feitos = set()
                try:
                    for r in json.load(open(anterior, encoding="utf-8")):
                        feitos.add((r.get("label"), r.get("fold")))
                except Exception:
                    pass
                obsoleto = ident is not None and tuple(ident) in feitos
                if obsoleto:
                    for p in (ck_local, ck_local + ".json"):
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                    self._append(job, "[painel] checkpoint de %s ja concluiu;"
                                 " descartado." % (ident,))
                else:
                    mb = os.path.getsize(ck_local) / 1e6
                    ok, s, _ = _cli("upload", "-s", job.sessao, ck_local,
                                    _dir_vm(job.grupo, job.perfil) + "/ckpt_atual.pt",
                                    timeout=1800)
                    if ok and os.path.exists(ck_local + ".json"):
                        _cli("upload", "-s", job.sessao, ck_local + ".json",
                             _dir_vm(job.grupo, job.perfil) + "/ckpt_atual.pt.json",
                             timeout=300)
                    self._append(job, "[painel] checkpoint %s de %.0f MB %s"
                                 % (ident, mb, "enviado" if ok else "FALHOU"))
            ok, s, _ = _cli("exec", "-s", job.sessao, "-f", RUNNER, timeout=600)
            self._append(job, s)
            if "RECUSADO" in s:
                self._evento(job, estado="recusado (ver log)", fim=True)
                return

            self._evento(job, estado="rodando")
            self._acompanha(job, log_vm, perfil)
        except Exception as e:
            self._append(job, "ERRO no painel: %r" % (e,))
            self._evento(job, estado="erro no painel", fim=True)
        finally:
            self._encerra(job)

    def _espera(self, job, caminho_vm, marca, tentativas):
        """Baixa um log da VM ate a marca aparecer."""
        alvo = os.path.join(job.dir_local, os.path.basename(caminho_vm))
        for _ in range(tentativas):
            if job.parar.is_set():
                return False
            veio, perdida = self._baixa(job, caminho_vm, alvo)
            if perdida:
                self._evento(job, estado="SESSAO PERDIDA")
                return False
            if veio:
                txt = open(alvo, encoding="utf-8", errors="replace").read()
                if marca in txt:
                    return True
            time.sleep(15)
        return False

    def _baixa(self, job, remoto, local):
        """Devolve (veio_arquivo, sessao_perdida).

        Arquivo ausente e situacao NORMAL: o kfold_fold_results.json so existe
        depois do primeiro fold. Quem decide se a sessao morreu e o chamador,
        contando falhas seguidas do log -- nao a ausencia de um artefato.
        """
        _, s, perdida = _cli("download", "-s", job.sessao, remoto,
                             local + ".tmp", timeout=300)
        tmp = local + ".tmp"
        # exit 0 nao significa sucesso; o que vale e o arquivo ter vindo
        if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, local)
            return True, perdida
        if os.path.exists(tmp):
            os.remove(tmp)
        return False, perdida

    def _acompanha(self, job, log_vm, perfil):
        """Espelha log e resultados enquanto o job roda."""
        log_local = os.path.join(job.dir_local, os.path.basename(log_vm))
        folds_vm = _dir_vm(job.grupo, job.perfil) + "/kfold_fold_results.json"
        folds_local = os.path.join(job.dir_local, "kfold_fold_results.json")
        def _conta(p):
            try:
                d = json.load(open(p, encoding="utf-8"))
                return len(d) if isinstance(d, list) else len(d.get("folds", []))
            except Exception:
                return 0

        def _sincroniza_folds():
            # Baixa para um temporario e so promove se NAO encurtar. Uma VM que
            # recomecou do zero devolveria 1 fold e apagaria o historico local
            # acumulado -- exatamente o que ocorreu em 22/08/2026.
            tmp = folds_local + ".vm"
            veio, _ = self._baixa(job, folds_vm, tmp)
            if not veio:
                return
            n_novo, n_velho = _conta(tmp), _conta(folds_local)
            if n_novo >= n_velho:
                os.replace(tmp, folds_local)
                self._evento(job, folds=n_novo)
            else:
                os.remove(tmp)
                self._append(job, "[painel] espelho RECUSADO: VM tem %d fold(s),"
                             " local tem %d. Historico local preservado."
                             % (n_novo, n_velho))

        ck_vm = _dir_vm(job.grupo, job.perfil) + "/ckpt_atual.pt"
        ck_local = os.path.join(job.dir_local, "ckpt_atual.pt")

        def _espelha_ckpt():
            # Cadencia propria: o checkpoint chega a centenas de MB (modelo +
            # estado do Adam), entao baixa-lo a cada ciclo como o log seria
            # desperdicio. Some da VM quando o fold conclui -- ausencia aqui e
            # normal, nao erro.
            veio, _ = self._baixa(job, ck_vm, ck_local + ".novo")
            if veio:
                os.replace(ck_local + ".novo", ck_local)
                self._baixa(job, ck_vm + ".json", ck_local + ".json")

        perdas = ciclo = 0
        while not job.parar.is_set():
            ciclo += 1
            veio, perdida = self._baixa(job, log_vm, log_local)
            if perdida:
                self._evento(job, estado="SESSAO PERDIDA", fim=True)
                return
            if veio:
                perdas = 0
                txt = open(log_local, encoding="utf-8", errors="replace").read()
                self._append(job, txt, substitui=True)
                if "RUNNER_PRONTO" in txt or "ERRO:" in txt:
                    _sincroniza_folds()      # ultima passada: pega o fold final
                    self._evento(job, estado="concluido" if "RUNNER_PRONTO" in txt
                                 else "erro no job", fim=True)
                    return
            else:
                perdas += 1
                if perdas >= 3:
                    self._evento(job, estado="SESSAO PERDIDA", fim=True)
                    return
            _sincroniza_folds()
            cada = CICLOS_CKPT.get(job.perfil or "campanha", CICLOS_CKPT_PADRAO)
            if ciclo % cada == 0:
                _espelha_ckpt()
            for _ in range(6):                      # ~60 s, mas responde ao Parar
                if job.parar.is_set():
                    return
                time.sleep(10)

    def _encerra(self, job):
        if job.estado not in ("parado",):
            _cli("stop", "-s", job.sessao, timeout=300)
        self._evento(job, estado="parado", fim=True)

    def _append(self, job, texto, substitui=False):
        self.fila.put(("log", (job.grupo, texto, substitui)))

    # ── bomba de eventos (unica thread que toca a UI) ─────────────────────────
    def _drena(self):
        try:
            while True:
                tipo, carga = self.fila.get_nowait()
                if tipo == "job":
                    grupo, kw = carga
                    job, w = self.jobs[grupo], self.w[grupo]
                    if "estado" in kw:
                        job.estado = kw["estado"]
                        w["estado"].config(text=job.estado)
                        if kw.get("fim"):
                            w["bt"].config(text="Iniciar")
                            motivo = kw["estado"]
                            job.estado = ("parado" if "erro" not in job.estado
                                          else job.estado)
                            self._talvez_religa(grupo, motivo)
                    if "sessao" in kw:
                        w["sessao"].config(text=kw["sessao"])
                    if "folds" in kw:
                        job.folds = kw["folds"]
                        w["folds"].config(text="%d/%d" % (job.folds, job.total))
                        w["pb"]["maximum"] = max(job.total, 1)
                        w["pb"]["value"] = job.folds
                elif tipo == "log":
                    grupo, texto, substitui = carga
                    job = self.jobs[grupo]
                    job.log = texto if substitui else (job.log + texto + "\n")
                    if self.grupo_log.get() == grupo:
                        self._mostra_log()
                elif tipo == "status":
                    self._status(" | ".join(carga) if carga else "")
        except queue.Empty:
            pass
        for g, _ in GRUPOS:
            job = self.jobs[g]
            if job.inicio and job.estado != "parado":
                dt = int(time.time() - job.inicio)
                self.w[g]["tempo"].config(text="%d:%02d:%02d" % (dt // 3600, dt % 3600 // 60, dt % 60))
        self.after(500, self._drena)

    def _talvez_religa(self, grupo, motivo):
        """Execucao perene: refaz o ciclo quando a VM cai por conta propria.

        Nao religa quando VOCE parou (job.parar setado) nem quando o grupo
        concluiu. Religa em queda de sessao e falha de criacao, que sao
        transitorias. Um job que viveu mais que PERENE_MIN_VIDA_S zera o
        contador: bater no teto de 63 min e o normal e nao deve contar como
        fracasso. Ja cinco fracassos RAPIDOS seguidos indicam problema real --
        sem cota, credencial vencida -- e ai desistir e o certo."""
        job = self.jobs[grupo]
        if not self.perene.get() or job.parar.is_set():
            return
        m = (motivo or '').lower()
        if not ('perdida' in m or 'falha ao criar' in m or 'sem cota' in m):
            return
        viveu = time.time() - (job.inicio or time.time())
        job.tentativas = 0 if viveu >= PERENE_MIN_VIDA_S else job.tentativas + 1
        if job.tentativas >= PERENE_MAX_SEGUIDAS:
            self._status('%s: %d falhas rapidas seguidas; a execucao perene'
                         ' desistiu deste grupo.' % (grupo, job.tentativas))
            return
        self._status('%s caiu apos %.0f min; religando em %ds (tentativa %d)'
                     % (grupo, viveu / 60.0, PERENE_ESPERA_S, job.tentativas + 1))
        self.w[grupo]['estado'].config(text='religando...')

        def _vai():
            if self.perene.get() and not self.jobs[grupo].parar.is_set():
                self.limpar_orfas()      # a sessao morta pode deixar GPU presa
                self.after(8000, lambda: self.alterna(grupo))

        self.after(PERENE_ESPERA_S * 1000, _vai)

    def _fechar(self):
        vivos = [g for g, _ in GRUPOS if self.jobs[g].estado != "parado"]
        if vivos:
            self._status("parando %s antes de sair..." % ", ".join(vivos))
            self.parar_todos()
            self.after(2500, self.destroy)
        else:
            self.destroy()


if __name__ == "__main__":
    os.makedirs(RESULTADOS, exist_ok=True)
    Painel().mainloop()
