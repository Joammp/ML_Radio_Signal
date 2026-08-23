"""Runner que roda DENTRO da VM do Colab, lancado pelo painel.

Resolve tres limitacoes do busca_hp.py que impedem orquestracao:

1. `busca_hp.py` roda SINCRONO sob `if __name__ == "__main__"`. Um
   `colab exec -f busca_hp.py` bloquearia a chamada por semanas. Aqui ele vai
   para uma thread nomeada e o exec devolve na hora.

2. `busca_hp.py` nao grava log -- so faz print. Sem arquivo, o painel nao tem
   o que acompanhar. Aqui o stdout do job e capturado para
   /content/runner_<grupo>.log.

3. Threads distintas no mesmo kernel escreviam no log errado (registrado no
   README, e visto de novo em 21/08/2026). A correcao aqui e estrutural: o
   sys.stdout do kernel vira um roteador que decide o destino pela thread
   CORRENTE. Cada job escreve no seu arquivo, sempre, sem convencao a seguir.

Parametros chegam por variavel de ambiente, porque o `colab exec` 0.6.0 nao
aceita args extras e o sys.argv de dentro do kernel e o do kernel launcher:

    RUNNER_GRUPO   grupo desta VM, ex. "QAM"       (obrigatorio)
    RUNNER_ALVO    caminho do script na VM         (default /content/busca_hp.py)
    RUNNER_ARGS    args do alvo, ex. "--grupos QAM --lr 4.5e-5"
    RUNNER_ENVVAR  nome da env que o alvo le       (default BUSCA_HP_ARGS)
"""
import os, runpy, sys, threading, time, traceback

GRUPO_RV = os.environ.get("RUNNER_GRUPO")
ALVO_RV = os.environ.get("RUNNER_ALVO", "/content/busca_hp.py")
ARGS_RV = os.environ.get("RUNNER_ARGS", "")
ENVVAR_RV = os.environ.get("RUNNER_ENVVAR", "BUSCA_HP_ARGS")

LOG_RV = "/content/runner_%s.log" % (GRUPO_RV or "sem_grupo")
FIM_RV = "RUNNER_PRONTO"
THREAD_RV = "job_%s" % (GRUPO_RV or "sem_grupo")


class _RoteadorRV(object):
    """stdout que escolhe o destino pela thread corrente.

    Instalado UMA vez no kernel. Threads nao registradas continuam indo para o
    stdout original, entao o comportamento normal do notebook fica intacto.
    """

    def __init__(self, base):
        self.base = base
        self.destinos = {}

    def registra(self, nome_thread, fh):
        self.destinos[nome_thread] = fh

    def _fh(self):
        return self.destinos.get(threading.current_thread().name, self.base)

    def write(self, s):
        fh = self._fh()
        fh.write(s)
        try:
            fh.flush()
        except Exception:
            pass
        return len(s)

    def flush(self):
        try:
            self._fh().flush()
        except Exception:
            pass

    def isatty(self):
        return False


def _instala_roteador():
    """Idempotente: reaproveita o roteador se outro job ja instalou um."""
    if isinstance(sys.stdout, _RoteadorRV):
        return sys.stdout
    r = _RoteadorRV(sys.stdout)
    sys.stdout = r
    sys.stderr = r
    return r


def _cala_tqdm():
    """Desliga as barras de progresso do alvo antes de executa-lo.

    POR QUE. O busca_hp.py usa `tqdm(tr_loader, ...)` por epoca. Fora de um
    terminal, cada atualizacao vira uma LINHA NOVA no log em vez de reescrever
    a mesma. Medido em 22/08/2026: ~12.000 reescritas de barra por MB, contra
    3 a 7 linhas uteis. Ou seja, ~99,9% do log e ruido.

    O estrago nao e so o disco. O painel baixa o log INTEIRO a cada ciclo para
    mostrar ao vivo; com o arquivo crescendo ~160 KB por epoca, o download
    passa a demorar mais que o proprio ciclo, estoura o timeout, e tres falhas
    seguidas fazem o painel declarar SESSAO PERDIDA e derrubar um job que
    estava saudavel. Foi o que encerrou a rodada de 22/08/2026.

    O que se perde: nada que importe. O busca_hp ja imprime uma linha de
    resumo por epoca (tr/vl/lr), que e o que o painel precisa mostrar.

    Feito por monkeypatch, e nao pela env TQDM_DISABLE, porque o alvo faz
    `from tqdm import tqdm` no topo: e preciso que o objeto ja esteja trocado
    quando esse import rodar.
    """
    trocados = []
    try:
        import tqdm as _tq
    except ImportError:
        return trocados

    base = _tq.tqdm

    class _Mudo(base):
        def __init__(self, *a, **k):
            k["disable"] = True
            super().__init__(*a, **k)

    for mod, nome in ((_tq, "tqdm"), (getattr(_tq, "auto", None), "auto.tqdm"),
                      (getattr(_tq, "notebook", None), "notebook.tqdm")):
        if mod is not None and hasattr(mod, "tqdm"):
            setattr(mod, "tqdm", _Mudo)
            trocados.append(nome)
    return trocados


def _trabalho(roteador):
    fh = open(LOG_RV, "a", buffering=1, encoding="utf-8", errors="replace")
    roteador.registra(THREAD_RV, fh)
    fh.write("=== runner %s | alvo=%s | args=%s | %s ===\n"
             % (GRUPO_RV, ALVO_RV, ARGS_RV, time.strftime("%H:%M:%S")))
    try:
        os.environ[ENVVAR_RV] = ARGS_RV
        os.environ["TQDM_DISABLE"] = "1"          # cinto, alem do monkeypatch
        calados = _cala_tqdm()
        fh.write('tqdm silenciado em: %s\n' % (", ".join(calados) or "nenhum"))
        runpy.run_path(ALVO_RV, run_name="__main__")
        fh.write("\n%s ok\n" % FIM_RV)
    except SystemExit as e:
        fh.write("\n%s saida=%s\n" % (FIM_RV, e.code))
    except Exception:
        fh.write("\nERRO:\n" + traceback.format_exc())
        fh.write("\n%s erro\n" % FIM_RV)
    finally:
        fh.flush()


if not GRUPO_RV:
    print("RECUSADO: RUNNER_GRUPO nao definido. O painel deve seta-la antes.")
elif not os.path.exists(ALVO_RV):
    print("RECUSADO: alvo nao existe na VM: %s (o painel precisa fazer upload)" % ALVO_RV)
elif [t for t in threading.enumerate() if t.name == THREAD_RV and t.is_alive()]:
    # mesma trava do escolhe_lr.py: duas instancias no mesmo kernel compartilham
    # o RNG global e corrompem os resultados um do outro.
    print("RECUSADO: ja existe um job %s vivo neste kernel." % GRUPO_RV)
else:
    _rot = _instala_roteador()
    open(LOG_RV, "w").close()
    threading.Thread(target=_trabalho, args=(_rot,), daemon=True,
                     name=THREAD_RV).start()
    print("runner %s iniciado | log=%s" % (GRUPO_RV, LOG_RV))
