r"""Torna preguicoso o import de console.py no colab-cli, para funcionar no Windows.

console.py importa termios/tty (POSIX-only) no topo e execution.py o importa
incondicionalmente -> o CLI inteiro falha ao carregar no Windows. Esses modulos sao
usados so em connect_console(), para por o terminal LOCAL em modo raw, e essa funcao
e chamada apenas pelo comando `colab console`.

Rode com o python do venv do colab-cli:
    C:/Users/mclar/.colab-cli/Scripts/python.exe C:/Users/mclar/.colab-cli-windows-patch.py

Idempotente. Reaplique apos qualquer upgrade do pacote.
"""
import importlib.util, pathlib, sys

TOP  = "from colab_cli.console import connect_console\n"
LAZY = "    from colab_cli.console import connect_console  # lazy: POSIX-only (termios/tty)\n"
CALL = "    try:\n        connect_console(s)\n"

spec = importlib.util.find_spec("colab_cli")
if spec is None or not spec.submodule_search_locations:
    sys.exit("[patch] ERRO: colab_cli nao encontrado neste interpretador: " + sys.executable)
base = pathlib.Path(list(spec.submodule_search_locations)[0])

f = base / "commands" / "execution.py"
src = f.read_text(encoding="utf-8")

if TOP not in src:
    print("[patch] import de topo ja ausente - nada a fazer (ja aplicado)")
    sys.exit(0)
if CALL not in src:
    sys.exit("[patch] ERRO: call site esperado nao encontrado; o arquivo mudou. Inspecione " + str(f))

f.with_suffix(".py.bak").write_text(src, encoding="utf-8")
f.write_text(src.replace(TOP, "", 1).replace(CALL, LAZY + CALL, 1), encoding="utf-8")
print("[patch] aplicado em", f)
