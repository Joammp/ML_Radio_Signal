# Libera a atribuicao T4 orfa usando a propria API do CLI (client.unassign),
# que o comando `colab stop` so nao alcanca por depender do registro local.
from colab_cli.common import state
a = state.client.list_assignments()
print("atribuicoes no servidor:", len(a))
for x in a:
    print("  endpoint=%s | %s" % (getattr(x,"endpoint",None), x))
gpu = [x for x in a if "gpu" in str(getattr(x,"endpoint","")).lower()]
for x in gpu:
    ep = x.endpoint
    print("-> unassign", ep)
    try:
        state.client.unassign(ep); print("   OK, liberada")
    except Exception as e:
        print("   FALHOU:", type(e).__name__, e)
print("restantes:", [getattr(x,"endpoint",None) for x in state.client.list_assignments()])
