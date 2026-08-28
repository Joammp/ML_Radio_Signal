"""Gera busca_hp.py a partir do notebook, mantendo o notebook como fonte da verdade.

Fonte: branch `atualiza-busca-hp-estratificacao` do repo ML_Radio_Signal, que ja
tem STRAT_BY_SNR=True, MIN_DELTA e a reducao escalonada do LR. NAO usar a versao
de Downloads, que e anterior a tudo isso.

A celula de busca (18) e quase autocontida: depende de apenas 5 nomes externos
(H5PyDataset, drive_push, drive_pull, path, modulation_classes_path). Este script
fornece esses 5, aplica as substituicoes de hiperparametro e troca o `run_all()`
final por um guard de __main__.

Cada substituicao e VERIFICADA: se o notebook mudar e um padrao nao casar, o
script falha em vez de gerar codigo silenciosamente errado.

Uso:
    python gerar_busca_hp.py [--notebook CAMINHO] [--saida busca_hp.py]
"""
import argparse
import json
import os
import sys
import urllib.request

RAW = ("https://raw.githubusercontent.com/Joammp/ML_Radio_Signal/"
       "atualiza-busca-hp-estratificacao/BUSCA_HP_Classes_1cap(3).ipynb")

CEL_DATASET = 13   # H5PyDataset
CEL_BUSCA = 18     # a celula de busca inteira

# (descricao, de, para) — todas obrigatorias
SUBS = [
    ('learning rate inicial',
     'LEARNING_RATE    = 1e-3',
     'LEARNING_RATE    = _ARGS.lr'),
    ('paciencia do LR',
     'LR_PATIENCE      = 5',
     'LR_PATIENCE      = _ARGS.lr_patience'),
    ('early stop',
     'EARLY_STOP       = 15',
     'EARLY_STOP       = _ARGS.early_stop'),
    ('piso do LR',
     'LR_MIN           = 1e-9',
     'LR_MIN           = _ARGS.lr_min'),
    ('min delta, estagnacao e checkpoint',
     'MIN_DELTA        = 0.05',
     'CKPT_EVERY          = _ARGS.ckpt_every           # epocas entre checkpoints\nSTAGNATION_PATIENCE = _ARGS.stagnation_patience  # epocas sem avanco real -> encerra\nSTAGNATION_MARGIN   = _ARGS.stagnation_margin    # p.p. que definem "avanco real"\nMIN_DELTA        = _ARGS.min_delta'),
    ('comentario da reducao escalonada (o exemplo ficou obsoleto)',
     '#   n=1 → 5.0e-04   n=3 → 1.6e-05   n=5 → 3.1e-08   n=6 → 4.8e-10\n# ⇒ 6 reduções consecutivas levam 1e-3 abaixo de LR_MIN=1e-9,\n#   ou seja 6*LR_PATIENCE = 30 épocas de estagnação (antes: inalcançável).',
     '# Caminho ate o piso, partindo de lr0: sao n reducoes tais que\n#   lr0 * BASE**(n(n+1)/2) <= LR_MIN.  Para lr0=4.5e-5 e LR_MIN=1e-7 -> n=4\n#   (2.3e-5, 5.6e-6, 7.0e-7, piso).\n# ATENCAO: epochs_no_improve zera A CADA REDUCAO, entao o caminho minimo ate o\n#   early stop e n*LR_PATIENCE + EARLY_STOP epocas de estagnacao ININTERRUPTA.\n#   Um unico falso recorde acima de MIN_DELTA reinicia a contagem inteira --\n#   foi o que fez um fold do ASK rodar as 120 epocas em 21/08/2026.'),
    ('assinatura do train_one_fold',
     '                   lr_factor_base=None, lr_factor_floor=None, min_delta=None):',
     '                   lr_factor_base=None, lr_factor_floor=None, min_delta=None,\n                   stag_patience=None, stag_margin=None,\n                   ckpt_path=None, ckpt_every=None, ckpt_id=None):'),
    ('defaults dos parametros de parada',
     '    if min_delta       is None: min_delta       = MIN_DELTA\n',
     '    if min_delta       is None: min_delta       = MIN_DELTA\n    if stag_patience   is None: stag_patience   = STAGNATION_PATIENCE\n    if stag_margin     is None: stag_margin     = STAGNATION_MARGIN\n    if ckpt_every      is None: ckpt_every      = CKPT_EVERY\n'),
    ('estado do marco absoluto',
     '    consec_drops      = 0      # reduções consecutivas sem melhora → escalonamento\n',
     '    consec_drops      = 0      # reduções consecutivas sem melhora → escalonamento\n    marco_acc         = -1e9   # último avanço REAL (supera stag_margin)\n    marco_ep          = 0      # época desse avanço; base da parada absoluta\n'),
    ('atualizacao do marco',
     '        # ── Paciência: só é melhora se superar min_delta ──────────────────────\n        if vl_acc > ref_acc + min_delta:\n',
     '        # ── Marco absoluto: base da parada por estagnação ─────────────────────\n        if vl_acc > marco_acc + stag_margin:\n            marco_acc, marco_ep = vl_acc, epoch + 1\n\n        # ── Paciência: só é melhora se superar min_delta ──────────────────────\n        if vl_acc > ref_acc + min_delta:\n'),
    ('chamada de train_one_fold',
     '                lr_patience= LR_PATIENCE,\n                early_stop = EARLY_STOP,\n',
     '                lr_patience= LR_PATIENCE,\n                early_stop = EARLY_STOP,\n                stag_patience = STAGNATION_PATIENCE,\n                stag_margin   = STAGNATION_MARGIN,\n                ckpt_path     = _ckpt_path(modelo),\n                ckpt_every    = CKPT_EVERY,\n                ckpt_id       = [label, fold_i, fold_seed],\n'),
    ('parada por estagnacao + checkpoint',
     '        # ── Early stop: só depois de o LR ter chegado ao piso ─────────────────\n        elif epochs_no_improve >= early_stop and cur_lr <= lr_min * 1.001:\n            print(f"    🛑 Early stopping na época {epoch+1} "\n                  f"(LR no piso {lr_min:.1e} + {early_stop} épocas sem melhora)")\n            break\n\n    if best_state is not None:',
     '        # ── Early stop: só depois de o LR ter chegado ao piso ─────────────────\n        elif epochs_no_improve >= early_stop and cur_lr <= lr_min * 1.001:\n            print(f"    🛑 Early stopping na época {epoch+1} "\n                  f"(LR no piso {lr_min:.1e} + {early_stop} épocas sem melhora)")\n            break\n\n        # ── Parada por estagnação ABSOLUTA ────────────────────────────────────\n        # Independente do LR de propósito. O critério acima só dispara com o LR\n        # no piso, e chegar lá depende de `epochs_no_improve`/`consec_drops`, que\n        # zeram a cada oscilação para cima da val_acc. Medido em 22/08/2026: o\n        # ruído p90 entre épocas é 0,33 p.p. no ASK, 0,28 no QAM e 1,33 no APSK\n        # — sempre acima de MIN_DELTA. No APSK isso travou o escalonamento em\n        # ×0.5, o LR parou em 3,5e-7 sem alcançar o piso de 1e-7, e o fold rodou\n        # as 120 épocas inteiras. Não há valor de MIN_DELTA que resolva: a\n        # varredura deu resultado não-monotônico, porque mexer no limiar muda o\n        # caminho do LR. Este contador não é zerado por redução de LR nem por\n        # ruído abaixo de `stag_margin`; conta desde o último avanço REAL.\n        if (epoch + 1) - marco_ep >= stag_patience:\n            print(f"    🛑 Parada por estagnação na época {epoch+1} "\n                  f"({stag_patience} épocas sem avanço de {stag_margin} p.p.; "\n                  f"último marco: {marco_acc:.2f}% na época {marco_ep})")\n            break\n\n        # ── Checkpoint periodico ──────────────────────────────────────────────\n        # Custa ~1 s e salva ate uma hora de GPU quando a VM cai. Fica so o\n        # arquivo mais recente (sobrescreve), gravado via .tmp + os.replace\n        # para uma queda no meio da escrita nao deixar um checkpoint corrompido.\n        if ckpt_path and ckpt_every and (epoch + 1) % ckpt_every == 0:\n            try:\n                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)\n                tmp = ckpt_path + ".tmp"\n                torch.save({\n                    "id"               : list(ckpt_id) if ckpt_id else None,\n                    "epoch"            : epoch + 1,\n                    "model"            : model.state_dict(),\n                    "optim"            : optimizer.state_dict(),\n                    "lr"               : optimizer.param_groups[0]["lr"],\n                    "best_val_acc"     : best_val_acc,\n                    "ref_acc"          : ref_acc,\n                    "epochs_no_improve": epochs_no_improve,\n                    "consec_drops"     : consec_drops,\n                    "marco_acc"        : marco_acc,\n                    "marco_ep"         : marco_ep,\n                    "history"          : history,\n                    # Sem estes, a retomada NAO e identica: o dropout e a ordem\n                    # de embaralhamento do DataLoader continuariam de outro\n                    # ponto da sequencia. O gerador do loader e o mesmo objeto\n                    # que o chamador semeou com fold_seed.\n                    "rng_torch"        : torch.get_rng_state(),\n                    "rng_cuda"         : (torch.cuda.get_rng_state_all()\n                                          if torch.cuda.is_available() else None),\n                    "rng_loader"       : (tr_loader.generator.get_state()\n                                          if getattr(tr_loader, "generator", None)\n                                          is not None else None),\n                    "rng_np"           : np.random.get_state(),\n                    "rng_py"           : random.getstate(),\n                }, tmp)\n                os.replace(tmp, ckpt_path)\n                # indice leve: o painel precisa saber a QUE fold este\n                # checkpoint pertence antes de decidir se vale subir centenas\n                # de MB para a VM nova. Ler o .pt so para isso seria absurdo.\n                with open(ckpt_path + \'.json\', \'w\') as _j:\n                    json.dump({\'id\': list(ckpt_id) if ckpt_id else None,\n                               \'epoch\': epoch + 1,\n                               \'best_val_acc\': best_val_acc}, _j)\n                print(f"    💾 checkpoint ep {epoch+1} "\n                      f"({os.path.getsize(ckpt_path)/1e6:.0f} MB)")\n            except Exception as e:\n                print(f"    (falha ao gravar checkpoint: {e!r})")\n\n    # Fold concluido: o checkpoint perdeu a razao de existir.\n    if ckpt_path:\n        for _p in (ckpt_path, ckpt_path + \'.json\'):\n            try:\n                os.remove(_p)\n            except OSError:\n                pass\n\n    if best_state is not None:'),
    ('banner com a parada por estagnacao',
     '    print(f"  LR escalonado: fator {LR_FACTOR_BASE}^n  │  "\n          f"piso {LR_MIN:.0e}  │  min_delta {MIN_DELTA} p.p.")\n',
     '    print(f"  LR escalonado: fator {LR_FACTOR_BASE}^n  │  "\n          f"piso {LR_MIN:.0e}  │  min_delta {MIN_DELTA} p.p.")\n    print(f"  Parada por estagnação: {STAGNATION_PATIENCE} épocas sem avanço "\n          f"de {STAGNATION_MARGIN} p.p. (independente do LR)")\n'),
    ('tf32 desligado (L4 tem, T4 nao)',
     'torch.backends.cudnn.deterministic = True\ntorch.backends.cudnn.benchmark     = False',
     'torch.backends.cudnn.deterministic = True\ntorch.backends.cudnn.benchmark     = False\n# TF32 desligado de proposito: a T4 (Turing) nao tem TF32, entao toda a\n# referencia do projeto foi medida em fp32 real. L4 (Ada) tem, e o PyTorch\n# liga cudnn.allow_tf32 por padrao -> as convolucoes cairiam para 10 bits de\n# mantissa (precisao de fp16) sem aviso, quebrando a comparabilidade.\ntorch.backends.cudnn.allow_tf32       = False\ntorch.backends.cuda.matmul.allow_tf32 = False\n\n_GPU_NOME = (torch.cuda.get_device_name(0) if torch.cuda.is_available()\n             else "cpu")'),
    ('retira AM e FM do GROUP_MAP',
     '    "AM-SSB-WC": 4, "AM-SSB-SC": 4, "AM-DSB-WC": 4, "AM-DSB-SC": 4,\n    "FM":        5,\n}\nGROUP_NAMES = {0: "ASK", 1: "PSK", 2: "APSK", 3: "QAM", 4: "AM", 5: "FM"}',
     '    # AM (AM-SSB-WC/SC, AM-DSB-WC/SC) e FM foram RETIRADOS do estudo.\n    # Sao modulacoes analogicas; alem disso o FM tem uma unica classe, o que\n    # torna a classificacao dentro do grupo degenerada. Restam 19 modulacoes\n    # digitais em 4 grupos.\n}\nGROUP_NAMES = {0: "ASK", 1: "PSK", 2: "APSK", 3: "QAM"}\nGRUPOS_DIGITAIS = ["ASK", "PSK", "APSK", "QAM"]\n# alvos que nao sao um grupo isolado:\n#   GROUP  -> todas as 19 digitais, rotuladas pelo GRUPO (4 classes)\n#   TODAS  -> todas as 19 digitais, rotuladas pela MODULACAO (19 classes)\nALVOS_ESPECIAIS = ["GROUP", "TODAS"]'),
    ('selecao de alvo (GROUP e TODAS)',
     '    target_id       = {v: k for k, v in GROUP_NAMES.items()}[modelo]\n    group_indices   = np.array(\n        sorted([i for i, m in enumerate(mod_classes)\n                if GROUP_MAP[m] == target_id]),\n        dtype=np.int64\n    )\n    num_classes     = len(group_indices)\n    global_to_local = {int(g): l for l, g in enumerate(group_indices)}\n',
     '    group_indices, global_to_local, num_classes = _alvo_info(modelo, mod_classes)\n'),
    ('_alvo_info + import da ResNet',
     'def ensure_hdf5(modelo, mod_classes):',
     'def _alvo_info(modelo, mod_classes):\n    """Indices globais, mapa global->rotulo e numero de classes de um alvo.\n\n    Tres formas de alvo, todas alimentando o mesmo _build_hdf5:\n\n      <grupo>  ASK/PSK/APSK/QAM -- so as modulacoes daquele grupo, rotuladas\n               pela modulacao (3, 7, 4 e 5 classes respectivamente).\n      GROUP    todas as 19 digitais, rotuladas pelo GRUPO -> 4 classes. E o\n               classificador de primeiro estagio.\n      TODAS    todas as 19 digitais, rotuladas pela MODULACAO -> 19 classes.\n               E a tarefa que o artigo faz de uma vez so (la com 24, porque\n               inclui as analogicas que retiramos).\n    """\n    digitais = sorted(i for i, m in enumerate(mod_classes) if m in GROUP_MAP)\n    if modelo == "GROUP":\n        idx = np.array(digitais, dtype=np.int64)\n        mapa = {int(g): GROUP_MAP[mod_classes[g]] for g in digitais}\n        return idx, mapa, len(GROUP_NAMES)\n    if modelo == "TODAS":\n        idx = np.array(digitais, dtype=np.int64)\n        return idx, {int(g): l for l, g in enumerate(digitais)}, len(digitais)\n    alvo = {v: k for k, v in GROUP_NAMES.items()}[modelo]\n    idx = np.array(sorted(i for i, m in enumerate(mod_classes)\n                          if GROUP_MAP.get(m) == alvo), dtype=np.int64)\n    return idx, {int(g): l for l, g in enumerate(idx)}, len(idx)\n\n\ndef ensure_hdf5(modelo, mod_classes):'),
    ('carregador do modulo resnet',
     'import torch.optim as optim\n',
     'import torch.optim as optim\n\n# ResNet fiel a Tabela IV / Figura 5 de arXiv:1712.04578; ver campanha/resnet.py\nimport importlib.util as _ilu, os as _os\n\n\ndef _carrega_resnet():\n    """resnet.py fica ao lado deste arquivo; na VM, em /content (o painel envia)."""\n    tentativas = []\n    try:\n        tentativas.append(_os.path.join(\n            _os.path.dirname(_os.path.abspath(__file__)), "resnet.py"))\n    except NameError:\n        pass\n    tentativas.append("/content/resnet.py")\n    for cam in tentativas:\n        if _os.path.exists(cam):\n            sp = _ilu.spec_from_file_location("resnet", cam)\n            mod = _ilu.module_from_spec(sp)\n            sp.loader.exec_module(mod)\n            return mod\n    raise FileNotFoundError("resnet.py nao encontrado em %s" % tentativas)\n\n\nresnet = _carrega_resnet()\n'),
    ('artefatos separados por tipo de modelo',
     'def _drive_dir(modelo):\n    """Retorna e cria o diretório do modelo no Drive."""\n    d = os.path.join(DRIVE_BASE, modelo)',
     'def _drive_dir(modelo):\n    """Diretorio dos artefatos do alvo, SEPARADO por tipo de modelo.\n\n    A CNN mantem o caminho historico <base>/<alvo>, para os folds ja\n    concluidos continuarem validos. A ResNet vai para <base>/<alvo>_resnet.\n    Sem isso as duas gravariam no mesmo kfold_fold_results.json e a retomada\n    trataria uma busca de 12 arquiteturas e uma rede fixa como a mesma coisa.\n    """\n    nome = modelo if MODELO_TIPO == "cnn" else "%s_%s" % (modelo, MODELO_TIPO)\n    d = os.path.join(DRIVE_BASE, nome)'),
    ('ResNet como arquitetura unica + seletor',
     'ARCHITECTURES = [\n    {\n        "label": "2L_32-64",\n        "arch": [\n            {"out_channels": 32,  "kernel_size": 7, "pool": True},\n            {"out_channels": 64,  "kernel_size": 5, "pool": True},\n        ],\n    },\n    {\n        "label": "2L_64-128",\n        "arch": [\n            {"out_channels": 64,  "kernel_size": 7, "pool": True},\n            {"out_channels": 128, "kernel_size": 5, "pool": True},\n        ],\n    },\n    {\n        "label": "3L_32-64-128",\n        "arch": [\n            {"out_channels": 32,  "kernel_size": 7, "pool": True},\n            {"out_channels": 64,  "kernel_size": 5, "pool": True},\n            {"out_channels": 128, "kernel_size": 3, "pool": True},\n        ],\n    },\n    {\n        "label": "3L_64-128-256",\n        "arch": [\n            {"out_channels": 64,  "kernel_size": 7, "pool": True},\n            {"out_channels": 128, "kernel_size": 5, "pool": True},\n            {"out_channels": 256, "kernel_size": 3, "pool": True},\n        ],\n    },\n    {\n        "label": "3L_128-256-512",\n        "arch": [\n            {"out_channels": 128, "kernel_size": 7, "pool": True},\n            {"out_channels": 256, "kernel_size": 5, "pool": True},\n            {"out_channels": 512, "kernel_size": 3, "pool": True},\n        ],\n    },\n    {\n        "label": "4L_32-64-128-256",\n        "arch": [\n            {"out_channels": 32,  "kernel_size": 11, "pool": True},\n            {"out_channels": 64,  "kernel_size": 7,  "pool": True},\n            {"out_channels": 128, "kernel_size": 5,  "pool": True},\n            {"out_channels": 256, "kernel_size": 3,  "pool": True},\n        ],\n    },\n    {\n        "label": "4L_64-128-256-512",\n        "arch": [\n            {"out_channels": 64,  "kernel_size": 11, "pool": True},\n            {"out_channels": 128, "kernel_size": 7,  "pool": True},\n            {"out_channels": 256, "kernel_size": 5,  "pool": True},\n            {"out_channels": 512, "kernel_size": 3,  "pool": True},\n        ],\n    },\n    {\n        "label": "4L_128-256-512-512",\n        "arch": [\n            {"out_channels": 128, "kernel_size": 11, "pool": True},\n            {"out_channels": 256, "kernel_size": 7,  "pool": True},\n            {"out_channels": 512, "kernel_size": 5,  "pool": True},\n            {"out_channels": 512, "kernel_size": 3,  "pool": True},\n        ],\n    },\n    {\n        "label": "5L_32-64-128-256-512",\n        "arch": [\n            {"out_channels": 32,   "kernel_size": 11, "pool": True},\n            {"out_channels": 64,   "kernel_size": 7,  "pool": True},\n            {"out_channels": 128,  "kernel_size": 5,  "pool": True},\n            {"out_channels": 256,  "kernel_size": 3,  "pool": True},\n            {"out_channels": 512,  "kernel_size": 3,  "pool": True},\n        ],\n    },\n    {\n        "label": "5L_64-128-256-512-1024",\n        "arch": [\n            {"out_channels": 64,   "kernel_size": 11, "pool": True},\n            {"out_channels": 128,  "kernel_size": 7,  "pool": True},\n            {"out_channels": 256,  "kernel_size": 5,  "pool": True},\n            {"out_channels": 512,  "kernel_size": 3,  "pool": True},\n            {"out_channels": 1024, "kernel_size": 3,  "pool": True},\n        ],\n    },\n    {\n        "label": "6L_64-64-128-128-256-512_mixpool",\n        "arch": [\n            {"out_channels": 64,  "kernel_size": 11, "pool": True},\n            {"out_channels": 64,  "kernel_size": 7,  "pool": False},\n            {"out_channels": 128, "kernel_size": 5,  "pool": True},\n            {"out_channels": 128, "kernel_size": 3,  "pool": False},\n            {"out_channels": 256, "kernel_size": 3,  "pool": True},\n            {"out_channels": 512, "kernel_size": 3,  "pool": True},\n        ],\n    },\n    {\n        "label": "6L_32-64-64-128-256-512_mixpool",\n        "arch": [\n            {"out_channels": 32,  "kernel_size": 11, "pool": True},\n            {"out_channels": 64,  "kernel_size": 7,  "pool": False},\n            {"out_channels": 64,  "kernel_size": 5,  "pool": True},\n            {"out_channels": 128, "kernel_size": 3,  "pool": False},\n            {"out_channels": 256, "kernel_size": 3,  "pool": True},\n            {"out_channels": 512, "kernel_size": 3,  "pool": True},\n        ],\n    },\n]\n',
     '# A ResNet do artigo e uma arquitetura FIXA: nao ha o que buscar. Ela entra\n# como candidata unica, e o restante do pipeline -- k-fold, estratificacao\n# conjunta (classe, SNR), train_one_fold, checkpoint, parada por estagnacao --\n# fica exatamente igual ao da CNN. E assim que o artigo compara VGG e ResNet.\nARQ_RESNET = [{"label": "ResNet_L%d_k%d" % (resnet.N_STACKS, resnet.KERNEL),\n               "arch": None}]\n\nARCHITECTURES_CNN = [\n    {\n        "label": "2L_32-64",\n        "arch": [\n            {"out_channels": 32,  "kernel_size": 7, "pool": True},\n            {"out_channels": 64,  "kernel_size": 5, "pool": True},\n        ],\n    },\n    {\n        "label": "2L_64-128",\n        "arch": [\n            {"out_channels": 64,  "kernel_size": 7, "pool": True},\n            {"out_channels": 128, "kernel_size": 5, "pool": True},\n        ],\n    },\n    {\n        "label": "3L_32-64-128",\n        "arch": [\n            {"out_channels": 32,  "kernel_size": 7, "pool": True},\n            {"out_channels": 64,  "kernel_size": 5, "pool": True},\n            {"out_channels": 128, "kernel_size": 3, "pool": True},\n        ],\n    },\n    {\n        "label": "3L_64-128-256",\n        "arch": [\n            {"out_channels": 64,  "kernel_size": 7, "pool": True},\n            {"out_channels": 128, "kernel_size": 5, "pool": True},\n            {"out_channels": 256, "kernel_size": 3, "pool": True},\n        ],\n    },\n    {\n        "label": "3L_128-256-512",\n        "arch": [\n            {"out_channels": 128, "kernel_size": 7, "pool": True},\n            {"out_channels": 256, "kernel_size": 5, "pool": True},\n            {"out_channels": 512, "kernel_size": 3, "pool": True},\n        ],\n    },\n    {\n        "label": "4L_32-64-128-256",\n        "arch": [\n            {"out_channels": 32,  "kernel_size": 11, "pool": True},\n            {"out_channels": 64,  "kernel_size": 7,  "pool": True},\n            {"out_channels": 128, "kernel_size": 5,  "pool": True},\n            {"out_channels": 256, "kernel_size": 3,  "pool": True},\n        ],\n    },\n    {\n        "label": "4L_64-128-256-512",\n        "arch": [\n            {"out_channels": 64,  "kernel_size": 11, "pool": True},\n            {"out_channels": 128, "kernel_size": 7,  "pool": True},\n            {"out_channels": 256, "kernel_size": 5,  "pool": True},\n            {"out_channels": 512, "kernel_size": 3,  "pool": True},\n        ],\n    },\n    {\n        "label": "4L_128-256-512-512",\n        "arch": [\n            {"out_channels": 128, "kernel_size": 11, "pool": True},\n            {"out_channels": 256, "kernel_size": 7,  "pool": True},\n            {"out_channels": 512, "kernel_size": 5,  "pool": True},\n            {"out_channels": 512, "kernel_size": 3,  "pool": True},\n        ],\n    },\n    {\n        "label": "5L_32-64-128-256-512",\n        "arch": [\n            {"out_channels": 32,   "kernel_size": 11, "pool": True},\n            {"out_channels": 64,   "kernel_size": 7,  "pool": True},\n            {"out_channels": 128,  "kernel_size": 5,  "pool": True},\n            {"out_channels": 256,  "kernel_size": 3,  "pool": True},\n            {"out_channels": 512,  "kernel_size": 3,  "pool": True},\n        ],\n    },\n    {\n        "label": "5L_64-128-256-512-1024",\n        "arch": [\n            {"out_channels": 64,   "kernel_size": 11, "pool": True},\n            {"out_channels": 128,  "kernel_size": 7,  "pool": True},\n            {"out_channels": 256,  "kernel_size": 5,  "pool": True},\n            {"out_channels": 512,  "kernel_size": 3,  "pool": True},\n            {"out_channels": 1024, "kernel_size": 3,  "pool": True},\n        ],\n    },\n    {\n        "label": "6L_64-64-128-128-256-512_mixpool",\n        "arch": [\n            {"out_channels": 64,  "kernel_size": 11, "pool": True},\n            {"out_channels": 64,  "kernel_size": 7,  "pool": False},\n            {"out_channels": 128, "kernel_size": 5,  "pool": True},\n            {"out_channels": 128, "kernel_size": 3,  "pool": False},\n            {"out_channels": 256, "kernel_size": 3,  "pool": True},\n            {"out_channels": 512, "kernel_size": 3,  "pool": True},\n        ],\n    },\n    {\n        "label": "6L_32-64-64-128-256-512_mixpool",\n        "arch": [\n            {"out_channels": 32,  "kernel_size": 11, "pool": True},\n            {"out_channels": 64,  "kernel_size": 7,  "pool": False},\n            {"out_channels": 64,  "kernel_size": 5,  "pool": True},\n            {"out_channels": 128, "kernel_size": 3,  "pool": False},\n            {"out_channels": 256, "kernel_size": 3,  "pool": True},\n            {"out_channels": 512, "kernel_size": 3,  "pool": True},\n        ],\n    },\n]\n\nMODELO_TIPO   = _ARGS.modelo\nARCHITECTURES = ARQ_RESNET if MODELO_TIPO == \'resnet\' else ARCHITECTURES_CNN\n'),
    ('instanciacao do modelo',
     '            model    = FlexCNN(num_classes=num_classes, arch=arch,\n                               classifier=CLASSIFIER_HEAD, dropout=DROPOUT)',
     '            if MODELO_TIPO == "resnet":\n                model = resnet.ResNet(num_classes=num_classes)\n            else:\n                model = FlexCNN(num_classes=num_classes, arch=arch,\n                                classifier=CLASSIFIER_HEAD, dropout=DROPOUT)'),
    ('print da arquitetura (ResNet nao tem filtros)',
     '        print(f"  Filtros : " +\n              " → ".join(str(b["out_channels"]) for b in arch))',
     '        if arch:\n            print(f"  Filtros : " +\n                  " → ".join(str(b["out_channels"]) for b in arch))\n        else:\n            print(f"  ResNet  : {resnet.N_STACKS} stacks × {resnet.CHANNELS} canais"\n                  f"  │  kernel {resnet.KERNEL}  │  arXiv:1712.04578")'),
    ('ambiente e tipo no fold_result',
     '                "arch"        : arch,\n',
     '                "modelo_tipo" : MODELO_TIPO,\n                # Ambiente. Sem isto nao da para auditar depois: medimos em\n                # 22/08/2026 que L4 e T4 NAO reproduzem o mesmo resultado com a\n                # mesma semente, e os folds rodam em qualquer GPU que esteja\n                # livre. Registrar nao controla a variavel, mas a torna visivel.\n                "gpu"         : _GPU_NOME,\n                "torch"       : torch.__version__,\n                "cuda"        : torch.version.cuda,\n                "tf32"        : bool(torch.backends.cudnn.allow_tf32),\n                "arch"        : arch,\n                "n_layers"    : len(arch) if arch else resnet.N_STACKS,\n                "filters"     : ([b["out_channels"] for b in arch] if arch\n                                 else [resnet.CHANNELS] * resnet.N_STACKS),\n                "classifier"  : CLASSIFIER_HEAD,'),
    ('semente independente da candidata',
     '            # Seed determinístico e único por (arch, fold)\n            fold_seed = SEED + arch_idx * 100 + fold_i',
     '            # Seed determinístico, funcao APENAS do fold.\n            #\n            # Era `SEED + arch_idx*100 + fold_i`, o que dava a cada candidata um\n            # conjunto proprio de sementes -- gap1 usava 142-146, gap8_head128\n            # 242-246, e assim por diante. Num projeto cuja tese central e que o\n            # colapso depende da INICIALIZACAO, isso confundia arquitetura com\n            # semente: no ASK, quatro candidatas ficaram separadas por 0,10 a\n            # 0,27 p.p., abaixo da propria dispersao entre folds (0,16 a 0,41).\n            # E o colapso da gap8_head128 no QAM (4 de 5 folds) nao podia ser\n            # atribuido a arquitetura, porque so ela viu aquelas sementes.\n            #\n            # Agora todas as candidatas partem da MESMA inicializacao em cada\n            # fold, e a diferenca entre elas e atribuivel ao que se quis variar.\n            fold_seed = SEED + fold_i'),
    ('aviso de troca de GPU',
     '    all_results, done_set = load_results(modelo)\n',
     '    all_results, done_set = load_results(modelo)\n\n    # A GPU e a unica condicao que nao conseguimos fixar: os folds rodam em\n    # qualquer aceleradora que esteja livre. Medimos que L4 e T4 nao reproduzem\n    # o mesmo resultado com a mesma semente, entao misturar as duas dentro de um\n    # mesmo conjunto compromete a comparacao. Nao da para impedir daqui, mas da\n    # para avisar alto -- e o registro no fold_result permite auditar depois.\n    _gpus_antes = {r.get(\'gpu\') for r in all_results if r.get(\'gpu\')}\n    if _gpus_antes and _GPU_NOME not in _gpus_antes:\n        print(f"  ATENCAO: folds anteriores deste alvo rodaram em "\n              f"{sorted(_gpus_antes)}, e esta sessao esta em {_GPU_NOME}.")\n        print(f"           Misturar GPUs dentro do mesmo conjunto compromete a "\n              f"comparacao entre candidatas.")\n    elif _gpus_antes:\n        print(f"  GPU consistente com os folds anteriores: {_GPU_NOME}")\n\n    # Trava de semeadura. Ate 27/08/2026 a semente era SEED + arch_idx*100 +\n    # fold_i, dando a cada candidata um conjunto proprio. Retomar um diretorio\n    # daquela geracao acrescentaria folds da semeadura NOVA ao mesmo\n    # kfold_fold_results.json, e nada no arquivo denunciaria a mistura depois:\n    # o checkpoint confere (rotulo, fold), nunca a formula da semente.\n    _semeadura_ruim = [(r.get("label"), r.get("fold"), r.get("fold_seed"))\n                       for r in all_results\n                       if r.get("fold_seed") is not None\n                       and r.get("fold_seed") != SEED + r.get("fold", 0)]\n    if _semeadura_ruim:\n        raise SystemExit(\n            "\\n  RECUSADO: %d fold(s) neste diretorio foram gravados com outra\\n"\n            "  semeadura. Esperado fold_seed = SEED + fold (%d + fold);\\n"\n            "  encontrado, por exemplo: %s\\n"\n            "  Diretorio: %s\\n"\n            "  Misturar as duas geracoes no mesmo conjunto e irreversivel.\\n"\n            "  Aponte --drive-base para uma raiz nova, ou arquive o diretorio.\\n"\n            % (len(_semeadura_ruim), SEED, _semeadura_ruim[:3],\n               _drive_dir(modelo)))\n'),
    ('grupos alvo',
     'GRUPOS_ALVO = ["APSK","QAM"]',
     'GRUPOS_ALVO = _ARGS.grupos'),
    ('selecao de device',
     'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
     'device = torch.device(_ARGS.device)'),
    ('execucao automatica no import',
     '\nrun_all()',
     '\nif __name__ == "__main__":\n    run_all()'),
    ('retomada de fold interrompido',
     '    history           = []\n    model.to(device)\n\n    for epoch in range(epochs):',
     '    history           = []\n    model.to(device)\n\n    # ── RETOMADA DE FOLD INTERROMPIDO ─────────────────────────────────────────\n    # A VM do Colab cai a cada ~1 h. Sem isto, um fold interrompido perde tudo:\n    # em 22/08/2026 o PSK morreu na epoca 93 de 120 e recomecou do zero. O\n    # kfold_fold_results.json so grava folds CONCLUIDOS, entao nao ajuda aqui.\n    # O `best_state` de proposito NAO entra no checkpoint: ele so alimenta um\n    # load_state_dict no fim que ninguem consome (o chamador usa apenas\n    # best_val_acc e history), e guarda-lo dobraria o tamanho do arquivo.\n    ep_inicial = 0\n    if ckpt_path and os.path.exists(ckpt_path):\n        try:\n            ck = torch.load(ckpt_path, map_location=device, weights_only=False)\n            if ckpt_id is not None and ck.get(\'id\') != list(ckpt_id):\n                # sobra de outro (arquitetura, fold): ignorar e comecar limpo\n                raise ValueError(\'checkpoint de %r, esperado %r\'\n                                 % (ck.get(\'id\'), list(ckpt_id)))\n\n            # ATOMICIDADE. Em 22/08/2026 um fold do QAM voltou com history de\n            # 185 entradas para um teto de 120: as epocas 1-65 do checkpoint\n            # MAIS um treino completo 1-120. A causa era este bloco atribuir\n            # direto nas variaveis e o `except` so zerar ep_inicial -- o\n            # history restaurado sobrevivia e o treino recomecava por cima.\n            # Agora nada e comprometido antes de TUDO dar certo.\n            _model_sd = ck[\'model\']\n            _optim_sd = ck[\'optim\']\n            _novo = dict(ep_inicial=ck[\'epoch\'], best=ck[\'best_val_acc\'],\n                         ref=ck[\'ref_acc\'], sem=ck[\'epochs_no_improve\'],\n                         drops=ck[\'consec_drops\'], m_acc=ck[\'marco_acc\'],\n                         m_ep=ck[\'marco_ep\'], hist=list(ck[\'history\']),\n                         lr=ck[\'lr\'])\n            # torch.load(..., map_location=device) move TODOS os tensores do\n            # checkpoint para a GPU -- inclusive os estados de RNG. Mas\n            # set_rng_state e Generator.set_state exigem ByteTensor de CPU, e\n            # falham com TypeError(\'RNG state must be a torch.ByteTensor\').\n            # Visto em producao em 22/08/2026 no fold 1 do TODAS/ResNet.\n            def _byte_cpu(t):\n                return t.detach().to(\'cpu\', torch.uint8) if torch.is_tensor(t) else t\n\n            if ck.get(\'rng_torch\') is not None:\n                torch.set_rng_state(_byte_cpu(ck[\'rng_torch\']))\n            if ck.get(\'rng_cuda\') and torch.cuda.is_available():\n                try:\n                    torch.cuda.set_rng_state_all([_byte_cpu(t) for t in ck[\'rng_cuda\']])\n                except Exception:\n                    pass          # numero de GPUs mudou entre as VMs\n            if ck.get(\'rng_loader\') is not None and getattr(tr_loader, \'generator\', None):\n                tr_loader.generator.set_state(_byte_cpu(ck[\'rng_loader\']))\n            if ck.get(\'rng_np\') is not None:\n                np.random.set_state(ck[\'rng_np\'])\n            if ck.get(\'rng_py\') is not None:\n                random.setstate(ck[\'rng_py\'])\n            model.load_state_dict(_model_sd)\n            optimizer.load_state_dict(_optim_sd)\n            for pg in optimizer.param_groups:\n                pg[\'lr\'] = _novo[\'lr\']\n\n            # so aqui o estado do treino e efetivamente trocado\n            ep_inicial        = _novo[\'ep_inicial\']\n            best_val_acc      = _novo[\'best\']\n            ref_acc           = _novo[\'ref\']\n            epochs_no_improve = _novo[\'sem\']\n            consec_drops      = _novo[\'drops\']\n            marco_acc         = _novo[\'m_acc\']\n            marco_ep          = _novo[\'m_ep\']\n            history           = _novo[\'hist\']\n            print(f\'    ↺ Retomando da epoca {ep_inicial+1} \'\n                  f\'(best ate aqui {best_val_acc:.2f}%, lr={_novo["lr"]:.2e})\')\n        except Exception as e:\n            print(f\'    (checkpoint descartado, comecando do zero: {e!r})\')\n            ep_inicial, history = 0, []\n            best_val_acc = ref_acc = 0.0\n            epochs_no_improve = consec_drops = 0\n            marco_acc, marco_ep = -1e9, 0\n\n    for epoch in range(ep_inicial, epochs):'),
    ('caminho do checkpoint',
     'def _idx_path(modelo, name):',
     'def _ckpt_path(modelo):\n    """Checkpoint do fold EM ANDAMENTO. Nome fixo: so um fold corre por vez\n    por grupo, e assim o painel sabe o que espelhar sem listar o diretorio.\n    A identidade (arquitetura, fold) vai DENTRO do arquivo e e conferida."""\n    return os.path.join(_drive_dir(modelo), "ckpt_atual.pt")\n\n\ndef _idx_path(modelo, name):'),
]

CABECALHO = '''# -*- coding: utf-8 -*-
"""GERADO POR gerar_busca_hp.py — NAO EDITAR A MAO.

Fonte: {fonte}
Celulas {cel_ds} (H5PyDataset) e {cel_busca} (busca), com hiperparametros
parametrizados por linha de comando.

Diferencas em relacao ao notebook, e por que:
  LEARNING_RATE  1e-3 -> {lr:g}   1e-3 colapsa o QAM (15/15 folds medidos em 19/08/2026)
  LR_PATIENCE    5    -> {pat}      pedido do usuario
  EARLY_STOP     15   -> {es}     acompanha a paciencia maior
  drive_push/pull     -> no-op    o upload ao Drive e feito pelo app LOCAL, para
                                  o token de escopo `drive` nunca sair da maquina

RESSALVA: lr={lr:g} escolhido por medicao em 21/08/2026 (campanha/escolhe_lr.py,
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

PARADA ANTECIPADA: MIN_DELTA={md:g} e LR_MIN={lrm:g} vieram de medicao em
21/08/2026, sobre o history de um fold real do ASK (2L_32-64) que rodou as 120
epocas sem o early stop disparar. Naquele fold o plato foi alcancado na ep. 27
(a 0,25 p.p. do melhor); as 93 epocas seguintes renderam 0,25 p.p., contra um
ruido de p90=0,24 p.p. entre epocas vizinhas -- ganho dentro do ruido.

Com os valores antigos (0.05 / 1e-9) o replay fiel do laco para so na ep. 118:
2% de economia. Com 0.1 / 1e-07 para na ep. 94: 22% por 0,04 p.p.

MIN_DELTA tem de ficar ACIMA do ruido da val_acc; abaixo dele, uma oscilacao
para cima conta como melhora e zera a paciencia. Confira com
campanha/analisa_folds.py se o valor continua acima do ruido no SEU grupo e
arquitetura -- isto foi medido no ASK 2L_32-64, o grupo mais facil e a rede menor.
"""
import argparse
import os
import shlex
import sys

_p = argparse.ArgumentParser(description="Busca de HP de um grupo de modulacao")
_p.add_argument("--grupos", nargs="+", required=True,
                help="grupos a processar, ex.: QAM  (ASK PSK APSK QAM)")
_p.add_argument("--lr", type=float, default={lr:g},
                help="learning rate inicial (default: %(default)g)")
_p.add_argument("--lr-patience", type=int, default={pat},
                help="epocas sem melhora ate reduzir o LR (default: %(default)d)")
_p.add_argument("--early-stop", type=int, default={es},
                help="epocas sem melhora, com LR no piso, ate encerrar (default: %(default)d)")
_p.add_argument("--min-delta", type=float, default={md:g},
                help="p.p. de val_acc que contam como melhora (default: %(default)g)")
_p.add_argument("--lr-min", type=float, default={lrm:g},
                help="piso do LR; o early stop so dispara nele (default: %(default)g)")
_p.add_argument("--stagnation-patience", type=int, default={sp},
                help="epocas sem avanco real ate encerrar o fold (default: %(default)d)")
_p.add_argument("--stagnation-margin", type=float, default={sm:g},
                help="p.p. de val_acc que contam como avanco real (default: %(default)g)")
_p.add_argument("--modelo", choices=["cnn", "resnet"], default="cnn",
                help="cnn = busca de arquitetura; resnet = arquitetura fixa"
                     " de arXiv:1712.04578 (default: %(default)s)")
_p.add_argument("--ckpt-every", type=int, default={ce},
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


'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", default=None,
                    help="caminho local do .ipynb; se omitido, baixa do branch")
    ap.add_argument("--saida", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "busca_hp.py"))
    ap.add_argument("--lr", type=float, default=4.5e-5)
    ap.add_argument("--lr-patience", type=int, default=8)
    ap.add_argument("--early-stop", type=int, default=20)
    ap.add_argument("--min-delta", type=float, default=0.10)
    ap.add_argument("--lr-min", type=float, default=1e-7)
    ap.add_argument("--stagnation-patience", type=int, default=25)
    ap.add_argument("--stagnation-margin", type=float, default=0.25)
    ap.add_argument("--ckpt-every", type=int, default=5)
    a = ap.parse_args()

    if a.notebook:
        fonte = a.notebook
        nb = json.load(open(a.notebook, encoding="utf-8"))
    else:
        fonte = RAW
        print("baixando notebook do branch...")
        with urllib.request.urlopen(RAW) as r:
            nb = json.loads(r.read().decode("utf-8"))

    cells = nb["cells"]
    if len(cells) <= CEL_BUSCA:
        sys.exit("notebook tem %d celulas; esperava mais de %d" % (len(cells), CEL_BUSCA))

    ds = "".join(cells[CEL_DATASET]["source"])
    busca = "".join(cells[CEL_BUSCA]["source"])

    if "class H5PyDataset" not in ds:
        sys.exit("celula %d nao contem H5PyDataset — o notebook mudou de forma" % CEL_DATASET)
    if "def run_all" not in busca:
        sys.exit("celula %d nao contem run_all — o notebook mudou de forma" % CEL_BUSCA)

    aplicadas = []
    for desc, de, para in SUBS:
        n = busca.count(de)
        if n != 1:
            sys.exit("substituicao '%s' casou %d vezes (esperava 1).\n"
                     "  padrao: %r\n"
                     "  o notebook mudou; ajuste SUBS em gerar_busca_hp.py" % (desc, n, de))
        busca = busca.replace(de, para, 1)
        aplicadas.append(desc)

    cabecalho = CABECALHO.format(fonte=fonte, cel_ds=CEL_DATASET, cel_busca=CEL_BUSCA,
                                 lr=a.lr, pat=a.lr_patience, es=a.early_stop,
                                 md=a.min_delta, lrm=a.lr_min,
                                 sp=a.stagnation_patience, sm=a.stagnation_margin,
                                 ce=a.ckpt_every)
    saida = cabecalho + ds.rstrip() + "\n\n\n" + busca

    # Rede de seguranca. A lista SUBS cresceu muito, e uma regeneracao que
    # falhe em silencio produziria um busca_hp.py sem checkpoint, sem parada
    # por estagnacao ou sem TF32 desligado -- e o arquivo PARECERIA correto.
    exigidos = ["ckpt_path", "ckpt_id", "ep_inicial", "rng_torch", "stag_patience",
                "STAGNATION_PATIENCE", "CKPT_EVERY", "_ckpt_path",
                "allow_tf32", "_argv_bh", "STRAT_BY_SNR",
                # AM/FM fora, alvos GROUP/TODAS, ResNet do artigo
                "_alvo_info", "ALVOS_ESPECIAIS", "_carrega_resnet",
                "MODELO_TIPO", "ARQ_RESNET", "resnet.ResNet"]
    proibidos = ["\"FM\":        5", "4: \"AM\""]   # analogicas nao podem voltar
    faltando = [x for x in exigidos if x not in saida]
    voltaram = [x for x in proibidos if x in saida]
    assert not voltaram, ("AM/FM reapareceram na regeneracao: %s" % voltaram)
    assert not faltando, (
        "regeneracao perdeu funcionalidade: %s." % faltando +
        " O notebook fonte mudou e alguma entrada de SUBS deixou de casar.")

    with open(a.saida, "w", encoding="utf-8") as f:
        f.write(saida)

    print("gerado: %s  (%d linhas)" % (a.saida, saida.count("\n") + 1))
    for d in aplicadas:
        print("   aplicada: %s" % d)

    import ast
    try:
        ast.parse(saida)
        print("   sintaxe OK")
    except SyntaxError as e:
        sys.exit("   SINTAXE INVALIDA na linha %s: %s" % (e.lineno, e.msg))


if __name__ == "__main__":
    main()
