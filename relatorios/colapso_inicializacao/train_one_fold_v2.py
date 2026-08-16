# -*- coding: utf-8 -*-
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║   TREINO DE FOLD v2 — detecção de colapso + RESTART                          ║
# ║                                                                              ║
# ║  Substitui `train_one_fold`. Mudança central:                                ║
# ║    Um fold "morto" (loss == ln(C), acc == 1/C) tem GRADIENTE ZERO.           ║
# ║    Reduzir o LR não o revive — só congela o estado morto mais devagar.       ║
# ║    A única saída é RE-INICIALIZAR os pesos com outra seed.                   ║
# ║                                                                              ║
# ║  Evidência (PSK, 7 classes, kfold_fold_results.json):                        ║
# ║    38/60 folds travados em 14.29% com trloss=1.9459=ln(7) constante,         ║
# ║    inalterado após o LR cair de 1e-3 para 2.5e-4.                            ║
# ║    Mesma arquitetura, folds diferentes: 14.29% vs 96.00%.                    ║
# ║    ⇒ o que decide não é o LR, é a inicialização.                             ║
# ║                                                                              ║
# ║  Estratégia em 3 camadas:                                                    ║
# ║    1. PREVENIR  — init explícita, warmup de LR, clip de gradiente            ║
# ║    2. DETECTAR  — intra-época (rápido) e por época, 2 strikes consecutivos   ║
# ║    3. REAGIR    — restart com nova seed + LR menor, não redução de LR isolada║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# ▶▶  CONFIGURAÇÃO NOVA — acrescente ao bloco de config do notebook  ◀◀
# ══════════════════════════════════════════════════════════════════════════════

# ── Prevenção ──────────────────────────────────────────────────────────────────
WARMUP_STEPS      = 300     # passos de aquecimento linear (lr/100 → lr)
GRAD_CLIP         = 5.0     # norma máxima do gradiente; None desativa
HEAD_INIT_GAIN    = 0.01    # ganho da última camada — logits pequenos no início

# ── Detecção de colapso ────────────────────────────────────────────────────────
# "Morto" = a rede prevê a distribuição uniforme. Dois sinais têm que bater:
#   • acurácia colada no chance level 1/C
#   • loss colada em ln(C)   ← este é o sinal decisivo, muito menos ruidoso
DEAD_ACC_TOL      = 0.15    # acc <= chance*(1+0.15)
DEAD_LOSS_TOL     = 0.005   # |loss - ln(C)| <= 0.005
DEAD_ACC_PROGRESS = 0.5     # p.p. de ganho de acurácia entre checagens que caracteriza vida
DEAD_STRIKES      = 2       # checagens consecutivas para confirmar (pedido do usuário)
DEAD_PROBE_BATCH  = 150     # checagem intra-época a cada N batches (detecção rápida)

# ── Reação ─────────────────────────────────────────────────────────────────────
MAX_RESTARTS      = 4       # tentativas de re-inicialização por fold
RESTART_LR_FACTOR = 0.3     # LR da tentativa n = lr0 * 0.3**n
RESTART_LR_MIN    = 1e-5    # abaixo disso, desistir do fold em vez de insistir

# ── Defaults do scheduler ──────────────────────────────────────────────────────
# Definidos aqui para o módulo funcionar isolado. No notebook, o bloco de config
# redefine estes nomes e estes valores ficam sem efeito.
LR_FACTOR_BASE    = 0.5
LR_FACTOR_FLOOR   = 0.01
MIN_DELTA         = 0.05


# ══════════════════════════════════════════════════════════════════════════════
# CAMADA 1 — PREVENÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def init_weights(model, head_gain=HEAD_INIT_GAIN):
    """
    Kaiming nas conv/linear (a default do PyTorch é Kaiming uniform com a=√5,
    que subestima o ganho para ReLU e favorece ativação morta em redes fundas),
    e última camada com ganho pequeno para os logits nascerem próximos de zero.
    """
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
            last_linear = m

    # Cabeça de classificação: logits ~0 no início ⇒ softmax uniforme SEM
    # saturação. Uniforme por logits pequenos ainda tem gradiente; uniforme
    # por ReLU morta não tem. A diferença é exatamente o bug que estamos matando.
    if last_linear is not None:
        with torch.no_grad():
            last_linear.weight.mul_(head_gain)
    return model


def _warmup_lr(step, base_lr, warmup_steps=WARMUP_STEPS):
    """Aquecimento linear de base_lr/100 até base_lr ao longo de warmup_steps."""
    if warmup_steps <= 0 or step >= warmup_steps:
        return base_lr
    frac = step / float(warmup_steps)
    return base_lr * (0.01 + 0.99 * frac)


# ══════════════════════════════════════════════════════════════════════════════
# CAMADA 2 — DETECÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def is_dead(loss, acc_pct, num_classes, prev_acc=None,
            acc_tol=DEAD_ACC_TOL, loss_tol=DEAD_LOSS_TOL,
            acc_progress=DEAD_ACC_PROGRESS):
    """
    True quando a rede está presa na solução uniforme.

    Exige TRÊS sinais, e o terceiro é o que separa morte de lentidão:

      1. acurácia colada no chance level 1/C
      2. loss colada em ln(C) — valor exato da entropia cruzada de uma
         predição uniforme
      3. acurácia SEM PROGRESSO — não subiu mais que `acc_progress` p.p.
         desde a checagem anterior

    Sobre o item 3: a primeira versão usava estagnação da *loss*, e falhou nos
    dois sentidos. Uma rede viva porém lenta foi marcada como morta (saiu de
    25% para 33% logo depois do alarme); e uma rede genuinamente colapsando
    passou batido, porque sua loss ainda escorregava ~0.0035/época — ela estava
    convergindo PARA ln(C), e esse movimento foi lido como vida.

    A tendência da acurácia não tem essa ambiguidade: descer em direção ao
    uniforme mantém a acurácia parada no chance, enquanto aprender devagar a
    faz subir. É também o sinal que o usuário propôs originalmente.

    `prev_acc=None` (primeira checagem) pula o critério — com um único ponto
    não há como medir tendência.
    """
    chance_acc  = 100.0 / num_classes
    chance_loss = math.log(num_classes)

    if not (acc_pct <= chance_acc * (1.0 + acc_tol)
            and abs(loss - chance_loss) <= loss_tol):
        return False

    if prev_acc is None:
        return True

    return (acc_pct - prev_acc) <= acc_progress


# ══════════════════════════════════════════════════════════════════════════════
# CAMADA 3 — TREINO COM RESTART
# ══════════════════════════════════════════════════════════════════════════════

def _train_attempt(model, tr_loader, vl_loader, device, num_classes,
                   lr, epochs, lr_patience, early_stop, lr_min,
                   lr_factor_base, lr_factor_floor, min_delta, attempt_tag):
    """
    Uma tentativa de treino. Retorna (best_val_acc, history, dead).
    `dead=True` significa: abortada por colapso confirmado — o chamador
    deve re-inicializar, não reduzir o LR.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    chance_acc = 100.0 / num_classes

    # Warmup limitado a uma FRAÇÃO do orçamento total de passos, não a uma época.
    #   • Sem limite algum: 300 passos fixos num loader curto (63 batches/época)
    #     comiam 5 épocas inteiras — o treino todo rodava com LR ~50x abaixo do
    #     alvo e o detector disparava contra uma rede que só estava aquecendo.
    #   • Limitando a 1 época: caía para 63 passos, MENOS que os 150 que a
    #     ablação mostrou serem suficientes — e o colapso voltava.
    # 30% do orçamento acomoda os dois regimes: em treino real (2600 batches/
    # época × 120 épocas) quem manda é WARMUP_STEPS; em treino curto o teto
    # proporcional impede o aquecimento de engolir a corrida inteira.
    total_steps  = max(1, epochs * len(tr_loader))
    warmup_steps = min(WARMUP_STEPS, max(1, int(0.30 * total_steps)))

    best_val_acc      = 0.0
    # ── CORREÇÃO: a referência de paciência começa NO CHANCE LEVEL ──────────────
    # No código antigo `ref_acc = 0.0`, então a primeira época (mesmo colapsada,
    # a 1/C) contava como "melhora" e zerava a paciência. Isso atrasava toda a
    # reação do scheduler em LR_PATIENCE épocas logo de saída.
    ref_acc           = chance_acc
    best_state        = None
    epochs_no_improve = 0
    consec_drops      = 0
    history           = []
    global_step       = 0
    # Contadores SEPARADOS. Com um contador único, a sonda e a checagem de
    # época zeravam o strike uma da outra e o colapso nunca chegava a 2.
    strikes_probe     = 0
    strikes_epoch     = 0
    prev_probe_acc    = None   # acurácia da sonda anterior — mede tendência
    prev_epoch_acc    = None   # acurácia da época anterior — idem

    model.to(device)

    for epoch in range(epochs):

        # ── Treino ────────────────────────────────────────────────────────────
        model.train()
        tr_loss = tr_correct = tr_total = 0
        probe_loss = probe_correct = probe_total = probe_n = 0

        pbar = tqdm(tr_loader, desc=f"    {attempt_tag} Ép {epoch+1:>3}/{epochs} [tr]",
                    leave=False, ncols=80)

        for xb, yb in pbar:
            # Warmup: sobrescreve o LR nos primeiros passos
            if global_step < warmup_steps:
                wlr = _warmup_lr(global_step, lr, warmup_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = wlr
            elif global_step == warmup_steps:
                # Fecha o aquecimento exatamente no LR alvo antes de entregar
                # o controle ao scheduler de platô.
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()

            if GRAD_CLIP is not None:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()
            global_step += 1

            batch_correct = (out.argmax(1) == yb).sum().item()
            tr_loss    += loss.item()
            tr_correct += batch_correct
            tr_total   += yb.size(0)

            probe_loss    += loss.item()
            probe_correct += batch_correct
            probe_total   += yb.size(0)
            probe_n       += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # ── Sonda intra-época: detecta morte em ~150 batches, não em 1 época
            # Só depois do warmup — durante o aquecimento a rede legitimamente
            # ainda está perto do uniforme e um alarme aqui seria falso positivo.
            if (probe_n >= DEAD_PROBE_BATCH and global_step > warmup_steps):
                p_loss = probe_loss / probe_n
                p_acc  = 100.0 * probe_correct / probe_total
                dead_now = is_dead(p_loss, p_acc, num_classes, prev_probe_acc)
                prev_probe_acc = p_acc
                probe_loss = probe_correct = probe_total = probe_n = 0

                if dead_now:
                    strikes_probe += 1
                    print(f"    ⚠  {attempt_tag} sonda: loss={p_loss:.4f} "
                          f"≈ ln({num_classes})={math.log(num_classes):.4f}, "
                          f"acc={p_acc:.2f}% ≈ {chance_acc:.2f}%, sem progresso "
                          f"→ strike {strikes_probe}/{DEAD_STRIKES}")
                    if strikes_probe >= DEAD_STRIKES:
                        pbar.close()
                        print(f"    💀 {attempt_tag} COLAPSO confirmado na época "
                              f"{epoch+1} — abortando para re-inicializar")
                        return best_val_acc, history, True
                else:
                    strikes_probe = 0

        # ── Validação ─────────────────────────────────────────────────────────
        model.eval()
        vl_loss = vl_correct = vl_total = 0
        with torch.no_grad():
            for xb, yb in vl_loader:
                xb, yb  = xb.to(device), yb.to(device)
                out     = model(xb)
                loss    = criterion(out, yb)
                vl_loss    += loss.item()
                vl_correct += (out.argmax(1) == yb).sum().item()
                vl_total   += yb.size(0)

        tr_acc   = 100.0 * tr_correct / tr_total
        vl_acc   = 100.0 * vl_correct / vl_total
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
            "attempt"   : attempt_tag,
        })

        print(f"    {attempt_tag} Ép {epoch+1:>3}/{epochs} │ "
              f"tr={tr_loss:.4f}/{tr_acc:.2f}% │ "
              f"vl={vl_loss:.4f}/{vl_acc:.2f}% │ "
              f"lr={cur_lr:.2e}  "
              f"{'★' if vl_acc > best_val_acc else ''}")

        # ── Detecção por época (rede de segurança da sonda) ────────────────────
        # Só depois do aquecimento: durante o warmup a rede treina com LR muito
        # abaixo do alvo e fica legitimamente perto do chance level. Marcar isso
        # como colapso é falso positivo — e foi o que derrubou o cenário 1 do
        # teste antes desta guarda existir.
        if global_step <= warmup_steps:
            prev_epoch_acc = tr_acc
        else:
            dead_now = is_dead(tr_loss, tr_acc, num_classes, prev_epoch_acc)
            prev_epoch_acc = tr_acc
            if dead_now:
                strikes_epoch += 1
                print(f"    ⚠  {attempt_tag} época morta (acc parada em "
                      f"{tr_acc:.2f}%, loss {tr_loss:.4f}) → strike "
                      f"{strikes_epoch}/{DEAD_STRIKES}")
                if strikes_epoch >= DEAD_STRIKES:
                    print(f"    💀 {attempt_tag} COLAPSO confirmado — re-inicializando")
                    return best_val_acc, history, True
            else:
                strikes_epoch = 0

        # ── Checkpoint ────────────────────────────────────────────────────────
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = copy.deepcopy(model.state_dict())

        # ── Paciência ─────────────────────────────────────────────────────────
        if vl_acc > ref_acc + min_delta:
            ref_acc           = vl_acc
            epochs_no_improve = 0
            consec_drops      = 0
        else:
            epochs_no_improve += 1

        # ── Redução escalonada do LR (mantida do original) ─────────────────────
        if epochs_no_improve >= lr_patience and cur_lr > lr_min:
            factor = max(lr_factor_base ** (consec_drops + 1), lr_factor_floor)
            new_lr = max(cur_lr * factor, lr_min)
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr
            consec_drops     += 1
            epochs_no_improve = 0
            print(f"    ↘  LR reduzido: {cur_lr:.2e} → {new_lr:.2e}  "
                  f"(fator={factor:.4f} │ redução #{consec_drops})")

        elif epochs_no_improve >= early_stop and cur_lr <= lr_min * 1.001:
            print(f"    🛑 Early stopping na época {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_acc, history, False


def train_one_fold(model_fn, tr_loader, vl_loader, device, num_classes,
                   lr, epochs, lr_patience, early_stop, lr_min,
                   fold_seed=None,
                   lr_factor_base=None, lr_factor_floor=None, min_delta=None,
                   max_restarts=MAX_RESTARTS):
    """
    Drop-in do `train_one_fold` original, com duas diferenças de assinatura:

      • `model_fn`     — FÁBRICA de modelo (callable sem argumentos), não uma
                         instância. Necessário porque o restart precisa criar
                         uma rede nova; reciclar a instância morta não adianta.
      • `num_classes`  — usado para calcular o chance level (1/C e ln C).

    Retorna (best_val_acc, history, info) onde info registra o que aconteceu:
      {"restarts": n, "final_lr0": lr, "status": "ok"|"dead"}

    Chamada típica no lugar do original:

        best_val_acc, history, info = train_one_fold(
            model_fn    = lambda: FlexCNN(num_classes=num_classes, arch=arch,
                                          classifier=CLASSIFIER_HEAD,
                                          dropout=DROPOUT),
            tr_loader   = tr_loader,
            vl_loader   = vl_loader,
            device      = device,
            num_classes = num_classes,
            lr          = LEARNING_RATE,
            epochs      = EPOCHS_PER_FOLD,
            lr_patience = LR_PATIENCE,
            early_stop  = EARLY_STOP,
            lr_min      = LR_MIN,
            fold_seed   = fold_seed,
        )
    """
    if lr_factor_base  is None: lr_factor_base  = LR_FACTOR_BASE
    if lr_factor_floor is None: lr_factor_floor = LR_FACTOR_FLOOR
    if min_delta       is None: min_delta       = MIN_DELTA

    base_seed  = fold_seed if fold_seed is not None else 0
    cur_lr0    = lr
    full_hist  = []

    for attempt in range(max_restarts + 1):
        tag = f"[t{attempt}]"

        # Seed distinta por tentativa — é o ponto inteiro do restart.
        # Mesma seed + mesmo LR reproduziria exatamente a mesma rede morta.
        seed = base_seed + 9973 * attempt
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = init_weights(model_fn())

        if attempt > 0:
            print(f"\n    🔄 Tentativa {attempt}/{max_restarts} — "
                  f"nova seed={seed}, lr={cur_lr0:.2e}")

        best_val_acc, history, dead = _train_attempt(
            model, tr_loader, vl_loader, device, num_classes,
            lr=cur_lr0, epochs=epochs, lr_patience=lr_patience,
            early_stop=early_stop, lr_min=lr_min,
            lr_factor_base=lr_factor_base, lr_factor_floor=lr_factor_floor,
            min_delta=min_delta, attempt_tag=tag,
        )
        full_hist.extend(history)

        if not dead:
            info = {"restarts": attempt, "final_lr0": cur_lr0, "status": "ok",
                    "seed": seed}
            if attempt > 0:
                print(f"    ✅ Recuperado após {attempt} restart(s) "
                      f"— val_acc={best_val_acc:.2f}%")
            return best_val_acc, full_hist, info

        # Morreu: baixa o LR inicial da PRÓXIMA tentativa. O LR menor é seguro
        # secundário — quem realmente resolve é a re-inicialização — mas ajuda
        # nos casos em que o primeiro passo grande é o que mata a rede.
        cur_lr0 *= RESTART_LR_FACTOR
        if cur_lr0 < RESTART_LR_MIN:
            print(f"    ⛔ LR inicial abaixo do piso ({RESTART_LR_MIN:.1e}) — "
                  f"desistindo deste fold em vez de queimar épocas")
            break

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    info = {"restarts": attempt, "final_lr0": cur_lr0, "status": "dead",
            "seed": base_seed}
    print(f"    ❌ Fold não convergiu após {attempt} restarts — marcado como falho")
    return 0.0, full_hist, info
