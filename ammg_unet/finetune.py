"""
finetune.py
Phase 2: Fine-tune the simplified AMMG-style model on a target dataset.

Runs four paper-inspired experimental conditions:
  Condition 1: 20% data, NO transfer learning (scratch)
  Condition 2: 20% data, WITH transfer learning (fine-tune)
  Condition 3: 60% data, NO transfer learning (scratch)
  Condition 4: 60% data, WITH transfer learning (fine-tune)

Target-domain settings:
  LR: 0.001 for every condition
  Epochs: 50
  All layers updated (no freezing)

Usage:
  python finetune.py --pretrained save_weights/AMGUnet_pretrained_best.pth
                     --dataset hokkaido
"""

import os
import argparse
import random

import numpy as np

import torch
import matplotlib.pyplot as plt

from config        import *
from dataset       import make_finetune_loaders
from models        import AMGUnet
from loss          import CombinedLoss
from metrics       import print_metrics, accumulate_metrics
from utils.trainer import train_one_epoch, validate, save_checkpoint


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", type=str,
                   default="save_weights/AMGUnet_pretrained_best.pth",
                   help="Path to pretrained weights from pretrain.py")
    p.add_argument("--dataset", choices=["hokkaido", "bijie"],
                   default="hokkaido")
    p.add_argument("--epochs",  type=int,   default=FINETUNE_EPOCHS)
    p.add_argument("--batch",   type=int,   default=FINETUNE_BATCH)
    p.add_argument("--imgsize", type=int,   default=IMG_SIZE)
    p.add_argument("--workers", type=int,   default=FINETUNE_WORKERS)
    p.add_argument(
        "--augment", action="store_true",
        help="Opt in to random target-domain flips (off by default)")
    p.add_argument("--conditions", nargs="+",
                   choices=["1","2","3","4"], default=["1","2","3","4"],
                   help="Which conditions to run (default: all 4)")
    return p.parse_args()


def get_dirs(name):
    return {
        "hokkaido": (HOK_IMG_DIR, HOK_MASK_DIR, FINETUNE_LABEL_DIR),
        "bijie": (BIJIE_IMG_DIR, BIJIE_MASK_DIR, None),
    }[name]


def load_pretrained(model, path, device):
    """Load pretrained weights into model with informative logging."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {path}\n"
            f"Run pretrain.py first."
        )
    state = torch.load(path, map_location=device)
    # handle both raw state_dict and full checkpoint dict
    if isinstance(state, dict) and 'model_state' in state:
        state = state['model_state']
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing:
        print(f"  [warn] missing keys: {missing}")
    print(f"  Pretrained weights loaded from {path}")
    return model


def build_optimizer_scheduler(model, lr, epochs):
    """SGD + cubic polynomial decay — exactly as paper specifies."""
    opt = torch.optim.SGD(model.parameters(),
                           lr=lr,
                           momentum=MOMENTUM,
                           weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.PolynomialLR(
                opt, total_iters=epochs, power=LR_POWER)
    return opt, sched


# ─────────────────────────────────────────────────────────────────────────────

def run_condition(cond_id, use_tl, data_frac,
                  pretrained_path, img_dir, mask_dir,
                  label_dir, args, criterion, device):
    """
    Run one experimental condition end-to-end.
    Returns: best_f1 (float), final_metrics (dict), history (dict)
    """
    label = (f"{args.dataset}_Cond{cond_id}_"
             f"{'TL' if use_tl else 'noTL'}_{int(data_frac*100)}pct")

    # Make results independent of the order in which conditions are run.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"  TL={use_tl}  data={int(data_frac*100)}%  epochs={args.epochs}")
    print(f"{'─'*55}")

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, test_dataset = make_finetune_loaders(
        img_dir, mask_dir,
        data_fraction=data_frac,
        batch_size=args.batch,
        img_size=args.imgsize,
        label_dir=label_dir,
        num_workers=args.workers,
        augment_train=args.augment,
    )

    # ── Model ──────────────────────────────────────────────────────────
    model = AMGUnet(
        in_ch=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        features=FEATURES,
        grb_ratio=GRB_RATIO,
        dropout_rate=DROPOUT_RATE,
    ).to(device)

    if use_tl:
        # Transfer learning: initialise from pretrained weights
        model = load_pretrained(model, pretrained_path, device)

    # Keep target-domain hyperparameters identical across TL/no-TL conditions.
    # The controlled experiment should differ only in weight initialisation.
    lr = FINETUNE_LR

    opt, sched = build_optimizer_scheduler(model, lr, args.epochs)

    # ── Training loop ──────────────────────────────────────────────────
    best_f1   = float("-inf")
    best_path = os.path.join(SAVE_DIR, f"{label}_best.pth")
    history   = {'train_loss': [], 'val_loss': [], 'f1': []}

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, criterion, device)
        vl_loss, metrics = validate(model, val_loader, criterion, device)
        sched.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['f1'].append(metrics['f1'])

        if epoch % LOG_INTERVAL == 0 or epoch == args.epochs:
            print(f"  Ep{epoch:3d}: tr={tr_loss:.4f}  vl={vl_loss:.4f}  "
                  f"F1={metrics['f1']:.2f}%")

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), best_path)

    # ── Final evaluation on TEST set ───────────────────────────────────
    model.load_state_dict(torch.load(best_path, map_location=device))
    _, test_metrics = validate(model, test_loader, criterion, device)

    print(f"\n  [{label}] TEST RESULTS (best checkpoint):")
    print_metrics(test_metrics, prefix="    ")

    return best_f1, test_metrics, history


# ─────────────────────────────────────────────────────────────────────────────

def main():
    args  = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = DEVICE
    criterion = CombinedLoss()

    img_dir, mask_dir, label_dir = get_dirs(args.dataset)

    print("=" * 60)
    print("AMMG-UNet  —  FINE-TUNING  (Transfer Learning)")
    print("=" * 60)
    print(f"  Dataset    : {args.dataset}")
    print(f"  Pretrained : {args.pretrained}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Conditions : {args.conditions}")
    print(f"  Device     : {device}")
    print("=" * 60)

    # Map condition id → (use_tl, data_fraction)
    cond_cfg = {
        "1": (False, SMALL_DATA_FRAC),   # 20%, no TL
        "2": (True,  SMALL_DATA_FRAC),   # 20%, TL
        "3": (False, FULL_DATA_FRAC),    # 60%, no TL
        "4": (True,  FULL_DATA_FRAC),    # 60%, TL
    }

    all_results = {}
    all_histories = {}

    for cid in args.conditions:
        use_tl, frac = cond_cfg[cid]
        best_f1, test_m, hist = run_condition(
            cid, use_tl, frac,
            args.pretrained, img_dir, mask_dir,
            label_dir, args, criterion, device
        )
        all_results[cid]   = (use_tl, frac, best_f1, test_m)
        all_histories[cid] = hist

    # ── Print summary table ────────────────────────────────────────────
    cond_names = {
        "1": "20% data, no TL",
        "2": "20% data, WITH TL",
        "3": "60% data, no TL",
        "4": "60% data, WITH TL",
    }
    print(f"\n{'='*60}")
    print(f"RESULTS TABLE  —  {args.dataset}")
    print(f"{'='*60}")
    print(f"{'Condition':<25} {'Recall':>8} {'Spec':>8} {'Prec':>8} {'F1':>8}")
    print("-" * 60)
    for cid, (_, _, _, m) in all_results.items():
        print(f"{cond_names.get(cid,''):<25} "
              f"{m['recall']:>7.1f}% "
              f"{m['specificity']:>7.1f}% "
              f"{m['precision']:>7.1f}% "
              f"{m['f1']:>7.1f}%")
    print("=" * 60)

    # Compare the common held-out test set, not best validation scores.
    if "2" in all_results and "3" in all_results:
        f1_2 = all_results["2"][3]["f1"]
        f1_3 = all_results["3"][3]["f1"]
        diff = abs(f1_2 - f1_3)
        print(f"\nTest comparison: 20%+TL ({f1_2:.1f}%) vs 60%+noTL ({f1_3:.1f}%)")
        print(f"  Difference: {diff:.1f}%  "
              f"({'within 4 points' if diff < 4 else 'not within 4 points'})")

    # ── Save curves ───────────────────────────────────────────────────
    _plot_all_conditions(all_histories, args.dataset)


def _plot_all_conditions(histories, tag):
    colours = {"1": "royalblue", "2": "coral", "3": "green", "4": "purple"}
    labels  = {"1":"20%,noTL","2":"20%,TL","3":"60%,noTL","4":"60%,TL"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for cid, h in histories.items():
        c = colours.get(cid, "grey")
        l = labels.get(cid, cid)
        axes[0].plot(h['val_loss'], label=l, color=c)
        axes[1].plot(h['f1'],      label=l, color=c)

    axes[0].set_title('Val Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title('Val F1 (%)'); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 100)
    plt.suptitle(f"Fine-tuning results — {tag}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"finetune_{tag}_curves.png")
    plt.savefig(path, dpi=120)
    print(f"Curves saved → {path}")
    plt.close()


if __name__ == "__main__":
    main()
