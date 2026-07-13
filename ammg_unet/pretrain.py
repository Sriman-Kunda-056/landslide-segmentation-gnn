"""
pretrain.py
Phase 1: Pretrain the simplified AMMG-style model on CAS or Bijie.

Training settings based on the paper:
  Optimizer : SGD, momentum=0.9, weight_decay=0.001
  LR        : 0.01, cubic polynomial decay over 150 epochs
  Loss      : Dice + 0.5 * CrossEntropy

Usage:
    python pretrain.py --dataset bijie          # quick smoke-test on Bijie
    python pretrain.py                       # full pretraining on CAS at 512
    python pretrain.py --dataset cas --epochs 150
"""

import os
import sys
import argparse

import torch
import matplotlib.pyplot as plt

from config import (
    DEVICE,
    SAVE_DIR,
    PRETRAIN_EPOCHS,
    PRETRAIN_BATCH,
    PRETRAIN_LR,
    PRETRAIN_WORKERS,
    SAVE_INTERVAL,
    LOG_INTERVAL,
    LR_POWER,
    MOMENTUM,
    WEIGHT_DECAY,
    SEED,
    IMG_SIZE,
    IN_CHANNELS,
    NUM_CLASSES,
    FEATURES,
    GRB_RATIO,
    DROPOUT_RATE,
    RESULTS_DIR,
    BIJIE_IMG_DIR,
    BIJIE_MASK_DIR,
    CAS_IMG_DIR,
    CAS_MASK_DIR,
    HOK_IMG_DIR,
    HOK_MASK_DIR,
)
from dataset       import make_loaders
from models        import AMGUnet
from loss          import CombinedLoss
from metrics       import print_metrics
from utils.trainer import train_one_epoch, validate, save_checkpoint


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["bijie", "cas", "hokkaido"],
                   default="cas",
                   help="Which dataset to pretrain on (paper default: CAS)")
    p.add_argument("--epochs",  type=int,   default=PRETRAIN_EPOCHS)
    p.add_argument("--batch",   type=int,   default=PRETRAIN_BATCH)
    p.add_argument("--lr",      type=float, default=PRETRAIN_LR)
    p.add_argument("--imgsize", type=int,   default=IMG_SIZE)
    p.add_argument("--resume",  type=str,   default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


def get_dirs(dataset_name):
    return {
        "bijie":    (BIJIE_IMG_DIR,  BIJIE_MASK_DIR),
        "cas":      (CAS_IMG_DIR,  CAS_MASK_DIR),
        "hokkaido": (HOK_IMG_DIR,  HOK_MASK_DIR),
    }[dataset_name]


# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(SEED)

    print("=" * 60)
    print("AMMG-UNet  —  PRETRAINING")
    print("=" * 60)
    print(f"  Dataset : {args.dataset}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  LR      : {args.lr}")
    print(f"  Batch   : {args.batch}")
    print(f"  ImgSize : {args.imgsize}")
    print(f"  Device  : {DEVICE}")
    print("=" * 60)

    # ── Data ─────────────────────────────────────────────────────────
    img_dir, mask_dir = get_dirs(args.dataset)
    print(f"  DataDir : {img_dir}")
    print(f"  MaskDir : {mask_dir}")
    train_loader, val_loader = make_loaders(
        img_dir, mask_dir,
        batch_size=args.batch,
        img_size=args.imgsize,
        num_workers=PRETRAIN_WORKERS
    )

    # Print dataset sizes and batch counts at startup
    try:
        train_samples = len(train_loader.dataset)
    except Exception:
        train_samples = "unknown"
    try:
        val_samples = len(val_loader.dataset)
    except Exception:
        val_samples = "unknown"
    print(f"  Train samples: {train_samples}   Val samples: {val_samples}")
    print(f"  Train batches: {len(train_loader)}   Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────
    model = AMGUnet(
        in_ch=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        features=FEATURES,
        grb_ratio=GRB_RATIO,
        dropout_rate=DROPOUT_RATE,
    ).to(DEVICE)
    print(f"\nModel parameters: {model.count_parameters()/1e6:.1f}M")

    # ── Loss ──────────────────────────────────────────────────────────
    criterion = CombinedLoss()

    # ── Optimizer (paper: SGD, momentum=0.9, wd=0.001) ────────────────
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    # ── Scheduler: cubic polynomial decay (paper: power=3) ────────────
    # lr(e) = lr_0 * (1 - e/max_epochs)^3
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=args.epochs,
        power=LR_POWER
    )

    # ── Resume from checkpoint if given ───────────────────────────────
    start_epoch = 1
    best_f1     = float("-inf")
    history     = {'train_loss': [], 'val_loss': [], 'f1': []}

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_f1     = ckpt.get('best_f1', float("-inf"))
        history     = ckpt.get('history', history)
        print(f"Resumed from epoch {ckpt['epoch']}  best_f1={best_f1:.2f}%")

    # ── Training loop ─────────────────────────────────────────────────
    best_path = os.path.join(SAVE_DIR, "AMGUnet_pretrained_best.pth")
    print(f"  Best checkpoint: {best_path}")
    print(f"  Log interval   : {LOG_INTERVAL} epochs")
    print(f"  Save interval  : {SAVE_INTERVAL} epochs")

    for epoch in range(start_epoch, args.epochs + 1):
        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch:3d}/{args.epochs}  lr={current_lr:.6f}")

        print(f">>> PHASE: TRAINING  (train_samples={train_samples}  train_batches={len(train_loader)})")
        train_loss = train_one_epoch(model, train_loader, optimizer,
                          criterion, DEVICE)

        print(f">>> PHASE: VALIDATING (val_samples={val_samples}  val_batches={len(val_loader)})")
        val_loss, metrics = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['f1'].append(metrics['f1'])

        if epoch % LOG_INTERVAL == 0 or epoch == args.epochs:
            print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
            print_metrics(metrics, prefix="  ")
            print(f"  train_batches={len(train_loader)}  val_batches={len(val_loader)}")

        # Save best checkpoint (by F1)
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), best_path)
            print(f"  *** Best F1={best_f1:.2f}%  saved → {best_path} ***")

        # Periodic checkpoint
        if epoch % SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(SAVE_DIR,
                                      f"pretrain_epoch{epoch:03d}.pth")
            save_checkpoint({
                'epoch': epoch,
                'model_state':     model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_f1':         best_f1,
                'history':         history,
            }, ckpt_path)

    # ── Final summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Pretraining complete.")
    print(f"Best F1:   {best_f1:.2f}%")
    print(f"Saved to:  {best_path}")
    print(f"Next step: python finetune.py --pretrained {best_path}")
    print(f"{'='*60}")

    _plot_history(history, args.dataset)


def _plot_history(history, tag):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='train')
    axes[0].plot(history['val_loss'],   label='val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history['f1'], color='green')
    axes[1].set_title('F1-score (%)'); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"pretrain_{tag}_curves.png")
    plt.savefig(path, dpi=120)
    print(f"Curves saved → {path}")
    plt.close()


if __name__ == "__main__":
    main()
