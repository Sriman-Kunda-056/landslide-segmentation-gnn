"""
pretrain.py
Pretrain Graph U-Net on source domain (Bijie or CAS).

Usage:
  python pretrain.py --dataset bijie --epochs 5   # quick smoke test
  python pretrain.py --dataset bijie              # full pretrain
  python pretrain.py --dataset cas                # paper-scale
"""
import os, sys, argparse, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config        import *
from dataset       import make_loaders
from models        import GraphUNet
from loss          import CombinedLoss
from metrics       import print_metrics
from utils.trainer import train_one_epoch, validate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  choices=["bijie","cas","hokkaido"], default="bijie")
    p.add_argument("--epochs",   type=int,   default=PRETRAIN_EPOCHS)
    p.add_argument("--batch",    type=int,   default=PRETRAIN_BATCH)
    p.add_argument("--lr",       type=float, default=PRETRAIN_LR)
    p.add_argument("--imgsize",  type=int,   default=IMG_SIZE)
    p.add_argument("--resume",   type=str,   default=None)
    return p.parse_args()


def get_dirs(name):
    return {"bijie":(BIJ_IMG_DIR,BIJ_MASK_DIR),
            "cas":(CAS_IMG_DIR,CAS_MASK_DIR),
            "hokkaido":(HOK_IMG_DIR,HOK_MASK_DIR)}[name]


def main():
    args = parse_args()
    torch.manual_seed(SEED)

    print("="*55)
    print("Graph U-Net  —  PRETRAINING")
    print("="*55)
    print(f"  Dataset : {args.dataset}")
    print(f"  Epochs  : {args.epochs}  LR:{args.lr}  ImgSize:{args.imgsize}")
    print(f"  Device  : {DEVICE}")
    print("="*55)

    img_dir, mask_dir = get_dirs(args.dataset)
    train_loader, val_loader = make_loaders(
        img_dir, mask_dir,
        batch_size=args.batch, img_size=args.imgsize, num_workers=0)

    model = GraphUNet(
        in_ch=IN_CHANNELS, num_classes=NUM_CLASSES,
        feat_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS, pool_ratios=POOL_RATIOS,
        dropout=GCN_DROPOUT, img_size=args.imgsize
    ).to(DEVICE)

    print(f"\nModel params: {model.count_parameters():,}")

    criterion = CombinedLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                  momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
                    optimizer, total_iters=args.epochs, power=LR_POWER)

    start = 1; best_f1 = 0.0
    history = {'train_loss':[], 'val_loss':[], 'f1':[]}
    best_path = os.path.join(SAVE_DIR, "GraphUNet_pretrained_best.pth")

    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ck['model_state'])
        optimizer.load_state_dict(ck['optimizer_state'])
        scheduler.load_state_dict(ck['scheduler_state'])
        start    = ck['epoch'] + 1
        best_f1  = ck.get('best_f1', 0.0)
        history  = ck.get('history', history)
        print(f"Resumed from epoch {ck['epoch']}")

    for epoch in range(start, args.epochs + 1):
        lr_now = scheduler.get_last_lr()[0]
        print(f"\nEp {epoch:3d}/{args.epochs}  lr={lr_now:.6f}")
        tl = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        vl, m  = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['f1'].append(m['f1'])

        if epoch % LOG_INTERVAL == 0 or epoch == args.epochs:
            print(f"  train={tl:.4f}  val={vl:.4f}")
            print_metrics(m, "  ")

        if m['f1'] > best_f1:
            best_f1 = m['f1']
            torch.save(model.state_dict(), best_path)
            print(f"  *** best F1={best_f1:.2f}%  saved ***")

        if epoch % SAVE_INTERVAL == 0:
            torch.save({'epoch':epoch,'model_state':model.state_dict(),
                        'optimizer_state':optimizer.state_dict(),
                        'scheduler_state':scheduler.state_dict(),
                        'best_f1':best_f1,'history':history},
                       os.path.join(SAVE_DIR,f"pretrain_ep{epoch:03d}.pth"))

    print(f"\nDone. Best F1: {best_f1:.2f}%  saved → {best_path}")
    print(f"Next: python finetune.py --pretrained {best_path} --dataset hokkaido")

    fig, ax = plt.subplots(1,2,figsize=(10,4))
    ax[0].plot(history['train_loss'],label='train'); ax[0].plot(history['val_loss'],label='val')
    ax[0].set_title('Loss'); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(history['f1'],color='green'); ax[1].set_title('F1 (%)'); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,f"pretrain_{args.dataset}_curves.png"),dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
