"""
finetune.py
Transfer learning across four controlled conditions for this prototype.

Condition 1: 20% data, NO transfer learning
Condition 2: 20% data, WITH transfer learning
Condition 3: 60% data, NO transfer learning
Condition 4: 60% data, WITH transfer learning

Usage:
  python finetune.py --pretrained save_weights/GraphUNet_pretrained_best.pth
                     --dataset hokkaido
"""
import os, argparse, random, torch, matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config        import *
from dataset       import make_finetune_loaders
from models        import GraphUNet, load_pretrained_graph_unet
from loss          import CombinedLoss
from metrics       import print_metrics, accumulate_metrics
from utils.trainer import train_one_epoch, validate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", default="save_weights/GraphUNet_pretrained_best.pth")
    p.add_argument("--dataset",    choices=["hokkaido","niangniangba","bijie"],
                   default="hokkaido")
    p.add_argument("--epochs",     type=int,   default=FINETUNE_EPOCHS)
    p.add_argument("--batch",      type=int,   default=FINETUNE_BATCH)
    p.add_argument("--imgsize",    type=int,   default=IMG_SIZE)
    p.add_argument("--workers",    type=int,   default=0)
    p.add_argument("--conditions", nargs="+",
                   choices=["1","2","3","4"], default=["1","2","3","4"])
    return p.parse_args()


def get_dirs(name):
    return {"hokkaido":(HOK_IMG_DIR,HOK_MASK_DIR),
            "niangniangba":(NGB_IMG_DIR,NGB_MASK_DIR),
            "bijie":(BIJ_IMG_DIR,BIJ_MASK_DIR)}[name]


def build_model(args):
    return GraphUNet(
        in_ch=IN_CHANNELS, num_classes=NUM_CLASSES,
        feat_dim=FEAT_DIM, hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS, pool_ratios=POOL_RATIOS,
        dropout=GCN_DROPOUT, img_size=args.imgsize
    ).to(DEVICE)


def run_condition(cid, use_tl, frac, pretrained_path,
                   img_dir, mask_dir, args, criterion):
    label = (f"{args.dataset}_Cond{cid}_"
             f"{'TL' if use_tl else 'noTL'}_{int(frac*100)}pct")

    # Make results independent of the order in which conditions are run.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"\n{'─'*50}\n  {label}\n{'─'*50}")

    train_loader, val_loader, test_loader = make_finetune_loaders(
        img_dir, mask_dir, data_fraction=frac,
        batch_size=args.batch, img_size=args.imgsize,
        num_workers=args.workers)

    model = build_model(args)
    # The controlled comparison differs only in weight initialisation.
    lr = FINETUNE_LR

    if use_tl:
        model = load_pretrained_graph_unet(model, pretrained_path, DEVICE)

    opt   = torch.optim.SGD(model.parameters(), lr=lr,
                              momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.PolynomialLR(
                opt, total_iters=args.epochs, power=LR_POWER)

    best_f1   = float("-inf")
    best_path = os.path.join(SAVE_DIR, f"{label}_best.pth")
    history   = {'train_loss':[], 'val_loss':[], 'f1':[]}

    for epoch in range(1, args.epochs + 1):
        tl = train_one_epoch(model, train_loader, opt, criterion, DEVICE)
        vl, m = validate(model, val_loader, criterion, DEVICE)
        sched.step()
        history['train_loss'].append(tl); history['val_loss'].append(vl)
        history['f1'].append(m['f1'])

        if epoch % LOG_INTERVAL == 0 or epoch == args.epochs:
            print(f"  Ep{epoch:3d}: tr={tl:.4f} vl={vl:.4f} F1={m['f1']:.2f}%")

        if m['f1'] > best_f1:
            best_f1 = m['f1']
            torch.save(model.state_dict(), best_path)

    # Final test set evaluation
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    _, test_m = validate(model, test_loader, criterion, DEVICE)
    print(f"\n  [{label}] TEST RESULTS:")
    print_metrics(test_m, "    ")
    return best_f1, test_m, history


def main():
    args      = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    criterion = CombinedLoss()

    img_dir, mask_dir = get_dirs(args.dataset)

    print("="*55)
    print("Graph U-Net  —  TRANSFER LEARNING")
    print("="*55)
    print(f"  Dataset    : {args.dataset}")
    print(f"  Pretrained : {args.pretrained}")
    print(f"  Epochs     : {args.epochs}  ImgSize:{args.imgsize}")
    print(f"  Conditions : {args.conditions}")
    print("="*55)

    cond_cfg = {
        "1": (False, SMALL_DATA_FRAC),
        "2": (True,  SMALL_DATA_FRAC),
        "3": (False, FULL_DATA_FRAC),
        "4": (True,  FULL_DATA_FRAC),
    }
    cond_names = {
        "1":"20% data, no TL","2":"20% data, WITH TL",
        "3":"60% data, no TL","4":"60% data, WITH TL"}

    results = {}; histories = {}
    for cid in args.conditions:
        use_tl, frac = cond_cfg[cid]
        best_f1, test_m, hist = run_condition(
            cid, use_tl, frac, args.pretrained,
            img_dir, mask_dir, args, criterion)
        results[cid]   = (best_f1, test_m)
        histories[cid] = hist

    # Summary table
    print(f"\n{'='*55}")
    print(f"RESULTS — {args.dataset}")
    print(f"{'='*55}")
    print(f"{'Condition':<25} {'Recall':>8} {'Spec':>8} {'Prec':>8} {'F1':>8}")
    print("-"*55)
    for cid, (_, m) in results.items():
        print(f"{cond_names.get(cid,''):<25} "
              f"{m['recall']:>7.1f}% {m['specificity']:>7.1f}% "
              f"{m['precision']:>7.1f}% {m['f1']:>7.1f}%")
    print("="*55)

    if "2" in results and "3" in results:
        f2 = results["2"][1]["f1"]
        f3 = results["3"][1]["f1"]
        diff   = abs(f2 - f3)
        print(f"\nTest comparison: 20%+TL ({f2:.1f}%) vs "
              f"60%+noTL ({f3:.1f}%)")
        print(f"  Difference={diff:.1f}%  "
              f"({'within 4 points' if diff < 4 else 'not within 4 points'})")

    # Plot
    colours = {"1":"royalblue","2":"coral","3":"green","4":"purple"}
    labels  = {"1":"20%noTL","2":"20%TL","3":"60%noTL","4":"60%TL"}
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    for cid, h in histories.items():
        c=colours.get(cid,"grey"); l=labels.get(cid,cid)
        ax[0].plot(h['val_loss'], label=l, color=c)
        ax[1].plot(h['f1'],      label=l, color=c)
    ax[0].set_title('Val Loss'); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].set_title('Val F1 (%)'); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.suptitle(f"Graph U-Net TL — {args.dataset}")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"finetune_{args.dataset}.png"), dpi=120)
    plt.close()
    print(f"Curves saved → results/finetune_{args.dataset}.png")


if __name__ == "__main__":
    main()
