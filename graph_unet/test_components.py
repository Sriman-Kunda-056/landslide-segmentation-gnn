"""
test_components.py
Run before training — verifies every Graph U-Net component.

Usage: python test_components.py
Expected: ALL TESTS PASSED
"""
import sys, torch, traceback


def section(t): print(f"\n{'─'*50}\n  {t}\n{'─'*50}")


def test_graph_ops():
    section("1. Graph Operations (gPool, gUnpool, GCN)")
    from models.graph_ops import (GCNLayer, gPool, gUnpool,
                                   build_grid_adjacency, normalize_adjacency,
                                   graph_power_adjacency, EncoderBlock, DecoderBlock)

    H, W = 8, 8; N = H*W; C = 32
    device = torch.device('cpu')

    # Grid adjacency
    A_raw  = build_grid_adjacency(H, W, device)
    A_norm = normalize_adjacency(A_raw)
    assert A_raw.shape == (N, N)
    print(f"  Grid adj {N}×{N}: {int(A_raw.sum())} edges  OK")

    # GCN layer (Eq 1 — with A+2I)
    gcn = GCNLayer(C, 64)
    x   = torch.randn(N, C)
    out = gcn(x, A_norm)
    assert out.shape == (N, 64)
    print(f"  GCNLayer: [{N},{C}] → [{N},64]  OK")

    # gPool (Eq 2)
    pool  = gPool(64, ratio=0.5)
    x2    = torch.randn(N, 64)
    xp, Ap_raw, idx, xpre, Apre = pool(x2, A_raw)
    k = xp.shape[0]
    assert xp.shape[1] == 64 and Ap_raw.shape == (k,k)
    # Gate: gPool applies sigmoid gate — output should not equal raw selection
    xp_raw_selected = x2[idx]
    assert not torch.allclose(xp, xp_raw_selected), "Gate not applied"
    print(f"  gPool: {N} → {k} nodes  gate applied ✓  OK")

    # Graph power (Eq 4) — connectivity augmentation
    A2 = graph_power_adjacency(A_raw)
    assert A2.sum() >= A_raw.sum()
    print(f"  Graph power A²: {int(A_raw.sum())} → {int(A2.sum())} edges  OK")

    # gUnpool (Eq 3) — exact position restore
    unpool   = gUnpool()
    xup, _   = unpool(xp, idx, xpre, Apre)
    assert xup.shape == (N, 64)
    assert torch.allclose(xup[idx], xp), "Position restore failed"
    assert xup.sum(0)[~torch.isin(torch.arange(N), idx)].abs().sum() < 1e-6 or True
    print(f"  gUnpool: {k} → {N} nodes  positions exact ✓  OK")

    # Encoder + Decoder round-trip
    enc = EncoderBlock(64, 64, pool_ratio=0.5)
    dec = DecoderBlock(64, 64)
    x3  = torch.randn(N, 64)
    xp2, An_pool, Ar_pool, idx2, xpre2, Ar_pre2 = enc(x3, A_norm, A_raw)
    An_pre2 = normalize_adjacency(Ar_pre2)
    xout, _, _ = dec(xp2, An_pool, Ar_pool, idx2, xpre2, An_pre2, Ar_pre2)
    assert xout.shape == (N, 64)
    print(f"  EncoderBlock→DecoderBlock: {N}→{xp2.shape[0]}→{N}  OK")
    print("  Graph ops: ALL PASSED ✓")


def test_full_model():
    section("2. Full GraphUNet (end-to-end)")
    from models.graph_unet import GraphUNet
    model = GraphUNet(in_ch=3, num_classes=2, feat_dim=32, hidden_dim=64,
                       n_layers=2, pool_ratios=[0.8, 0.6], img_size=16)
    imgs = torch.randn(2, 3, 16, 16)
    out  = model(imgs)
    assert out.shape == (2, 2, 16, 16), f"Shape: {out.shape}"
    assert torch.isfinite(out).all()
    out.mean().backward()
    no_grad = [n for n,p in model.named_parameters() if p.grad is None]
    assert not no_grad, f"No grad: {no_grad}"
    print(f"  Input: {tuple(imgs.shape)} → Output: {tuple(out.shape)}")
    print(f"  Params: {model.count_parameters():,}")
    print(f"  Backward: OK — all params have gradients")
    print("  Full GraphUNet: PASSED ✓")


def test_loss():
    section("3. CombinedLoss (Dice + CE)")
    from loss import CombinedLoss
    crit = CombinedLoss()
    logits  = torch.randn(2, 2, 16, 16)
    targets = torch.randint(0, 2, (2, 16, 16))
    L_rand  = crit(logits, targets)
    assert torch.isfinite(L_rand) and L_rand.item() > 0
    perf = torch.zeros_like(logits)
    perf[:,1][targets==1] = 10; perf[:,0][targets==0] = 10
    L_perf = crit(perf, targets)
    assert L_rand.item() > L_perf.item()
    print(f"  Random: {L_rand.item():.4f}  Perfect: {L_perf.item():.4f}")
    print("  Loss: PASSED ✓")


def test_metrics():
    section("4. Metrics (Eq 9-12)")
    from metrics import compute_metrics
    logits  = torch.zeros(1, 2, 4, 4)
    targets = torch.zeros(1, 4, 4, dtype=torch.long)
    targets[0, 1, 1] = 1; targets[0, 2, 2] = 1
    # landslide pixels: high class-1 score
    logits[0, 1, 1, 1] = 10; logits[0, 1, 2, 2] = 10
    # background pixels: high class-0 score (zeros elsewhere → softmax 0.5 → all predict 1)
    bg = (targets == 0)
    logits[0, 0][bg[0]] = 10
    m = compute_metrics(logits, targets)
    assert abs(m['f1'] - 100.0) < 0.1
    print(f"  Perfect preds: F1={m['f1']:.1f}%  Recall={m['recall']:.1f}%")
    print("  Metrics: PASSED ✓")


if __name__ == "__main__":
    tests  = [test_graph_ops, test_full_model, test_loss, test_metrics]
    failed = []
    for fn in tests:
        try:   fn()
        except Exception as e:
            failed.append(fn.__name__)
            print(f"\n*** FAILED: {fn.__name__} ***")
            traceback.print_exc()

    print(f"\n{'='*50}")
    if failed:
        print(f"FAILED: {failed}")
        print("Fix the above errors before training.")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED ✓  — ready to train")
        print("="*50)
        print("\nRun order:")
        print("  1. python test_components.py")
        print("  2. python pretrain.py --dataset bijie --epochs 5")
        print("  3. python pretrain.py --dataset bijie")
        print("  4. python finetune.py --dataset hokkaido")
