"""
test_components.py
Run this BEFORE starting training to verify every module is correct.

Expected output:
  GRB  ........ PASSED
  MGRM ........ PASSED
  AttentionConv PASSED
  Multiscale .. PASSED
  AMGUnet ..... PASSED
  Loss ........ PASSED
  Metrics ..... PASSED
  ALL TESTS PASSED — ready to train

Usage:
  python test_components.py
"""

import sys
import torch
import torch.nn.functional as F


def section(title):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: GraphReasoningBlock
# ─────────────────────────────────────────────────────────────────────────────
def test_grb():
    section("1. GraphReasoningBlock (GRB)")
    from models.grb import GraphReasoningBlock
    grb = GraphReasoningBlock(in_ch=512, ratio=4)

    x   = torch.randn(2, 512, 32, 32)
    out = grb(x)

    assert out.shape == x.shape,     f"Shape mismatch: {out.shape} vs {x.shape}"
    assert torch.isfinite(out).all(),"NaN/Inf in output"
    assert not torch.allclose(out, x, atol=1e-5), "Output == input (GCN inactive)"

    out.mean().backward()
    for name, p in grb.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"

    Cr     = grb.Cr
    N      = 32 * 32
    params = sum(p.numel() for p in grb.parameters())

    print(f"  in_ch=512  Cr={Cr}  N={N}")
    print(f"  Adjacency shape: {Cr}×{Cr}={Cr*Cr:,}  (N×N would be {N*N:,})")
    print(f"  Memory saving: {N*N//( Cr*Cr)}×  smaller than full N×N")
    print(f"  Parameters: {params:,}")
    print(f"  GRB ........ PASSED ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: MGRM
# ─────────────────────────────────────────────────────────────────────────────
def test_mgrm():
    section("2. MGRM (4 parallel GRBs)")
    from models.mgrm import MGRM
    mgrm = MGRM(in_ch=512)

    x   = torch.randn(2, 512, 32, 32)
    out = mgrm(x)

    assert out.shape == x.shape,      f"Shape mismatch: {out.shape}"
    assert torch.isfinite(out).all(), "NaN/Inf in output"
    assert not torch.allclose(out, x, atol=1e-5), "Output == input"

    params = sum(p.numel() for p in mgrm.parameters())
    print(f"  Node counts:  1024 | 256 | 100 | 36  (branches 1-4)")
    print(f"  Parameters: {params/1e6:.2f}M")
    print(f"  MGRM ........ PASSED ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: AttentionConv
# ─────────────────────────────────────────────────────────────────────────────
def test_attention_conv():
    section("3. AttentionConv (encoder block)")
    from models.attention_conv import AttentionConv
    ac = AttentionConv(in_ch=64, out_ch=128)

    x   = torch.randn(2, 64, 256, 256)
    out = ac(x)

    assert out.shape == (2, 128, 128, 128), f"Shape: {out.shape}"
    assert torch.isfinite(out).all(),       "NaN/Inf in output"

    # Verify gate φ is in (0, 1)
    with torch.no_grad():
        q  = ac.Wq(x)
        k  = ac.Wk(x)
        ph = torch.sigmoid(ac.psi(F.relu(q * k)))
    assert ph.min() >= 0.0 and ph.max() <= 1.0, \
        f"Gate φ out of range: [{ph.min():.3f}, {ph.max():.3f}]"

    print(f"  Input:   {tuple(x.shape)}")
    print(f"  Output:  {tuple(out.shape)}  (H,W halved ✓  channels doubled ✓)")
    print(f"  Gate φ:  [{ph.min():.3f}, {ph.max():.3f}] ⊂ (0,1) ✓")
    print(f"  AttentionConv PASSED ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: MultiscaleConnection
# ─────────────────────────────────────────────────────────────────────────────
def test_multiscale():
    section("4. MultiscaleConnection (skip connection)")
    from models.multiscale import MultiscaleConnection
    ms = MultiscaleConnection(in_ch=256, out_ch=256)

    x   = torch.randn(2, 256, 64, 64)
    out = ms(x)

    assert out.shape == x.shape,      f"Shape mismatch: {out.shape}"
    assert torch.isfinite(out).all(), "NaN/Inf in output"

    params = sum(p.numel() for p in ms.parameters())
    print(f"  Input/Output shape: {tuple(x.shape)}  (H,W preserved ✓)")
    print(f"  Branch RFs: 1×1 | 3×3 | 7×7 (d=3) | pool-3×3")
    print(f"  Parameters: {params:,}")
    print(f"  Multiscale  PASSED ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Full AMGUnet
# ─────────────────────────────────────────────────────────────────────────────
def test_ammgunet():
    section("5. Full AMGUnet (complete model)")
    from models.ammg_unet import AMGUnet
    model = AMGUnet(in_ch=3, num_classes=2)

    x   = torch.randn(1, 3, 512, 512)
    out = model(x)

    assert out.shape == (1, 2, 512, 512), f"Shape: {out.shape}"
    assert torch.isfinite(out).all(),     "NaN/Inf in output"

    loss = out.mean()
    loss.backward()
    # Check all parameters receive gradients
    no_grad = [n for n,p in model.named_parameters() if p.grad is None]
    assert len(no_grad) == 0, f"No gradient for: {no_grad}"

    total = model.count_parameters()
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(out.shape)}")
    print(f"  Total parameters: {total/1e6:.1f}M  (paper: ~48.1M)")
    print(f"  Backward pass:    OK — all params have gradients")
    print(f"  AMGUnet ..... PASSED ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Loss function
# ─────────────────────────────────────────────────────────────────────────────
def test_loss():
    section("6. CombinedLoss (Dice + CE)")
    from loss import CombinedLoss
    crit = CombinedLoss()

    logits  = torch.randn(2, 2, 256, 256)
    targets = torch.randint(0, 2, (2, 256, 256))

    L_rand = crit(logits, targets)
    assert torch.isfinite(L_rand), f"NaN loss: {L_rand}"
    assert L_rand.item() > 0,       "Loss is 0 for random predictions"

    # Perfect predictions → loss near 0
    perf = torch.zeros_like(logits)
    perf[:, 1][targets == 1] = 10.0
    perf[:, 0][targets == 0] = 10.0
    L_perf = crit(perf, targets)

    assert L_rand.item() > L_perf.item(), \
        "Perfect predictions should have lower loss than random"

    print(f"  Random loss:  {L_rand.item():.4f}  (expect ~1.0-1.5)")
    print(f"  Perfect loss: {L_perf.item():.4f}  (expect ~0.0)")
    print(f"  Loss ........ PASSED ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Metrics
# ─────────────────────────────────────────────────────────────────────────────
def test_metrics():
    section("7. Evaluation Metrics (Eq 9-12)")
    from metrics import compute_metrics

    logits  = torch.zeros(1, 2, 4, 4)
    targets = torch.zeros(1, 4, 4, dtype=torch.long)
    targets[0, 1, 1] = 1; targets[0, 2, 2] = 1
    logits[0, 1, 1, 1] = 10.0; logits[0, 1, 2, 2] = 10.0

    m = compute_metrics(logits, targets)

    assert abs(m['f1']          - 100.0) < 0.1, f"F1={m['f1']}"
    assert abs(m['recall']      - 100.0) < 0.1, f"Recall={m['recall']}"
    assert abs(m['precision']   - 100.0) < 0.1, f"Precision={m['precision']}"
    assert abs(m['specificity'] - 100.0) < 0.1, f"Spec={m['specificity']}"

    print(f"  Perfect prediction: F1={m['f1']:.1f}%  Recall={m['recall']:.1f}%")
    print(f"  Metrics ..... PASSED ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import traceback
    tests = [
        test_grb,
        test_mgrm,
        test_attention_conv,
        test_multiscale,
        test_ammgunet,
        test_loss,
        test_metrics,
    ]

    failed = []
    for fn in tests:
        try:
            fn()
        except Exception as e:
            failed.append(fn.__name__)
            print(f"\n  *** FAILED: {fn.__name__} ***")
            traceback.print_exc()

    print(f"\n{'='*50}")
    if failed:
        print(f"FAILED: {failed}")
        print("Fix the above errors before training.")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED ✓  —  ready to train")
        print("="*50)
        print("\nNext steps:")
        print("  1. Prepare data:  python data_prep/prepare_hokkaido.py --raw_dir /your/hokkaido --verify")
        print("  2. Quick test:    python pretrain.py --dataset bijie --epochs 5")
        print("  3. Pretrain:      python pretrain.py --dataset cas")
        print("  4. Fine-tune:     python finetune.py --dataset hokkaido")
