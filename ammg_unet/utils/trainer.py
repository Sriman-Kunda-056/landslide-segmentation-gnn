"""
utils/trainer.py
Shared training and validation functions used by pretrain.py and finetune.py.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from metrics import accumulate_metrics, print_metrics
from config  import GRAD_CLIP


def train_one_epoch(model, loader, optimizer, criterion, device):
    """One full pass through training data. Returns average loss."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="  train", leave=False, ncols=80)
    for imgs, masks in pbar:
        imgs  = imgs.to(device,  non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)
        loss   = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """One full pass through validation data. Returns loss + metrics dict."""
    model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []

    for imgs, masks in tqdm(loader, desc="  val  ", leave=False, ncols=80):
        imgs  = imgs.to(device,  non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)
        total_loss += criterion(logits, masks).item()
        all_logits.append(logits.cpu())
        all_targets.append(masks.cpu())

    metrics = accumulate_metrics(all_logits, all_targets)
    return total_loss / len(loader), metrics


def save_checkpoint(state: dict, path: str) -> None:
    torch.save(state, path)


def load_checkpoint(path: str, device: torch.device) -> dict:
    return torch.load(path, map_location=device)
