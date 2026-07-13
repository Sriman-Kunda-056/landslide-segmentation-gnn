import torch, torch.nn as nn
from tqdm import tqdm
from metrics import accumulate_metrics
from config  import GRAD_CLIP

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train(); total = 0.0
    for imgs, masks in tqdm(loader, desc="  train", leave=False, ncols=70):
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss   = criterion(logits, masks)
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step(); total += loss.item()
    return total / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval(); total = 0.0; al, at = [], []
    for imgs, masks in tqdm(loader, desc="  val  ", leave=False, ncols=70):
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs); total += criterion(logits, masks).item()
        al.append(logits.cpu()); at.append(masks.cpu())
    return total/len(loader), accumulate_metrics(al, at)
