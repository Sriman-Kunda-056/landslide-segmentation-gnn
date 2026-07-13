"""
metrics.py  —  Equations 9-12 from paper.
"""
import torch

def compute_metrics(logits, targets, threshold=0.5):
    preds = torch.argmax(logits, dim=1).long()
    pf    = preds.view(-1); tf = targets.view(-1)
    TP = ((pf==1)&(tf==1)).sum().float()
    FP = ((pf==1)&(tf==0)).sum().float()
    FN = ((pf==0)&(tf==1)).sum().float()
    TN = ((pf==0)&(tf==0)).sum().float()
    eps = 1e-7
    rec  = TP/(TP+FN+eps); spe = TN/(TN+FP+eps)
    pre  = TP/(TP+FP+eps); f1  = 2*pre*rec/(pre+rec+eps)
    return {'recall':rec.item()*100,'specificity':spe.item()*100,
            'precision':pre.item()*100,'f1':f1.item()*100,
            'TP':int(TP),'FP':int(FP),'FN':int(FN),'TN':int(TN)}

def accumulate_metrics(all_logits, all_targets):
    return compute_metrics(torch.cat(all_logits), torch.cat(all_targets))

def print_metrics(m, prefix=""):
    print(f"{prefix}Recall:{m['recall']:.2f}%  Spec:{m['specificity']:.2f}%  "
          f"Prec:{m['precision']:.2f}%  F1:{m['f1']:.2f}%")
