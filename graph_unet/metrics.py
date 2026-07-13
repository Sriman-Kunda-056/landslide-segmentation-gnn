"""
metrics.py — Recall, Specificity, Precision, F1  (Paper Eq 9-12)
"""
import torch

def compute_metrics(logits, targets, threshold=0.5):
    p  = (torch.softmax(logits,1)[:,1] >= threshold).long().view(-1)
    t  = targets.view(-1)
    TP = ((p==1)&(t==1)).sum().float()
    FP = ((p==1)&(t==0)).sum().float()
    FN = ((p==0)&(t==1)).sum().float()
    TN = ((p==0)&(t==0)).sum().float()
    e  = 1e-7
    rec = TP/(TP+FN+e); spe = TN/(TN+FP+e)
    pre = TP/(TP+FP+e); f1  = 2*pre*rec/(pre+rec+e)
    return {'recall':rec.item()*100,'specificity':spe.item()*100,
            'precision':pre.item()*100,'f1':f1.item()*100}

def accumulate_metrics(all_l, all_t):
    return compute_metrics(torch.cat(all_l), torch.cat(all_t))

def print_metrics(m, prefix=""):
    print(f"{prefix}Recall:{m['recall']:.2f}%  Spec:{m['specificity']:.2f}%  "
          f"Prec:{m['precision']:.2f}%  F1:{m['f1']:.2f}%")
