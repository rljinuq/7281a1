# -*- coding: utf-8 -*-
# 快速二分类：是否痴呆（NonDemented vs Demented）— 对称采样 + 阈值扫描版
import os, random, json, time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    f1_score, recall_score, precision_score, accuracy_score
)
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

# ======== 配置 ========
DATA_DIR   = "/Users/lingjieruan/Desktop/7281a1/Data"  # ← 改为你的路径
IMG_SIZE   = 128
BATCH_SIZE = 64
EPOCHS     = 3               # 先出一个可交差的 baseline
LR         = 2e-3
SEED       = 2025
CAP_PER_CLASS = 6000         # 每类上限（对称采样：Normal & Demented 各 CAP_PER_CLASS）
MODEL_OUT  = "fast_oasis_mbv3_binary.pth"
USE_FOCAL  = True            # 想先用交叉熵就设为 False

# 二分类：0=Normal, 1=Demented
BIN_NAMES = ["Normal","Demented"]

# 类别名映射（目录 -> 规范名）
CANONICAL = {
    "non demented":"NonDemented","nondemented":"NonDemented",
    "very mild dementia":"VeryMild","very mild demented":"VeryMild","very mild":"VeryMild",
    "mild dementia":"Mild","mild demented":"Mild",
    "moderate dementia":"Moderate","moderate demented":"Moderate",
}

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def canonicalize(name: str) -> str:
    key = name.strip().lower().replace("_"," ")
    key = " ".join(key.split())
    return CANONICAL.get(key, name.strip())

def collect_paths(root_dir: str):
    root = Path(root_dir)
    cls_to_files = defaultdict(list)
    for d in root.iterdir():
        if not d.is_dir(): continue
        cname = canonicalize(d.name)
        for p in d.rglob("*"):
            if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp"]:
                cls_to_files[cname].append(str(p))
    return cls_to_files

def build_binary_list(cls_to_files):
    # 打印原始四类数量
    print("=== 原始四类数量 ===")
    for k in ["NonDemented","VeryMild","Mild","Moderate"]:
        print(f"{k:12s}: {len(cls_to_files.get(k, []))}")

    nd  = cls_to_files.get("NonDemented", [])
    dem = cls_to_files.get("VeryMild", []) + cls_to_files.get("Mild", []) + cls_to_files.get("Moderate", [])

    # 对称采样：每类相同数量，最多 CAP_PER_CLASS
    cap = min(len(nd), len(dem), CAP_PER_CLASS)
    if len(nd)  > cap: nd  = random.sample(nd, cap)
    if len(dem) > cap: dem = random.sample(dem, cap)

    X = nd + dem
    y = [0]*len(nd) + [1]*len(dem)

    print("\n=== 二分类数量(对称采样) ===")
    print(f"Normal  : {len(nd)}")
    print(f"Demented: {len(dem)}")
    return X, y

class ImgListDS(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths; self.labels = labels; self.tfm = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tfm(img), self.labels[i]

def stratified_split(X, y, train=0.7, val=0.15, seed=SEED):
    X = np.array(X); y = np.array(y)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1-train, random_state=seed)
    tr_idx, vt_idx = next(sss1.split(X, y))
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_vt, y_vt = X[vt_idx], y[vt_idx]
    # 再划 val/test
    val_ratio = val / (1-train)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1-val_ratio, random_state=seed)
    v_idx, te_idx = next(sss2.split(X_vt, y_vt))
    return (X_tr.tolist(), y_tr.tolist(),
            X_vt[v_idx].tolist(), y_vt[v_idx].tolist(),
            X_vt[te_idx].tolist(), y_vt[te_idx].tolist())

def make_loaders(Xtr, ytr, Xv, yv, Xte, yte):
    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    test_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])

    tr_ds = ImgListDS(Xtr, ytr, train_tfms)
    v_ds  = ImgListDS(Xv,  yv,  test_tfms)
    te_ds = ImgListDS(Xte, yte, test_tfms)

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=pin_mem)
    val_loader   = DataLoader(v_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_mem)
    test_loader  = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_mem)

    return train_loader, val_loader, test_loader

def build_model(num_classes=2, freeze_backbone=True):
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_f = m.classifier[0].in_features if hasattr(m.classifier[0], 'in_features') else 576
    m.classifier = nn.Sequential(
        nn.Linear(in_f, 128),
        nn.Hardswish(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )
    if freeze_backbone:
        for p in m.features.parameters():
            p.requires_grad = False
    return m

class FocalLoss(nn.Module):
    """Focal Loss：降低易分类样本权重，alpha 对正类(痴呆)更关注"""
    def __init__(self, alpha=(1.0, 1.5), gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')
    def forward(self, logits, target):
        ce = self.ce(logits, target)  # [B]
        pt = torch.softmax(logits, dim=1)[torch.arange(logits.size(0)), target]
        alpha = self.alpha.to(logits.device)[target]
        loss = alpha * (1 - pt) ** self.gamma * ce
        return loss.mean() if self.reduction=='mean' else loss.sum()

def train_fast():
    set_seed()
    cls_to_files = collect_paths(DATA_DIR)
    X, y = build_binary_list(cls_to_files)
    Xtr, ytr, Xv, yv, Xte, yte = stratified_split(X, y, train=0.7, val=0.15)

    train_loader, val_loader, test_loader = make_loaders(Xtr, ytr, Xv, yv, Xte, yte)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model(num_classes=2, freeze_backbone=True).to(device)

    if USE_FOCAL:
        criterion = FocalLoss(alpha=(1.0, 1.5), gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()  # 对称采样下直接平权交叉熵也可

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    scaler    = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    best_val = 0.0
    patience = 2
    for ep in range(1, EPOCHS+1):
        # ------- train -------
        model.train()
        tr_loss = tr_correct = tr_total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{EPOCHS}", unit="batch")
        t0 = time.time()
        for x, yb in pbar:
            x, yb = x.to(device, non_blocking=True), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()

            tr_loss += loss.item()*x.size(0)
            tr_correct += (logits.argmax(1)==yb).sum().item()
            tr_total += x.size(0)
            pbar.set_postfix({
                "loss": f"{tr_loss/tr_total:.4f}",
                "acc":  f"{tr_correct/tr_total:.3f}",
                "b/s":  f"{pbar.n/max(time.time()-t0,1e-6):.1f}"
            })
        # ------- val -------
        model.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for x, yb in val_loader:
                x, yb = x.to(device, non_blocking=True), yb.to(device)
                logits = model(x)
                v_correct += (logits.argmax(1)==yb).sum().item()
                v_total += x.size(0)
        val_acc = v_correct / v_total
        scheduler.step(val_acc)
        print(f"[INFO] epoch {ep} done | val_acc={val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc; patience = 2
            torch.save({"state_dict": model.state_dict(),
                        "bin_names": BIN_NAMES,
                        "img_size": IMG_SIZE}, MODEL_OUT)
        else:
            patience -= 1
            if patience == 0:
                print(f"Early stop at epoch {ep}, best val_acc={best_val:.3f}")
                break

    # ------- test -------
    ckpt = torch.load(MODEL_OUT, map_location=device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    all_y, all_pred, all_prob = [], [], []
    with torch.no_grad():
        for x, yb in test_loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            pred = prob.argmax(1)
            all_prob.extend(prob.tolist())
            all_pred.extend(pred.tolist())
            all_y.extend(list(yb))

    print("\n=== 测试集（阈值=0.5）混淆矩阵 ===")
    print(confusion_matrix(all_y, all_pred))
    print("\n=== 测试集（阈值=0.5）分类报告 ===")
    print(classification_report(all_y, all_pred, target_names=BIN_NAMES, digits=4))

    y_true_1hot = np.eye(2)[np.array(all_y)]
    auc = roc_auc_score(y_true_1hot, np.array(all_prob), average="macro")
    print(f"Macro AUC: {auc:.4f}")

    # ------- 阈值扫描（优化 F1 或召回） -------
    probs = np.array(all_prob)[:,1]   # Demented 概率
    ys    = np.array(all_y)

    best_f1, best_thr, best_rec, best_prec, best_acc = -1, 0.5, 0, 0, 0
    for thr in np.linspace(0.30, 0.60, 31):
        pred = (probs >= thr).astype(int)
        f1  = f1_score(ys, pred)
        rec = recall_score(ys, pred)
        pre = precision_score(ys, pred)
        acc = accuracy_score(ys, pred)
        if f1 > best_f1:
            best_f1, best_thr, best_rec, best_prec, best_acc = f1, thr, rec, pre, acc

    print(f"\n阈值扫描 -> best_thr={best_thr:.2f} | F1={best_f1:.3f} | "
          f"Recall(Demented)={best_rec:.3f} | Precision={best_prec:.3f} | Acc={best_acc:.3f}")

    best_pred = (probs >= best_thr).astype(int)
    print("\n=== 测试集（最佳阈值）混淆矩阵 ===")
    print(confusion_matrix(ys, best_pred))
    print("\n=== 测试集（最佳阈值）分类报告 ===")
    print(classification_report(ys, best_pred, target_names=BIN_NAMES, digits=4))

    print(f"\n最优模型: {MODEL_OUT}")

if __name__ == "__main__":
    train_fast()


