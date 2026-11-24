# -*- coding: utf-8 -*-
"""Four-class MRI training with optional Grad-CAM helper functions."""

import os
import random
import re
import time
from pathlib import Path  # 路径处理 / path handling
from collections import defaultdict, Counter  # 计数与默认字典 / counters & dicts

import numpy as np
from PIL import Image, ImageOps, ImageEnhance  # 图像读写与增强 / image IO & basic aug

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler  # 数据加载与加权采样 / data loading & weighted sampling
from torchvision import transforms, models  # 视觉模型与预处理 / vision models & transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score  # 评估指标 / metrics
from sklearn.model_selection import StratifiedShuffleSplit  # 分层划分 / stratified split
from tqdm import tqdm  # 进度条 / progress bar

# ===== 配置 / configuration =====
DATA_DIR   = "/Users/lingjieruan/Desktop/7281a1/Data"   # 数据目录 / dataset root
IMG_SIZE   = 160                                        # 输入尺寸 / input size
BATCH_SIZE = 64                                         # 批大小 / batch size
EPOCHS     = 8                                          # 训练轮数 / epochs
LR         = 1e-3                                       # 学习率 / learning rate
SEED       = 2025                                       # 随机种子 / random seed
CAP_NONDEM = 6000                                       # NonDemented 上限 / cap
CAP_VERY   = 6000                                       # VeryMild 上限 / cap
CAP_MILD   = 5000                                       # Mild 上限 / cap
CAP_MOD    = 488                                        # Moderate 上限 / cap
MODEL_OUT  = "oasis_fourclass_balanced.pth"             # 模型保存名 / output path
UNFREEZE_LAST_BLOCKS = True                             # 解冻尾部特征块 / unfreeze tail blocks
PRINT_EVAL_CM = True                                    # 打印混淆矩阵 / print confusion matrix

# 类别映射 / class mappings
CLASSES = ["NonDemented","VeryMild","Mild","Moderate"]
CLASS2IDX = {c:i for i,c in enumerate(CLASSES)}  # 类名→索引 / name→index
IDX2CLASS = {i:c for c,i in CLASS2IDX.items()}   # 索引→类名 / index→name

# 目录名规范化 / canonical folder-to-class mapping
CANONICAL = {
    "non demented":"NonDemented","nondemented":"NonDemented",
    "very mild dementia":"VeryMild","very mild demented":"VeryMild","very mild":"VeryMild",
    "mild dementia":"Mild","mild demented":"Mild",
    "moderate dementia":"Moderate","moderate demented":"Moderate",
}

def set_seed(s=SEED):
    """固定随机性 / set RNG for reproducibility"""
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def canonicalize(name: str) -> str:
    """类名清洗：小写、去下划线、合并空格并映射 / normalize and map to canonical class name"""
    key = name.strip().lower().replace("_"," ")
    key = " ".join(key.split())
    return CANONICAL.get(key, name.strip())

def collect_paths(root_dir: str):
    """按子文件夹收集图像路径 / collect image paths grouped by class subfolders"""
    root = Path(root_dir)
    cls_to_files = defaultdict(list)
    for d in root.iterdir():
        if not d.is_dir(): continue  # 仅保留目录 / dirs only
        cname = canonicalize(d.name)  # 目录名→标准类名 / folder→class
        for p in d.rglob("*"):  # 递归遍历 / recursive walk
            if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp"]:
                cls_to_files[cname].append(str(p))
    return cls_to_files

def cap_per_class(cls_to_files):
    """
    按预设上限裁剪各类样本（保留小类全量）/ cap large classes for speed (keep minority intact)
    返回：样本路径列表 X 与标签列表 y / return X paths and y labels
    """
    caps = {"NonDemented":CAP_NONDEM, "VeryMild":CAP_VERY, "Mild":CAP_MILD, "Moderate":CAP_MOD}
    X, y = [], []
    print("=== 原始数量 ===")
    for c in CLASSES:
        print(f"{c:12s}: {len(cls_to_files.get(c, []))}")
    print("\n=== 采样（上限） ===")
    for c in CLASSES:
        files = cls_to_files.get(c, [])
        cap = caps[c]
        if len(files) > cap:
            files = random.sample(files, cap)  # 随机下采样 / random downsample
        print(f"{c:12s}: {len(files)}")
        X.extend(files)
        y.extend([CLASS2IDX[c]]*len(files))
    return X,y

# 针对小类的更强增强 / stronger aug for minority classes
class SmallClassAugment:
    def __init__(self, cls_idx):
        self.cls_idx = cls_idx  # 目标类索引（未直接使用）/ target class index (not directly used)
        self.enhance = transforms.Compose([
            transforms.RandomRotation(15),                 # 轻旋转 / mild rotation
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),  # 轻裁剪缩放 / mild resized crop
        ])
    def __call__(self, img):
        # 自适应对比度 + 亮度/对比度抖动 / autocontrast + light brightness/contrast jitter
        img = ImageOps.autocontrast(img)
        img = ImageEnhance.Brightness(img).enhance(0.9 + 0.2*random.random())
        img = ImageEnhance.Contrast(img).enhance(0.9 + 0.2*random.random())
        return self.enhance(img)

class ImgListDS(Dataset):
    """基于路径列表的数据集定义 / dataset backed by path list"""
    def __init__(self, paths, labels, is_train=True):
        self.paths = paths; self.labels = labels; self.is_train = is_train
        self.base_train = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),   # 统一尺寸 / resize
            transforms.RandomHorizontalFlip(),         # 随机翻转 / random flip
            transforms.ToTensor(),                     # 转张量 / to tensor
            transforms.Normalize([0.5]*3,[0.5]*3),     # 归一化 / normalize to [-1,1]
        ])
        self.base_test = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3),
        ])
        self.aug_small = SmallClassAugment(cls_idx=CLASS2IDX["Moderate"])  # 小类额外增强器 / extra aug for minority

    def __len__(self): return len(self.paths)  # 数据量 / dataset size

    def __getitem__(self, i):
        """读取图像并应用训练/验证流程 / load image and apply train/eval pipeline"""
        p = self.paths[i]; y = self.labels[i]
        img = Image.open(p).convert("RGB")
        if self.is_train:
            # 对 Mild/Moderate 以 0.7 概率追加更强增强 / stronger aug with p=0.7 for Mild/Moderate
            if y in [CLASS2IDX["Mild"], CLASS2IDX["Moderate"]] and random.random() < 0.7:
                img = self.aug_small(img)
            return self.base_train(img), y
        else:
            return self.base_test(img), y

def stratified_split(X, y, train=0.7, val=0.15):
    """
    Stratify OASIS subjects, then expand their images into disjoint splits.
    Ratios apply to subjects, not slices. Insufficient subject counts stop the split.
    """
    X = np.array(X)
    y = np.array(y)
    subject_indices = defaultdict(list)
    for index, path in enumerate(X):
        match = re.match(r"^(OAS\d+_\d+)_MR\d+_", Path(path).stem)
        if match is None:
            raise ValueError(f"Cannot identify an OASIS subject from filename: {path}")
        # Keep repeat scans (MR1, MR2, ...) and all slices with the same subject.
        subject_indices[match.group(1)].append(index)

    subjects = np.array(sorted(subject_indices))
    subject_labels = []
    for subject in subjects:
        labels = set(y[subject_indices[subject]].tolist())
        if len(labels) != 1:
            raise ValueError(f"Subject {subject} has conflicting class labels.")
        subject_labels.append(next(iter(labels)))
    subject_labels = np.array(subject_labels)
    subject_counts = Counter(subject_labels.tolist())
    if any(subject_counts[label] < 3 for label in range(len(CLASSES))):
        raise ValueError("Each class needs at least three subjects for disjoint train/val/test coverage.")

    # Split subject IDs only; never fall back to an image-level split.
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1-train, random_state=SEED)
    val_ratio = val / (1-train)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1-val_ratio, random_state=SEED)
    try:
        tr_subjects, vt_subjects = next(sss1.split(subjects, subject_labels))
        v_subjects, te_subjects = next(sss2.split(subjects[vt_subjects], subject_labels[vt_subjects]))
    except ValueError as exc:
        raise ValueError("Too few subjects for the requested stratified split; no image-level fallback is used.") from exc

    for label in set(subject_labels.tolist()):
        if not all(label in subject_labels[indices] for indices in
                   (tr_subjects, vt_subjects[v_subjects], vt_subjects[te_subjects])):
            raise ValueError("The requested subject split does not cover every class in all three sets.")

    def expand(indices):
        image_indices = [i for j in indices for i in subject_indices[subjects[j]]]
        return X[image_indices].tolist(), y[image_indices].tolist()

    X_tr, y_tr = expand(tr_subjects)
    X_val, y_val = expand(vt_subjects[v_subjects])
    X_test, y_test = expand(vt_subjects[te_subjects])
    return X_tr, y_tr, X_val, y_val, X_test, y_test

def make_loaders(Xtr, ytr, Xv, yv, Xte, yte):
    """
    构建 DataLoader；训练集使用加权随机采样以平衡各类频次
    build dataloaders; train loader uses WeightedRandomSampler for class balance
    """
    tr_ds = ImgListDS(Xtr, ytr, is_train=True)
    v_ds  = ImgListDS(Xv,  yv,  is_train=False)
    te_ds = ImgListDS(Xte, yte, is_train=False)

    # 统计训练集中每类数量 / count per class
    cnt = Counter(ytr)
    total = sum(cnt.values())
    # 频率倒数作为类权重：少数类权重大 / inverse-frequency class weights (rarer→heavier)
    cls_w = {c: total/cnt[c] for c in range(len(CLASSES))}
    # 为每个样本赋采样权重 / per-sample weights
    sample_w = [cls_w[y] for y in ytr]
    # 放回采样使各类出现更均衡 / with-replacement sampling for balance
    sampler = WeightedRandomSampler(sample_w, num_samples=len(ytr), replacement=True)

    pin_mem = torch.cuda.is_available()  # GPU 时开启 pin_memory / enable pin_memory on GPU
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, sampler=sampler, shuffle=False, num_workers=0, pin_memory=pin_mem)
    v_loader  = DataLoader(v_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_mem)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_mem)
    return tr_loader, v_loader, te_loader, cnt

def build_model(num_classes=4):
    """
    构建 MobileNetV3-Small 并替换分类头；可选解冻尾部特征块
    build MobileNetV3-Small, replace classifier head; optionally unfreeze tail blocks
    """
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    # 兼容不同版本的 in_features 取法 / handle version variance
    in_f = m.classifier[0].in_features if hasattr(m.classifier[0],'in_features') else 576
    m.classifier = nn.Sequential(
        nn.Linear(in_f, 256),   # 降维至 256 / reduce to 256
        nn.Hardswish(),         # 非线性 / nonlinearity
        nn.Dropout(0.25),       # 正则化 / regularization
        nn.Linear(256, num_classes)  # 输出类别数 / logits
    )
    if UNFREEZE_LAST_BLOCKS:
        # 先全部冻结，再解冻末尾若干参数组 / freeze all then unfreeze last groups
        for _, p in m.features.named_parameters():
            p.requires_grad = False
        for name, p in list(m.features.named_parameters())[-6:]:  # 最后 6 个参数张量 / last 6 parameter tensors
            p.requires_grad = True
    else:
        for p in m.features.parameters():
            p.requires_grad = False
    return m

def effective_num_class_weights(counts, beta=0.999):  # CB-Loss 权重 / class-balanced weights
    """
    基于有效样本数计算类权重，均值约为1便于稳定训练
    compute class weights via effective number; normalized to mean≈1
    counts: 各类样本数 / per-class counts
    """
    eff_num = [(1 - beta**c)/(1 - beta) for c in counts]  # 有效样本数 / effective number
    weights = [1.0/e for e in eff_num]                    # 反比权重 / inverse to eff_num
    s = sum(weights); weights = [w * len(counts)/s for w in weights]  # 归一化 / normalize
    return torch.tensor(weights, dtype=torch.float32)

def evaluate(model, loader, device):
    """
    前向推理并计算宏F1，返回 (macro_f1, 全部真值, 全部预测)
    forward inference and compute macro-F1; return (macro_f1, all_y, all_pred)
    """
    model.eval()
    all_y, all_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = logits.argmax(1).cpu().numpy().tolist()
            all_pred.extend(pred); all_y.extend(yb.numpy().tolist())
    macro_f1 = f1_score(all_y, all_pred, average='macro')
    return macro_f1, all_y, all_pred

def train():
    set_seed()  # 固定随机性 / set RNG
    cls_to_files = collect_paths(DATA_DIR)  # 扫描数据 / scan dataset
    X, y = cap_per_class(cls_to_files)  # 按上限裁剪 / cap large classes
    Xtr,ytr,Xv,yv,Xte,yte = stratified_split(X,y)  # 受试者分层划分 / subject-level stratified split

    tr_loader, v_loader, te_loader, train_counts = make_loaders(Xtr,ytr,Xv,yv,Xte,yte)  # 构建加载器 / build loaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备选择 / device
    model = build_model().to(device)  # 构建模型并上卡 / build & to device

    # —— 类平衡权重（CB-Loss 思想） —— #
    cls_counts_list = [train_counts[i] if i in train_counts else 1 for i in range(len(CLASSES))]  # 训练集计数 / counts
    cb_weights = effective_num_class_weights(cls_counts_list, beta=0.999).to(device)  # 计算权重 / compute weights
    criterion = nn.CrossEntropyLoss(weight=cb_weights)  # 加权交叉熵 / weighted CE

    optimizer = optim.AdamW(model.parameters(), lr=LR)  # 优化器 / optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)  # 自适应降LR / LR scheduler

    best_macro_f1, best_state = -1, None  # 最优指标缓存 / best metric cache
    patience, max_patience = 3, 3  # 早停耐心 / early stop patience

    for ep in range(1, EPOCHS+1):  # 训练轮次 / epochs
        model.train()
        tr_loss = tr_correct = tr_total = 0  # 统计项 / accumulators
        pbar = tqdm(tr_loader, desc=f"Epoch {ep}/{EPOCHS}", unit="batch")  # 进度条 / progress bar
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)  # 数据上卡 / to device
            optimizer.zero_grad(set_to_none=True)  # 清梯度 / zero grad
            logits = model(xb)  # 前向 / forward
            loss = criterion(logits, yb)  # 损失 / loss
            loss.backward(); optimizer.step()  # 反传与更新 / backward & step

            tr_loss += loss.item()*xb.size(0)  # 累计损失 / sum loss
            tr_correct += (logits.argmax(1)==yb).sum().item()  # 累计正确 / sum correct
            tr_total += yb.size(0)  # 累计样本 / sum samples

            pbar.set_postfix({
                "loss": f"{tr_loss/tr_total:.4f}",  # 当前均值损失 / running loss
                "acc":  f"{tr_correct/tr_total:.3f}"  # 当前准确率 / running acc
            })

        # 验证：用 Macro-F1 作早停指标 / validation: macro-F1 for early stop
        macro_f1, vy, vpred = evaluate(model, v_loader, device)
        scheduler.step(macro_f1)  # 根据指标调整LR / step scheduler by metric
        print(f"[INFO] epoch {ep} | val_macroF1={macro_f1:.3f}")

        if macro_f1 > best_macro_f1:  # 刷新最优 / update best
            best_macro_f1 = macro_f1
            best_state = {"state_dict": model.state_dict()}  # 仅权重 / weights only
            torch.save(best_state, MODEL_OUT)  # 保存检查点 / save checkpoint
            patience = max_patience  # 重置耐心 / reset patience
        else:
            patience -= 1  # 耐心-1 / decrement patience
            if patience == 0:  # 触发早停 / trigger early stop
                print(f"Early stop at epoch {ep}, best val_macroF1={best_macro_f1:.3f}")
                break

    # 测试 / testing
    ckpt = torch.load(MODEL_OUT, map_location=device)  # 载入最优 / load best
    model.load_state_dict(ckpt["state_dict"]); model.eval()  # 加载并评估 / load & eval

    all_y, all_pred = [], []  # 聚合结果 / collectors
    with torch.no_grad():
        for xb, yb in te_loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = logits.argmax(1).cpu().numpy().tolist()
            all_pred.extend(pred); all_y.extend(yb.numpy().tolist())

    if PRINT_EVAL_CM:  # 打印测试集报告 / print test report
        print("\n=== 四分类混淆矩阵（测试集） ===")
        print(confusion_matrix(all_y, all_pred))
        print("\n=== 四分类报告（测试集） ===")
        print(classification_report(all_y, all_pred, target_names=CLASSES, digits=4))

# ========== Optional Grad-CAM helpers (not called by the training entry point) ==========
import cv2  # 图像处理 / image ops
import matplotlib.pyplot as plt  # 可视化 / visualization

def _find_last_conv(module: nn.Module):
    """找到最后一个卷积层以用于Grad-CAM / find last Conv2d for Grad-CAM"""
    last_conv = None
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv

def _tensor_transform_for_infer():
    """推理时的图像预处理（与测试一致）/ eval-time transforms"""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

def _np_uint8(img_pil: Image.Image, size=(IMG_SIZE, IMG_SIZE)):
    """PIL转uint8数组并调整大小 / PIL->uint8 numpy with resize"""
    return np.array(img_pil.resize(size))

def gradcam_on_image(model: nn.Module,
                     img_path: str,
                     class_names=CLASSES,
                     target_class_idx: int = None,
                     alpha: float = 0.45,
                     out_path: str = None):
    """
    生成并保存单图Grad-CAM叠加图 / generate & save Grad-CAM overlay for one image
    返回(out_path, pred_idx) / return(out_path, pred_idx)
    """
    device = next(model.parameters()).device
    model.eval()

    target_layer = _find_last_conv(model)  # 目标卷积层 / target conv
    if target_layer is None:
        raise RuntimeError("未找到卷积层，无法计算 Grad-CAM。")  # 无卷积时报错 / no conv found

    conv_features, conv_grads = [], []  # 存特征与梯度 / holders

    def fwd_hook(module, inp, out):
        conv_features.append(out.detach())  # 前向输出 / forward features

    # 注册反向hook（兼容旧/新API）/ register backward hook (new/legacy)
    try:
        bwd_handle = target_layer.register_full_backward_hook(
            lambda module, grad_in, grad_out: conv_grads.append(grad_out[0].detach())
        )
    except Exception:
        bwd_handle = target_layer.register_backward_hook(
            lambda module, grad_in, grad_out: conv_grads.append(grad_out[0].detach())
        )
    fwd_handle = target_layer.register_forward_hook(fwd_hook)

    # 读图与预处理 / load & preprocess
    pil = Image.open(img_path).convert("RGB")
    ttf = _tensor_transform_for_infer()
    x = ttf(pil).unsqueeze(0).to(device)

    # 前向与目标类别 / forward & target class
    logits = model(x)
    pred_idx = int(logits.argmax(1).item()) if target_class_idx is None else int(target_class_idx)

    # 对目标类别做反向 / backprop on target logit
    model.zero_grad(set_to_none=True)
    logits[0, pred_idx].backward()

    # 取特征与梯度并计算CAM / gather features & grads, compute CAM
    feats = conv_features[-1].cpu().numpy()[0]   # [C,H,W]
    grads = conv_grads[-1].cpu().numpy()[0]      # [1,C,H,W] -> [C,H,W]
    weights = grads.mean(axis=(1, 2))            # 通道权重 / channel weights (GAP)
    cam = np.zeros(feats.shape[1:], dtype=np.float32)
    for c, w in enumerate(weights):
        cam += w * feats[c]

    cam = np.maximum(cam, 0)  # ReLU / relu
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)  # 归一化 / normalize
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))  # 调整到输入尺寸 / resize to input size

    # 叠加热力图 / overlay heatmap
    base = _np_uint8(pil, (IMG_SIZE, IMG_SIZE))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[:, :, ::-1]  # 伪彩色(BGR->RGB) / colormap
    overlay = (alpha * heat + (1 - alpha) * base).astype(np.uint8)  # 融合 / blend

    # 输出路径 / build output path
    if out_path is None:
        base_name, _ = os.path.splitext(img_path)
        out_path = base_name + f"_gradcam_{class_names[pred_idx]}.png"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)  # 创建目录 / ensure dir
    Image.fromarray(overlay).save(out_path)  # 保存 / save

    # 清理hook / remove hooks
    fwd_handle.remove()
    bwd_handle.remove()

    return out_path, pred_idx  # 返回文件与预测类 / return path & prediction

def load_trained_model_for_cam(ckpt_path=MODEL_OUT):
    """载入已训练权重用于Grad-CAM / load trained weights for Grad-CAM"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(CLASSES)).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)  # 兼容纯state_dict或打包字典 / handle both formats
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, device

def _gather_images_by_class(root_dir=DATA_DIR, class_names=CLASSES):
    """按类别收集图片路径（递归）/ gather image paths per class (recursive)"""
    root = Path(root_dir)
    mapping = {c: [] for c in class_names}
    for d in root.iterdir():
        if not d.is_dir():
            continue
        cname_std = canonicalize(d.name)  # 目录名标准化 / canonicalize
        if cname_std in mapping:
            files = [str(p) for p in d.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
            mapping[cname_std].extend(files)
    return mapping

def run_gradcam_batch(save_dir="outputs/gradcam", per_class=3, alpha=0.45, seed=SEED):
    """
    批量生成Grad-CAM图像 / batch generate Grad-CAM images
    每类随机选 per_class 张并保存 / per class sample per_class images
    """
    random.seed(seed); np.random.seed(seed)  # 固定随机性 / set RNG

    model, device = load_trained_model_for_cam(MODEL_OUT)  # 加载模型 / load model
    img_dict = _gather_images_by_class(DATA_DIR, CLASSES)  # 收集路径 / gather paths

    total = 0  # 计数 / counter
    for cname in CLASSES:
        imgs = img_dict.get(cname, [])
        if not imgs:
            print(f"[Grad-CAM] 警告：类别 {cname} 未找到图片。")  # 无图警告 / warn no images
            continue
        random.shuffle(imgs)  # 打乱 / shuffle
        sel = imgs[:min(per_class, len(imgs))]  # 取样 / select

        out_dir = os.path.join(save_dir, cname)
        os.makedirs(out_dir, exist_ok=True)  # 确保目录 / ensure dir

        print(f"[Grad-CAM] {cname}: 准备处理 {len(sel)} 张...")  # 进度 / progress
        for p in sel:
            fname = Path(p).stem + ".png"
            out_path = os.path.join(out_dir, fname)
            try:
                out_file, pred_idx = gradcam_on_image(
                    model, p,
                    class_names=CLASSES,
                    target_class_idx=None,  # 解释预测top1 / explain top-1
                    alpha=alpha,
                    out_path=out_path
                )
                total += 1
                print(f"  ✓ {Path(p).name} -> {Path(out_file).name} (pred={CLASSES[pred_idx]})")
            except Exception as e:
                print(f"  × {Path(p).name} 失败：{e}")  # 失败提示 / failure log

    print(f"[Grad-CAM] 完成，合计生成 {total} 张。保存目录: {save_dir}")  # 汇总 / summary

def make_gradcam_grid(save_root="outputs/gradcam", grid_path="outputs/gradcam_grid.png", per_class=3):
    """
    将每类若干张Grad-CAM拼接成网格图（行=类别，列=样本）/ make a grid figure (rows=classes, cols=samples)
    """
    from math import ceil  # 备用 / spare
    fig_w = per_class * 3  # 宽度（英寸）/ figure width (inches)
    fig_h = len(CLASSES) * 3  # 高度（英寸）/ figure height (inches)
    plt.figure(figsize=(fig_w, fig_h))
    idx = 1  # 子图索引 / subplot index
    for r, cname in enumerate(CLASSES):
        cdir = Path(save_root) / cname
        if not cdir.exists():
            continue
        files = sorted([p for p in cdir.glob("*.png")])[:per_class]  # 取前N张 / take first N
        for c in range(per_class):
            plt.subplot(len(CLASSES), per_class, idx); idx += 1
            if c < len(files):
                img = Image.open(files[c]).convert("RGB")
                plt.imshow(img)
                if c == 0:
                    plt.ylabel(cname, fontsize=10)  # 左侧标注类名 / class label on left
            plt.axis("off")
    os.makedirs(Path(grid_path).parent, exist_ok=True)  # 确保输出目录 / ensure output dir
    plt.tight_layout(); plt.savefig(grid_path, dpi=300)  # 保存高分图 / save hi-res
    print(f"[Grad-CAM] 拼图已保存: {grid_path}")  # 提示 / notify

if __name__ == "__main__":
    train()  # 脚本入口 / script entry


