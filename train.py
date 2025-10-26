# -*- coding: utf-8 -*-
# OASIS 四分类（NonDemented / VeryMild / Mild / Moderate）
# 重点强化：少数类均衡采样 + 类平衡权重 + 局部微调骨干 + 更强小类增强 + 宏F1早停
import os, random, time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from PIL import Image, ImageOps, ImageEnhance

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

# ===== 配置 =====
DATA_DIR   = "/Users/lingjieruan/Desktop/7281a1/Data"   # ← 改成你的路径
IMG_SIZE   = 160                                        # 略升分辨率，效果更稳
BATCH_SIZE = 64
EPOCHS     = 8
LR         = 1e-3                                       # 稍降学习率，收敛更稳
SEED       = 2025
CAP_NONDEM = 6000                                       # 训练样本上限（加速）
CAP_VERY   = 6000
CAP_MILD   = 5000
CAP_MOD    = 488                                        # 现有极少
MODEL_OUT  = "oasis_fourclass_balanced.pth"
UNFREEZE_LAST_BLOCKS = True                             # 解冻最后几个block
PRINT_EVAL_CM = True

CLASSES = ["NonDemented","VeryMild","Mild","Moderate"]
CLASS2IDX = {c:i for i,c in enumerate(CLASSES)}
IDX2CLASS = {i:c for c,i in CLASS2IDX.items()}

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

def cap_per_class(cls_to_files):
    # 上限裁剪（只裁大类以加速；小类保留全部）
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
            files = random.sample(files, cap)
        print(f"{c:12s}: {len(files)}")
        X.extend(files)
        y.extend([CLASS2IDX[c]]*len(files))
    return X,y

# 小类更强的数据增强（特别是 Moderate）
class SmallClassAugment:
    def __init__(self, cls_idx):
        self.cls_idx = cls_idx
        self.enhance = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
        ])
    def __call__(self, img):
        # 轻微亮度/对比度扰动
        img = ImageOps.autocontrast(img)
        img = ImageEnhance.Brightness(img).enhance(0.9 + 0.2*random.random())
        img = ImageEnhance.Contrast(img).enhance(0.9 + 0.2*random.random())
        return self.enhance(img)

class ImgListDS(Dataset):
    def __init__(self, paths, labels, is_train=True):
        self.paths = paths; self.labels = labels; self.is_train = is_train
        self.base_train = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3),
        ])
        self.base_test = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3),
        ])
        self.aug_small = SmallClassAugment(cls_idx=CLASS2IDX["Moderate"])  # 针对 Moderate

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]; y = self.labels[i]
        img = Image.open(p).convert("RGB")
        if self.is_train:
            # 若是少数类（Mild/Moderate），额外进行一次更强增强（概率）
            if y in [CLASS2IDX["Mild"], CLASS2IDX["Moderate"]] and random.random() < 0.7:
                img = self.aug_small(img)
            return self.base_train(img), y
        else:
            return self.base_test(img), y

def stratified_split(X, y, train=0.7, val=0.15):
    X = np.array(X); y = np.array(y)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1-train, random_state=SEED)
    tr_idx, vt_idx = next(sss1.split(X, y))
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_vt, y_vt = X[vt_idx], y[vt_idx]
    val_ratio = val / (1-train)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1-val_ratio, random_state=SEED)
    v_idx, te_idx = next(sss2.split(X_vt, y_vt))
    return (X_tr.tolist(), y_tr.tolist(),
            X_vt[v_idx].tolist(), y_vt[v_idx].tolist(),
            X_vt[te_idx].tolist(), y_vt[te_idx].tolist())

def make_loaders(Xtr, ytr, Xv, yv, Xte, yte):
    tr_ds = ImgListDS(Xtr, ytr, is_train=True)
    v_ds  = ImgListDS(Xv,  yv,  is_train=False)
    te_ds = ImgListDS(Xte, yte, is_train=False)

    # —— 关键：训练用 WeightedRandomSampler，让四类在一个 epoch 内出现频次相当 —— #
    cnt = Counter(ytr)
    total = sum(cnt.values())
    # 类频率 → 采样权重（出现少的类，权重更大）
    cls_w = {c: total/cnt[c] for c in range(len(CLASSES))}
    sample_w = [cls_w[y] for y in ytr]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(ytr), replacement=True)

    pin_mem = torch.cuda.is_available()
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=pin_mem)
    v_loader  = DataLoader(v_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_mem)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_mem)
    return tr_loader, v_loader, te_loader, cnt

def build_model(num_classes=4):
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_f = m.classifier[0].in_features if hasattr(m.classifier[0],'in_features') else 576
    m.classifier = nn.Sequential(
        nn.Linear(in_f, 256),
        nn.Hardswish(),
        nn.Dropout(0.25),
        nn.Linear(256, num_classes)
    )
    if UNFREEZE_LAST_BLOCKS:
        # 解冻最后若干层（names 视 torchvision 版本略异）
        for i, (name, p) in enumerate(m.features.named_parameters()):
            p.requires_grad = False
        # 仅解冻最后3个blocks（经验值，速度与效果平衡）
        for name, p in list(m.features.named_parameters())[-6:]:
            p.requires_grad = True
    else:
        for p in m.features.parameters():
            p.requires_grad = False
    return m

def effective_num_class_weights(counts, beta=0.999):  # CB-Loss 权重
    # counts: list[int]，各类样本数
    eff_num = [(1 - beta**c)/(1 - beta) for c in counts]
    weights = [1.0/e for e in eff_num]
    s = sum(weights); weights = [w * len(counts)/s for w in weights]  # 归一：均值≈1
    return torch.tensor(weights, dtype=torch.float32)

def evaluate(model, loader, device):
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
    set_seed()
    cls_to_files = collect_paths(DATA_DIR)
    X, y = cap_per_class(cls_to_files)
    Xtr,ytr,Xv,yv,Xte,yte = stratified_split(X,y)

    tr_loader, v_loader, te_loader, train_counts = make_loaders(Xtr,ytr,Xv,yv,Xte,yte)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)

    # —— 类平衡权重（CB-Loss 思想） —— #
    cls_counts_list = [train_counts[i] if i in train_counts else 1 for i in range(len(CLASSES))]
    cb_weights = effective_num_class_weights(cls_counts_list, beta=0.999).to(device)
    criterion = nn.CrossEntropyLoss(weight=cb_weights)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

    best_macro_f1, best_state = -1, None
    patience, max_patience = 3, 3

    for ep in range(1, EPOCHS+1):
        model.train()
        tr_loss = tr_correct = tr_total = 0
        pbar = tqdm(tr_loader, desc=f"Epoch {ep}/{EPOCHS}", unit="batch")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward(); optimizer.step()

            tr_loss += loss.item()*xb.size(0)
            tr_correct += (logits.argmax(1)==yb).sum().item()
            tr_total += yb.size(0)

            pbar.set_postfix({
                "loss": f"{tr_loss/tr_total:.4f}",
                "acc":  f"{tr_correct/tr_total:.3f}"
            })

        # 验证：用 Macro-F1 作早停指标
        macro_f1, vy, vpred = evaluate(model, v_loader, device)
        scheduler.step(macro_f1)
        print(f"[INFO] epoch {ep} | val_macroF1={macro_f1:.3f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state = {"state_dict": model.state_dict()}
            torch.save(best_state, MODEL_OUT)
            patience = max_patience
        else:
            patience -= 1
            if patience == 0:
                print(f"Early stop at epoch {ep}, best val_macroF1={best_macro_f1:.3f}")
                break

    # 测试
    ckpt = torch.load(MODEL_OUT, map_location=device)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    all_y, all_pred = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            xb = xb.to(device)
            logits = model(xb)
            pred = logits.argmax(1).cpu().numpy().tolist()
            all_pred.extend(pred); all_y.extend(yb.numpy().tolist())

    if PRINT_EVAL_CM:
        print("\n=== 四分类混淆矩阵（测试集） ===")
        print(confusion_matrix(all_y, all_pred))
        print("\n=== 四分类报告（测试集） ===")
        print(classification_report(all_y, all_pred, target_names=CLASSES, digits=4))

# ========== Grad-CAM 可解释性（方式A：训练后批量出图） ==========
import cv2
import matplotlib.pyplot as plt

def _find_last_conv(module: nn.Module):
    """
    在模型内递归查找最后一个 nn.Conv2d 层，以适配不同 torchvision 版本。
    MobileNetV3_small 常见最后卷积在 features 的末尾；此函数兜底全模型遍历。
    """
    last_conv = None
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv

def _tensor_transform_for_infer():
    # 与测试时一致的预处理（保持分布一致）
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

def _np_uint8(img_pil: Image.Image, size=(IMG_SIZE, IMG_SIZE)):
    return np.array(img_pil.resize(size))

def gradcam_on_image(model: nn.Module,
                     img_path: str,
                     class_names=CLASSES,
                     target_class_idx: int = None,
                     alpha: float = 0.45,
                     out_path: str = None):
    """
    对单张图片生成 Grad-CAM 并保存叠加图。
    - model: 已加载权重且处于 eval()
    - img_path: 原图路径
    - target_class_idx: 指定解释的类别；None 则解释该图的预测 top1
    - alpha: 叠加权重（0~1）
    - out_path: 保存路径（含文件名）；None 时用同名加后缀
    返回: (out_path, pred_idx)
    """
    device = next(model.parameters()).device
    model.eval()

    target_layer = _find_last_conv(model)
    if target_layer is None:
        raise RuntimeError("未找到卷积层，无法计算 Grad-CAM。")

    conv_features, conv_grads = [], []

    def fwd_hook(module, inp, out):
        conv_features.append(out.detach())

    # 兼容新版 PyTorch 的反向 hook
    try:
        bwd_handle = target_layer.register_full_backward_hook(
            lambda module, grad_in, grad_out: conv_grads.append(grad_out[0].detach())
        )
    except Exception:
        bwd_handle = target_layer.register_backward_hook(
            lambda module, grad_in, grad_out: conv_grads.append(grad_out[0].detach())
        )
    fwd_handle = target_layer.register_forward_hook(fwd_hook)

    # 读图 & 预处理
    pil = Image.open(img_path).convert("RGB")
    ttf = _tensor_transform_for_infer()
    x = ttf(pil).unsqueeze(0).to(device)

    # 前向
    logits = model(x)
    pred_idx = int(logits.argmax(1).item()) if target_class_idx is None else int(target_class_idx)

    # 反向到指定类别分数
    model.zero_grad(set_to_none=True)
    logits[0, pred_idx].backward()

    # 取特征与梯度
    feats = conv_features[-1].cpu().numpy()[0]   # [C, H, W]
    grads = conv_grads[-1].cpu().numpy()         # [C, H, W]

    # 通道权重（GAP）
    weights = grads.mean(axis=(1, 2))            # [C]
    cam = np.zeros(feats.shape[1:], dtype=np.float32)
    for c, w in enumerate(weights):
        cam += w * feats[c]

    # ReLU & 归一化 & resize
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

    # 叠加
    base = _np_uint8(pil, (IMG_SIZE, IMG_SIZE))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[:, :, ::-1]  # BGR->RGB
    overlay = (alpha * heat + (1 - alpha) * base).astype(np.uint8)

    # 保存
    if out_path is None:
        base_name, _ = os.path.splitext(img_path)
        out_path = base_name + f"_gradcam_{class_names[pred_idx]}.png"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(overlay).save(out_path)

    # 清理 hook
    fwd_handle.remove()
    bwd_handle.remove()

    return out_path, pred_idx

def load_trained_model_for_cam(ckpt_path=MODEL_OUT):
    """
    载入训练好的权重，结构与训练保持一致。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(CLASSES)).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, device

def _gather_images_by_class(root_dir=DATA_DIR, class_names=CLASSES):
    """
    收集每个类别下的所有图片路径（递归），返回 dict: {class_name: [paths...]}
    兼容你数据集里的大小写/空格/下划线命名。
    """
    root = Path(root_dir)
    mapping = {c: [] for c in class_names}
    # 建立 dir -> 规范类名 映射
    for d in root.iterdir():
        if not d.is_dir():
            continue
        cname_std = canonicalize(d.name)
        if cname_std in mapping:
            files = [str(p) for p in d.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
            mapping[cname_std].extend(files)
    return mapping

def run_gradcam_batch(save_dir="outputs/gradcam", per_class=3, alpha=0.45, seed=SEED):
    """
    批量生成 Grad-CAM：
    - per_class: 每个类别随机挑选多少张
    - alpha: 叠加权重（建议 0.35~0.55）
    - 输出: 保存到 outputs/gradcam/<ClassName>/xxx.png
    """
    random.seed(seed)
    np.random.seed(seed)

    model, device = load_trained_model_for_cam(MODEL_OUT)
    img_dict = _gather_images_by_class(DATA_DIR, CLASSES)

    total = 0
    for cname in CLASSES:
        imgs = img_dict.get(cname, [])
        if not imgs:
            print(f"[Grad-CAM] 警告：类别 {cname} 未找到图片。")
            continue
        random.shuffle(imgs)
        sel = imgs[:min(per_class, len(imgs))]

        out_dir = os.path.join(save_dir, cname)
        os.makedirs(out_dir, exist_ok=True)

        print(f"[Grad-CAM] {cname}: 准备处理 {len(sel)} 张...")
        for p in sel:
            fname = Path(p).stem + ".png"
            out_path = os.path.join(out_dir, fname)
            try:
                out_file, pred_idx = gradcam_on_image(model, p,
                                                      class_names=CLASSES,
                                                      target_class_idx=None,     # 解释预测的top1
                                                      alpha=alpha,
                                                      out_path=out_path)
                total += 1
                print(f"  ✓ {Path(p).name} -> {Path(out_file).name} (pred={CLASSES[pred_idx]})")
            except Exception as e:
                print(f"  × {Path(p).name} 失败：{e}")

    print(f"[Grad-CAM] 完成，合计生成 {total} 张。保存目录: {save_dir}")

def make_gradcam_grid(save_root="outputs/gradcam", grid_path="outputs/gradcam_grid.png", per_class=3):
    """
    可选：将每类前 per_class 张 Grad-CAM 拼成论文图（行=类别，列=样本）。
    """
    from math import ceil
    fig_w = per_class * 3
    fig_h = len(CLASSES) * 3
    plt.figure(figsize=(fig_w, fig_h))
    idx = 1
    for r, cname in enumerate(CLASSES):
        cdir = Path(save_root) / cname
        if not cdir.exists():
            continue
        files = sorted([p for p in cdir.glob("*.png")])[:per_class]
        for c in range(per_class):
            plt.subplot(len(CLASSES), per_class, idx)
            idx += 1
            if c < len(files):
                img = Image.open(files[c]).convert("RGB")
                plt.imshow(img)
                if c == 0:
                    plt.ylabel(cname, fontsize=10)
            plt.axis("off")
    os.makedirs(Path(grid_path).parent, exist_ok=True)
    plt.tight_layout()
    plt.savefig(grid_path, dpi=300)
    print(f"[Grad-CAM] 拼图已保存: {grid_path}")

if __name__ == "__main__":
    train()


