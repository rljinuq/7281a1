# gradcam_only.py  —  仅用已有 .pth 批量生成 Grad-CAM
import os, random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import matplotlib.pyplot as plt

# ===== 和你训练时保持一致的配置 =====
DATA_DIR   = "/Users/lingjieruan/Desktop/7281a1/Data"  # 改成你的数据路径
MODEL_OUT  = "oasis_fourclass_balanced.pth"
IMG_SIZE   = 160
CLASSES    = ["NonDemented","VeryMild","Mild","Moderate"]
SEED       = 2025

# ========== 构建与训练时一致的模型结构 ==========
def build_model(num_classes=4):
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_f = m.classifier[0].in_features if hasattr(m.classifier[0],'in_features') else 576
    m.classifier = nn.Sequential(
        nn.Linear(in_f, 256),
        nn.Hardswish(),
        nn.Dropout(0.25),
        nn.Linear(256, num_classes)
    )
    return m

def load_model(ckpt=MODEL_OUT):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = build_model(len(CLASSES)).to(device)
    state = torch.load(ckpt, map_location=device)
    if "state_dict" in state: state = state["state_dict"]
    m.load_state_dict(state, strict=True)
    m.eval()
    return m, device

# ========== Grad-CAM 核心 ==========
def _find_last_conv(module: nn.Module):
    last_conv = None
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv

def _tfm():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

def gradcam_one(model, img_path, alpha=0.45, out_path=None):
    device = next(model.parameters()).device
    target_layer = _find_last_conv(model)
    if target_layer is None:
        raise RuntimeError("未找到卷积层，无法计算 Grad-CAM。")

    conv_feats, conv_grads = [], []

    def fwd_hook(m, i, o): conv_feats.append(o.detach())
    try:
        bwd_handle = target_layer.register_full_backward_hook(lambda m, gin, gout: conv_grads.append(gout[0].detach()))
    except Exception:
        bwd_handle = target_layer.register_backward_hook(lambda m, gin, gout: conv_grads.append(gout[0].detach()))
    fwd_handle = target_layer.register_forward_hook(fwd_hook)

    pil = Image.open(img_path).convert("RGB")
    x = _tfm()(pil).unsqueeze(0).to(device)
    logits = model(x)
    pred = int(logits.argmax(1).item())

    model.zero_grad(set_to_none=True)
    logits[0, pred].backward()

    feats = conv_feats[-1].cpu().numpy()[0]   # [C, H, W]
    grads = conv_grads[-1].cpu().numpy()      # [C, H, W]
    weights = grads.mean(axis=(1, 2))
    cam = np.zeros(feats.shape[1:], dtype=np.float32)
    for c, w in enumerate(weights): cam += w * feats[c]
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

    base = np.array(pil.resize((IMG_SIZE, IMG_SIZE)))
    heat = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)[:, :, ::-1]  # BGR->RGB
    overlay = (alpha * heat + (1 - alpha) * base).astype(np.uint8)

    if out_path is None:
        out_path = str(Path(img_path).with_suffix("")) + f"_gradcam_{CLASSES[pred]}.png"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out_path)

    fwd_handle.remove(); bwd_handle.remove()
    return out_path, pred

def canonicalize(name: str):
    key = name.strip().lower().replace("_"," ")
    key = " ".join(key.split())
    mapping = {
        "non demented":"NonDemented","nondemented":"NonDemented",
        "very mild dementia":"VeryMild","very mild demented":"VeryMild","very mild":"VeryMild",
        "mild dementia":"Mild","mild demented":"Mild",
        "moderate dementia":"Moderate","moderate demented":"Moderate",
    }
    return mapping.get(key, name.strip())

def gather_images_per_class(root_dir):
    root = Path(root_dir)
    out = {c: [] for c in CLASSES}
    for d in root.iterdir():
        if not d.is_dir(): continue
        cname = canonicalize(d.name)
        if cname in out:
            out[cname].extend([str(p) for p in d.rglob("*") if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp"]])
    return out

def run_batch(data_dir=DATA_DIR, per_class=3, save_root="outputs/gradcam", alpha=0.45):
    random.seed(SEED); np.random.seed(SEED)
    model, device = load_model(MODEL_OUT)
    buckets = gather_images_per_class(data_dir)
    total = 0
    for cname in CLASSES:
        files = buckets.get(cname, [])
        if not files:
            print(f"[WARN] {cname} 无图片");
            continue
        random.shuffle(files)
        sel = files[:min(per_class, len(files))]
        out_dir = Path(save_root)/cname
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Grad-CAM] {cname}: {len(sel)}")
        for p in sel:
            out_path = out_dir / (Path(p).stem + ".png")
            try:
                ofile, pred = gradcam_one(model, p, alpha=alpha, out_path=str(out_path))
                total += 1
                print(f"  ✓ {Path(p).name} -> {Path(ofile).name} (pred={CLASSES[pred]})")
            except Exception as e:
                print(f"  × {Path(p).name} 失败: {e}")
    print(f"[DONE] 共生成 {total} 张，目录: {save_root}")

if __name__ == "__main__":
    # 直接运行本文件即可
    run_batch(DATA_DIR, per_class=3, save_root="outputs/gradcam", alpha=0.45)
