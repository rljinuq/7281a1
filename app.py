# agent_infer_app.py
# -*- coding: utf-8 -*-
"""
Two-stage AI agent:
- Action A: upload a single MRI slice -> 4-class probs + confidence + caution
- Action B: click "Generate Explanation" -> Grad-CAM overlay (original size) + Gemini text
"""

import pandas as pd
import os, io, time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

import cv2
import gradio as gr  # pip install gradio

# ========== Config ==========
CLASSES = ["NonDemented", "VeryMild", "Mild", "Moderate"]
IDX2CLASS = {i: c for i, c in enumerate(CLASSES)}

MODEL_OUT = "/Users/lingjieruan/Desktop/7281a1/oasis_fourclass_balanced.pth"
IMG_SIZE = 160
ABSTAIN_TH = 0.50

# ✅ 直接使用你已写入的 API key；可正常运行
USE_GEMINI = True
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDRGJbVKdsulrkBD_Aq1U1JXRnD5vOuaOg")

# 对于 macOS + MPS，建议在 CPU 上算 CAM 更稳
USE_CPU_FOR_CAM = True

# ========== Model definition ==========
def build_model(num_classes=4):
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_f = m.classifier[0].in_features if hasattr(m.classifier[0], "in_features") else 576
    m.classifier = nn.Sequential(
        nn.Linear(in_f, 256),
        nn.Hardswish(),
        nn.Dropout(0.25),
        nn.Linear(256, num_classes)
    )
    return m

def load_model(ckpt_path=MODEL_OUT):
    pred_device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    model = build_model(num_classes=len(CLASSES)).to(pred_device)
    ckpt = torch.load(ckpt_path, map_location=pred_device)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.inference_mode():
        _ = model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=pred_device))
    model_cam = None
    if USE_CPU_FOR_CAM and pred_device.type != "cpu":
        model_cam = build_model(num_classes=len(CLASSES)).to("cpu")
        model_cam.load_state_dict(state, strict=True)
        model_cam.eval()
    return model, pred_device, model_cam

# ========== Preprocess ==========
def preprocess_pil(pil: Image.Image):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    return tf(pil.convert("RGB"))

# ========== Action A: predict ==========
def predict_one(model, device, pil: Image.Image):
    x = preprocess_pil(pil).unsqueeze(0).to(device)
    t0 = time.time()
    with torch.inference_mode():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    dt = (time.time() - t0) * 1000
    top_idx = int(np.argmax(prob))
    top_prob = float(prob[top_idx])
    return {
        "top_class": IDX2CLASS[top_idx],
        "top_prob": top_prob,
        "uncertainty": float(1.0 - top_prob),
        "probs": {IDX2CLASS[i]: float(prob[i]) for i in range(len(CLASSES))},
        "latency_ms": round(dt, 1),
        "advice": "⚠️ Low confidence. Please review manually." if top_prob < ABSTAIN_TH else ""
    }

# ========== Grad-CAM (输出为原图尺寸) ==========
def _find_last_conv(module: nn.Module):
    last_conv = None
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv

def gradcam_overlay(model, device, pil: Image.Image, target_idx: int=None, alpha: float=0.45):
    cam_model = MODEL_CAM if (USE_CPU_FOR_CAM and MODEL_CAM is not None) else model
    cam_device = torch.device("cpu") if (USE_CPU_FOR_CAM and MODEL_CAM is not None) else device
    cam_model.eval()
    target_layer = _find_last_conv(cam_model)
    if target_layer is None:
        raise RuntimeError("No Conv2d layer found; cannot compute Grad-CAM.")

    feats_buf, grads_buf = [], []

    def fwd_hook(module, inp, out):
        feats_buf.append(out)
    try:
        bwd_handle = target_layer.register_full_backward_hook(
            lambda module, gin, gout: grads_buf.append(gout[0])
        )
    except Exception:
        bwd_handle = target_layer.register_backward_hook(
            lambda module, gin, gout: grads_buf.append(gout[0])
        )
    fwd_handle = target_layer.register_forward_hook(fwd_hook)

    x = preprocess_pil(pil).unsqueeze(0).to(cam_device)
    logits = cam_model(x)
    pred_idx = int(torch.argmax(logits, dim=1).item()) if target_idx is None else int(target_idx)

    cam_model.zero_grad(set_to_none=True)
    logits[0, pred_idx].backward()

    feats = feats_buf[-1].detach().cpu().numpy()[0]   # [C,H,W]
    grads = grads_buf[-1].detach().cpu().numpy()      # [C,H,W] or [1,C,H,W]
    if grads.ndim == 4:
        grads = grads[0]
    weights = grads.mean(axis=(1,2))
    cam = np.zeros(feats.shape[1:], dtype=np.float32)
    for c, w in enumerate(weights):
        cam += w * feats[c]
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)

    # 🔧 关键修复：把 CAM 放大到原图尺寸，而不是 128x128
    w, h = pil.size
    cam = cv2.resize(cam, (w, h))
    base = np.array(pil.convert("RGB").resize((w, h)))
    heat = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)[:, :, ::-1]
    overlay = (alpha*heat + (1-alpha)*base).astype(np.uint8)

    fwd_handle.remove(); bwd_handle.remove()
    return Image.fromarray(overlay), pred_idx

# ========== Gemini Explanation ==========
def explain_with_gemini(pil_overlay: Image.Image, pred_label: str):
    """
    自动列出可用模型并选择一个支持 generateContent 且可接收 image 的模型来生成解释。
    不再依赖硬编码模型名，能规避 404/不支持 等版本差异问题。
    """
    if not USE_GEMINI or not GEMINI_API_KEY:
        return "(Gemini disabled or API key not set. Highlighted areas show model focus; for teaching only.)"

    try:
        import google.generativeai as genai
    except Exception as e:
        return f"(Gemini SDK import failed: {e}. Try: pip install -U google-generativeai)"

    try:
        # 1) 配置并列出模型
        genai.configure(api_key=GEMINI_API_KEY)
        try:
            models = list(genai.list_models())
        except Exception as e_list:
            return f"(Gemini list_models failed: {e_list}. Try: pip install -U google-generativeai)"

        # 2) 过滤：必须支持 generateContent；尽量要求可以接收 image
        candidates = []
        for m in models:
            name = getattr(m, "name", None)
            methods = set(getattr(m, "supported_generation_methods", []) or [])
            input_modalities = set(getattr(m, "input_modalities", []) or [])
            if name and ("generateContent" in methods):
                # 优先能吃 image 的；否则先收集起来做兜底
                if ("image" in input_modalities) or (len(input_modalities) == 0):
                    candidates.append((name, "image" in input_modalities))

        if not candidates:
            return "(No Gemini model supporting generateContent found for your account. Please update google-generativeai or check AI Studio quotas.)"

        # 3) 选择策略：优先 image 支持的；并尽量优先包含 'gemini' / 'flash' / 'pro' 的型号
        def rank(item):
            name, has_image = item
            score = 0
            if has_image: score += 100
            if "gemini" in name: score += 10
            if "flash" in name: score += 5
            if "pro" in name: score += 3
            if "latest" in name: score += 1
            return -score  # 排序时升序，所以取负值
        candidates.sort(key=rank)
        model_name = candidates[0][0]

        # 4) 组织输入并生成
        buf = io.BytesIO()
        pil_overlay.convert("RGB").save(buf, format="PNG")
        img_bytes = buf.getvalue()

        prompt = f"""
        You are a neuroradiology teaching assistant. Input: one axial brain MRI slice with a Grad-CAM heatmap. 
        Predicted stage: "{pred_label}". Write a focused explanation of 3–5 sentences (≈120–180 words).

        Cover these points explicitly:
        • Localization & laterality: name one or two structures ONLY from this list — hippocampus, entorhinal cortex, 
          parahippocampal gyrus, temporal pole, lateral temporal cortex, posterior cingulate/precuneus, lateral parietal cortex, 
          medial temporal lobe, insula, dorsolateral prefrontal cortex — and state right/left/bilateral/asymmetric (R>L or L>R). 
          If laterality is unclear, say “uncertain laterality”.
        • Heatmap–anatomy relationship: describe whether the activation follows gray matter/cortical ribbon or medial temporal 
          contours vs. non-neural structures (eyeballs, skull/edge, venous sinuses). Mention shape/intensity (focal vs diffuse), 
          and whether it aligns with expected anatomy on T1. Avoid inventing structures outside the list.
        • Stage linkage: explain why attention in those regions could relate to "{pred_label}" at a network level 
          (e.g., medial-temporal memory circuits, Default Mode Network hubs in posterior cingulate/precuneus, 
          temporal–parietal semantic systems). Use calibrated language (“may be”, “suggestive of”, “consistent with”) 
          rather than definitive claims.
        • Uncertainty & artifacts: note at least one plausible confound (motion, coil/edge effect, intensity normalization, 
          partial-volume) and what further evidence would increase confidence (agreement across adjacent slices, multi-sequence 
          concordance such as T2/FLAIR, or clinical correlation).

        End with: “For teaching demonstration only, not a clinical diagnosis.”
        """.strip()

        # 有的 SDK 版本接受 [image, text] 的简写，有的要 parts；这里用通用的数组形式
        try:
            gmodel = genai.GenerativeModel(model_name)
            resp = gmodel.generate_content([
                {"mime_type": "image/png", "data": img_bytes},
                prompt
            ])
            return resp.text.strip() if hasattr(resp, "text") and resp.text else str(resp)
        except Exception as e_gen:
            # 显示我们选到的模型名，方便定位问题
            return f"(Gemini generation failed with model '{model_name}': {e_gen}. Try: pip install -U google-generativeai)"
    except Exception as e:
        return f"(Gemini explanation failed: {e}. For teaching demonstration only.)"

# ========== Gradio UI ==========
MODEL, PRED_DEVICE, MODEL_CAM = load_model(MODEL_OUT)

def ui_predict(image):
    if image is None:
        return "Please upload an image first.", pd.DataFrame([{"class": "-", "prob": 0.0}]), None
    pil = Image.fromarray(image)
    out = predict_one(MODEL, PRED_DEVICE, pil)
    lines = [
        f"Prediction: **{out['top_class']}**",
        f"Confidence: **{out['top_prob']:.3f}** (uncertainty {out['uncertainty']:.3f})",
        f"Latency: {out['latency_ms']} ms",
    ]
    if out['advice']:
        lines.append(out['advice'])
    bars_df = pd.DataFrame([{"class": k, "prob": float(v)} for k, v in out["probs"].items()])
    return "\n\n".join(lines), bars_df, None

def ui_explain(image):
    if image is None:
        return None, "Please upload an image first."
    pil = Image.fromarray(image)
    overlay, pred_idx = gradcam_overlay(MODEL, PRED_DEVICE, pil)
    pred_label = IDX2CLASS[pred_idx]
    text = explain_with_gemini(overlay, pred_label)
    return overlay, text

with gr.Blocks(title="OASIS 4-class Agent (Education Only)") as demo:
    gr.Markdown("### 🧠 OASIS 4-class AI Agent — Educational Demonstration (Not for Clinical Use)")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="numpy", label="Upload Brain MRI Slice")
            btn_pred = gr.Button("🔮 Predict Now")
            btn_exp = gr.Button("🧩 Generate Explanation (Grad-CAM + Gemini)")
        with gr.Column(scale=1):
            txt = gr.Markdown()
            bars = gr.BarPlot(
                value=pd.DataFrame([{"class": "-", "prob": 0.0}]),
                x="class", y="prob", title="Class Probabilities", x_label="class", y_label="prob",
                width=500, height=300
            )
            heat = gr.Image(label="Grad-CAM Overlay")
            txt_exp = gr.Markdown()

    btn_pred.click(fn=ui_predict, inputs=[inp], outputs=[txt, bars, heat])
    btn_exp.click(fn=ui_explain, inputs=[inp], outputs=[heat, txt_exp])

    gr.Markdown("> Trained on de-identified OASIS dataset. For teaching only, not a clinical diagnosis.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7281, show_api=False)


