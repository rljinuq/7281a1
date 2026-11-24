"""Chinese teaching interface with separate prediction, Grad-CAM, and Gemini actions."""

import pandas as pd
import os
import io
import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

import cv2
import gradio as gr

# ========== 1. 全局配置 ==========
CLASSES = ["NonDemented", "VeryMild", "Mild", "Moderate"]
CLASSES_CN = {
    "NonDemented": "无痴呆 (NonDemented)",
    "VeryMild": "极轻度痴呆 (VeryMild)",
    "Mild": "轻度痴呆 (Mild)",
    "Moderate": "中度痴呆 (Moderate)"
}
IDX2CLASS = {i: c for i, c in enumerate(CLASSES)}

# ⚠️ 请确保路径正确
MODEL_OUT = "/Users/lingjieruan/Desktop/7281a1/oasis_fourclass_balanced.pth"
IMG_SIZE = 160
ABSTAIN_TH = 0.50

# Gemini 设置；密钥仅从环境读取 / Read the key only from the environment
USE_GEMINI = True
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
USE_CPU_FOR_CAM = True


# ========== 2. 核心模型代码 (完整保留) ==========

def build_model(num_classes=4):
    """构建与训练一致的骨干与分类头"""
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
    """加载推理模型"""
    pred_device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    model = build_model(num_classes=len(CLASSES)).to(pred_device)

    try:
        ckpt = torch.load(ckpt_path, map_location=pred_device)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"⚠️ 模型加载失败: {e}，将使用未训练的初始化模型用于演示。")

    model.eval()
    # 预热
    with torch.inference_mode():
        _ = model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=pred_device))

    # 准备 CAM 模型 (CPU版)
    model_cam = None
    if USE_CPU_FOR_CAM and pred_device.type != "cpu":
        model_cam = build_model(num_classes=len(CLASSES)).to("cpu")
        if 'state' in locals():
            model_cam.load_state_dict(state, strict=True)
        model_cam.eval()

    return model, pred_device, model_cam


# 初始化模型
MODEL, PRED_DEVICE, MODEL_CAM = load_model(MODEL_OUT)


# ========== 3. 图像处理与核心功能 (完整保留) ==========

def preprocess_pil(pil: Image.Image):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return tf(pil.convert("RGB"))


def _find_last_conv(module: nn.Module):
    last_conv = None
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv


def gradcam_overlay(pil_img, target_idx=None):
    """计算 Grad-CAM 热力图的核心逻辑"""
    # 选择设备和模型
    cam_model = MODEL_CAM if (USE_CPU_FOR_CAM and MODEL_CAM is not None) else MODEL
    cam_device = torch.device("cpu") if (USE_CPU_FOR_CAM and MODEL_CAM is not None) else PRED_DEVICE
    cam_model.eval()

    target_layer = _find_last_conv(cam_model)
    if target_layer is None:
        return pil_img, 0  # Fallback

    feats_buf, grads_buf = [], []

    def fwd_hook(module, inp, out):
        feats_buf.append(out)

    # 注册 Hook
    try:
        bwd_handle = target_layer.register_full_backward_hook(lambda m, gi, go: grads_buf.append(go[0]))
    except:
        bwd_handle = target_layer.register_backward_hook(lambda m, gi, go: grads_buf.append(go[0]))
    fwd_handle = target_layer.register_forward_hook(fwd_hook)

    # 推理
    x = preprocess_pil(pil_img).unsqueeze(0).to(cam_device)
    logits = cam_model(x)
    pred_idx = int(torch.argmax(logits, dim=1).item()) if target_idx is None else int(target_idx)

    # 反向传播
    cam_model.zero_grad()
    logits[0, pred_idx].backward()

    # 计算 CAM
    feats = feats_buf[-1].detach().cpu().numpy()[0]
    grads = grads_buf[-1].detach().cpu().numpy()
    if grads.ndim == 4: grads = grads[0]

    weights = grads.mean(axis=(1, 2))
    cam = np.zeros(feats.shape[1:], dtype=np.float32)
    for c, w in enumerate(weights):
        cam += w * feats[c]

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)

    # 图像叠加
    w, h = pil_img.size
    cam_resized = cv2.resize(cam, (w, h))
    base = np.array(pil_img.convert("RGB").resize((w, h)))
    heat = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)[:, :, ::-1]  # RGB
    overlay = (0.45 * heat + 0.55 * base).astype(np.uint8)

    fwd_handle.remove()
    bwd_handle.remove()
    return Image.fromarray(overlay), pred_idx


def call_gemini_api(pil_overlay, pred_label_cn):
    """Gemini 调用逻辑"""
    if not USE_GEMINI or not GEMINI_API_KEY:
        return "Gemini 未启用或 Key 未配置，无法生成解释。"

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)

        model_name = "gemini-1.5-flash"
        # 尝试寻找可用模型
        try:
            for m in genai.list_models():
                if "generateContent" in m.supported_generation_methods and "gemini" in m.name:
                    model_name = m.name
                    break
        except:
            pass

        buf = io.BytesIO()
        pil_overlay.convert("RGB").save(buf, format="PNG")
        img_bytes = buf.getvalue()

        prompt = f"""
        你是一位神经放射学专家。输入是脑部MRI热力图，AI预测结果为"{pred_label_cn}"。
        请用【中文】撰写一份简短的教学分析（150字左右）：
        1. **解剖定位**：热力图高亮区域主要集中在哪些脑区？（例如海马体、颞叶内侧、脑室周围等）
        2. **病理关联**：这些区域的激活为何支持"{pred_label_cn}"的判断？
        3. **鉴别诊断**：提示需要排除的常见干扰（如伪影）。
        请保持语气客观、专业，注明仅供教学演示。
        """

        gmodel = genai.GenerativeModel(model_name)
        resp = gmodel.generate_content([{"mime_type": "image/png", "data": img_bytes}, prompt])
        return resp.text.strip()
    except Exception as e:
        return f"AI 分析服务暂时不可用: {e}"


# ========== 4. UI 逻辑函数 ==========

def ui_predict(image):
    if image is None: return "请上传图片", pd.DataFrame()
    pil = Image.fromarray(image)

    # 预测
    x = preprocess_pil(pil).unsqueeze(0).to(PRED_DEVICE)
    with torch.inference_mode():
        logits = MODEL(x)
        prob = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    top_idx = int(np.argmax(prob))
    top_prob = float(prob[top_idx])
    cls_cn = CLASSES_CN[IDX2CLASS[top_idx]]

    # 构造美化后的 HTML 结果
    color = "#fa541c" if top_idx > 0 else "#28a745"  # 有病红色，无病绿色
    res_html = f"""
    <div style="text-align: center; background-color: #f9f9f9; padding: 20px; border-radius: 12px; border: 2px solid #e0e0e0;">
        <h3 style="color: #666; margin: 0; text-transform: uppercase; font-size: 14px;">AI Prediction</h3>
        <div style="font-size: 28px; font-weight: 800; color: {color}; margin: 10px 0;">{cls_cn}</div>
        <div style="font-size: 14px; color: #888;">
            置信度 Confidence: <span style="color: #333; font-weight: bold;">{top_prob:.2%}</span>
        </div>
    </div>
    """

    bars_df = pd.DataFrame([{"类别": CLASSES_CN[IDX2CLASS[i]], "概率": float(prob[i])} for i in range(len(CLASSES))])
    return res_html, bars_df


def ui_gen_heatmap(image):
    if image is None: return None
    pil = Image.fromarray(image)
    overlay, _ = gradcam_overlay(pil)
    return overlay


def ui_gen_explanation(image):
    if image is None: return "请先上传图片。"
    pil = Image.fromarray(image)
    # 重新计算热力图以获取最新预测给AI
    overlay, pred_idx = gradcam_overlay(pil)
    pred_label = CLASSES_CN[IDX2CLASS[pred_idx]]
    return call_gemini_api(overlay, pred_label)


# ========== 5. 四宫格 UI 布局 ==========

custom_css = """
.container { max-width: 1200px; margin: auto; }
.big-btn { height: 50px !important; font-size: 16px !important; }
.panel-header { font-size: 18px; font-weight: bold; color: #444; margin-bottom: 10px; border-left: 4px solid #007bff; padding-left: 10px; }
"""

with gr.Blocks(title="MRI 智能诊断教学系统", css=custom_css) as demo:
    gr.Markdown("# 🧠 MRI 脑部影像 AI 辅助分析系统")

    # 第一行：输入 + 预测结果 (Top)
    with gr.Row():
        # 左上：输入区
        with gr.Column(scale=1):
            gr.Markdown("### 1. 影像输入 (Input)", elem_classes=["panel-header"])
            input_img = gr.Image(type="numpy", label="上传 MRI 切片", height=300)

            with gr.Row():
                btn_pred = gr.Button("🔮 运行诊断", variant="primary", elem_classes=["big-btn"])
                btn_cam = gr.Button("🔥 生成热力图", elem_classes=["big-btn"])
            btn_ai = gr.Button("🤖 Gemini 深度解读", elem_classes=["big-btn"])

        # 右上：数据区
        with gr.Column(scale=1):
            gr.Markdown("### 2. 诊断结论 (Result)", elem_classes=["panel-header"])
            out_html = gr.HTML()  # 预测结果大字展示
            out_bar = gr.BarPlot(
                x="类别", y="概率",
                x_label="诊断类别", y_label="概率值",
                title="概率分布分析",
                height=220,
                width=None,
                tooltip=["类别", "概率"],
                y_lim=[0, 1]
            )

    gr.Markdown("---")  # 分割线

    # 第二行：热力图 + AI 解释 (Bottom)
    with gr.Row():
        # 左下：热力图 (Visual)
        with gr.Column(scale=1):
            gr.Markdown("### 3. 病灶可视化 (Grad-CAM)", elem_classes=["panel-header"])
            out_heat = gr.Image(type="numpy", label="热力图叠加", height=350)

        # 右下：AI 解释 (Text)
        with gr.Column(scale=1):
            gr.Markdown("### 4. 智能教学报告 (AI Report)", elem_classes=["panel-header"])
            out_text = gr.Textbox(
                label="Gemini 分析内容",
                lines=12,
                placeholder="点击 [Gemini 深度解读] 按钮后，此处将显示 AI 针对该影像的详细病理分析...",
                show_copy_button=True
            )

    # 事件绑定
    btn_pred.click(fn=ui_predict, inputs=[input_img], outputs=[out_html, out_bar])
    btn_cam.click(fn=ui_gen_heatmap, inputs=[input_img], outputs=[out_heat])
    btn_ai.click(fn=ui_gen_explanation, inputs=[input_img], outputs=[out_text])

    gr.Markdown("> **注意**：本系统仅供医学影像教学演示，模型基于 OASIS 数据集训练，不具备临床诊断资质。")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7283, show_api=False)
