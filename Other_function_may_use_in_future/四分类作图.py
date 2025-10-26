import os, random
from PIL import Image
import matplotlib.pyplot as plt

root = "/Users/lingjieruan/Desktop/7281a1/Data"

# 想要的“规范顺序”
canon_order = ["NonDemented","VeryMild","Mild","Moderate"]

# 每个规范名对应的一组可能写法（含空格/大小写/不同表述）
aliases = {
    "NonDemented": ["Non Demented", "NonDemented", "non demented", "nondemented"],
    "VeryMild":    ["Very mild Dementia", "VeryMild", "very mild", "very mild demented"],
    "Mild":        ["Mild Dementia", "Mild", "mild demented"],
    "Moderate":    ["Moderate Dementia", "Moderate", "moderate demented"],
}

# 在 Data 下自动找到匹配的实际目录名
def find_dir_for(canon_name):
    cand = [a.lower().replace("_"," ").strip() for a in aliases[canon_name]]
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        key = name.lower().replace("_"," ").strip()
        if key in cand:
            return name  # 返回实际目录名（保留空格/大小写）
    raise FileNotFoundError(f"在 {root} 下找不到 {canon_name} 的目录，支持的别名有：{aliases[canon_name]}")

real_dirs = [find_dir_for(cn) for cn in canon_order]
print("匹配到的目录：", real_dirs)

def gather(dir_name, k=8):
    d = os.path.join(root, dir_name)
    files = [os.path.join(d, f) for f in os.listdir(d)
             if f.lower().endswith((".jpg",".png",".jpeg"))]
    if len(files) == 0:
        raise FileNotFoundError(f"目录里没有图片: {d}")
    if len(files) < k:
        k = len(files)
    return random.sample(files, k)

# 画 4×8 拼图
rows, cols = 4, 8
labels = ["NonDemented", "Very Mild", "Mild", "Moderate"]  # 行标题

fig, axes = plt.subplots(rows, cols, figsize=(cols*1.6, rows*1.6))

for r, dir_name in enumerate(real_dirs):  # real_dirs 是你实际匹配到的4个目录
    imgs = gather(dir_name, cols)
    for c, p in enumerate(imgs):
        img = Image.open(p).convert("L").resize((160,160))
        axes[r,c].imshow(img, cmap="gray")
        axes[r,c].axis("off")

# 在每一行最左侧写上标签（不依赖坐标轴）
for r, lab in enumerate(labels):
    # 使用 figure 坐标放文本，更不容易被裁掉
    y = 1.0 - (r + 0.5) / rows
    fig.text(0.03, y, lab, fontsize=12, va="center", ha="left")

plt.subplots_adjust(left=0.10, wspace=0.02, hspace=0.02)  # 给左侧留出空间
plt.savefig("fig_dataset_grid.png", dpi=300, bbox_inches="tight")
