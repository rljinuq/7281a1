# make_gradcam_grid.py
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from PIL import Image

SAVE_ROOT = "outputs/gradcam"                      # 你的Grad-CAM文件夹
CLASSES   = ["NonDemented","VeryMild","Mild","Moderate"]
PER_CLASS = 3                                      # 每类展示几张
OUT_PATH  = "outputs/gradcam_grid_labeled.png"     # 拼图保存路径

def make_gradcam_grid(save_root=SAVE_ROOT, out_path=OUT_PATH, per_class=PER_CLASS):
    rows, cols = len(CLASSES), per_class

    # 画布大小：给左侧行标题、右侧颜色条留位置
    fig = plt.figure(figsize=(cols*3.0 + 1.5, rows*3.0))
    gs  = fig.add_gridspec(rows, cols, left=0.14, right=0.90, top=0.95, bottom=0.05,
                           wspace=0.02, hspace=0.02)

    for r, cname in enumerate(CLASSES):
        files = sorted((Path(save_root)/cname).glob("*.png"))[:cols]
        # 行标题（类别名）
        fig.text(0.03, 1-(r+0.5)/rows, cname, ha="left", va="center",
                 fontsize=14, fontweight="bold")

        for c in range(cols):
            ax = fig.add_subplot(gs[r, c])
            if c < len(files):
                img = Image.open(files[c]).convert("RGB")
                ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])

    # 列标题（可有可无）
    for c in range(cols):
        fig.text(0.16 + (0.74/cols)*(c+0.5), 0.985, f"sample {c+1}",
                 ha="center", va="top", fontsize=11)

    # 颜色条（红=高关注，蓝=低关注）
    cmap = plt.get_cmap("jet")
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cax  = fig.add_axes([0.92, 0.15, 0.02, 0.70])   # 位置：右侧竖直颜色条
    cb   = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_label("Grad-CAM ", fontsize=11)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"[拼图完成] 已保存: {out_path}")

if __name__ == "__main__":
    make_gradcam_grid()
