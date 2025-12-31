import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"

# 数据
models = ["CastDet*",
          "PKINet*",
          "YOLO-World-L",
          "Grounding DINO-T",
          "OpenRSD-OBB (Ours)",
          "OpenRSD-HBB (Ours)"]
aps = [60.2, 49.2, 62.7,
       69.3, 69.2, 71.4]  # 每个模型的 Precision
fps = [6.8, 5.2, 17.2,
       5.4, 20.8, 20.8]      # 每个模型的 FPS
shapes = ["^", "s", "o", "p", "*", "*"]  # 图形形状
colors = ["orange", "green", "lightblue", "gray", "darkred", "red"]  # 图形颜色
sizes = [300, 300, 300, 300, 500, 500]  # 图形大小（用于突出某些点）

# 绘制图表
fig, ax = plt.subplots(figsize=(8, 6))  # 可自定义 fig_size
ax.grid(True, linestyle="--", alpha=0.6)

for model, recall, fp, shape, color, size in zip(models, aps, fps, shapes, colors, sizes):
    ax.scatter(fp, recall, label=model, s=size, c=color,
               marker=shape,
               # edgecolors='black',
               # linewidths=0.5
               )

# 图表设置
ax.set_xlabel("Frame Per Second", fontsize=20)
ax.set_ylabel("Mean Average Precision", fontsize=20)
# ax.set_xlabel("Frames Per Second", fontsize=18)
# ax.set_ylabel("Average Precision", fontsize=18)
# ax.set_title("Model Performance Comparison", fontsize=16)
ax.legend(title="Methods", fontsize=16, title_fontsize=16, loc="lower right")

ax.set_xlim(0, 25)
ax.set_xticks([0, 5, 10, 15, 20, 25])  # 设置 x 轴标签
ax.tick_params(axis="both", labelsize=20)

# 保存图表（可选）
plt.savefig('./fig1_compare.png', dpi=300)
