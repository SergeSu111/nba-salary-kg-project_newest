import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# ==================== 1. 全局学术风格设置（修复 Type 3 字体 + 字体一致） ====================
# 核心修复：
# 1) 固定使用 DejaVu Sans，避免回退到 Arial/Helvetica 触发 Arial-BoldMT
# 2) 强制 pdf.fonttype=42 / ps.fonttype=42，避免 Type 3 字体
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "text.color": "#333333",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "figure.autolayout": False,

    # ✅ 关键：避免 Type 3
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ==================== 2. 数据准备 ====================
models_display = [
    'Tabular Meta', 'RotatE', 'Node2Vec', 'V1 (Static GNN)',
    'V2-Ind (Dynamic)', 'V2-Trans (Dynamic)', 'V2-Full (Hetero)'
]
y_pos = np.arange(len(models_display))

# (a) Weak Baseline
rescue_weak = np.array([45.1, 44.8, 39.3, 47.4, 35.4, 41.2, 36.7])
misguide_weak = np.array([-11.7, -20.5, -17.9, -37.0, -41.2, -35.1, -17.5])

# (b) Strong Baseline
rescue_strong = np.array([0.0, 31.9, 27.7, 35.5, 22.5, 33.6, 27.4])
misguide_strong = np.array([0.0, -35.5, -36.5, -45.0, -52.4, -47.6, -36.5])

# ==================== 3. 绘图参数 ====================
COLOR_RESCUE = '#00796B'
COLOR_MISGUIDE = '#C62828'
BAR_HEIGHT = 0.7
EDGE_COLOR = 'white'
LINEWIDTH = 1.0

fig, axes = plt.subplots(
    1, 2, figsize=(15, 6), sharey=True,
    gridspec_kw={'wspace': 0.05}
)
fig.patch.set_facecolor('white')

# ==================== 4. 绘图主函数（确保百分比数字显示） ====================
def plot_bars(ax, y_positions, rescues, misguides, title):
    bars_r = ax.barh(
        y_positions, rescues, height=BAR_HEIGHT, align='center',
        color=COLOR_RESCUE, edgecolor=EDGE_COLOR, linewidth=LINEWIDTH, zorder=3
    )
    bars_m = ax.barh(
        y_positions, misguides, height=BAR_HEIGHT, align='center',
        color=COLOR_MISGUIDE, edgecolor=EDGE_COLOR, linewidth=LINEWIDTH, zorder=3
    )

    # ✅ 关键修改：不再用阈值过滤；并把文本强制放在条形内部，防止被裁掉/看不见
    for bar in bars_r:
        w = float(bar.get_width())
        y = bar.get_y() + bar.get_height() / 2

        # 放在绿色条形内部靠右（留一点边距）
        x = w - 1.2 if w >= 2 else w + 1.2
        ha = 'right' if w >= 2 else 'left'

        ax.text(
            x, y, f'{w:.1f}%',
            ha=ha, va='center',
            color='white', fontsize=10, fontweight='bold',
            zorder=5, clip_on=True
        )

    for bar in bars_m:
        w = float(bar.get_width())  # 负数
        y = bar.get_y() + bar.get_height() / 2

        # 放在红色条形内部靠左（负方向条形内部）
        # 红条从 0 往左延伸，内部靠近 0 的位置是 (w + 小正数)
        x = w + 1.2 if w <= -2 else w - 1.2
        ha = 'left' if w <= -2 else 'right'

        ax.text(
            x, y, f'{abs(w):.1f}%',
            ha=ha, va='center',
            color='white', fontsize=10, fontweight='bold',
            zorder=5, clip_on=True
        )

    ax.set_title(title, fontweight='bold', pad=15)

    # 坐标轴与网格
    ax.set_xlim(-65, 65)
    xticks = np.arange(-60, 61, 20)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(x)}" for x in xticks])

    ax.axvline(0, color='#555555', linewidth=1.5, zorder=4)
    ax.grid(axis='x', linestyle=':', color='#cccccc', linewidth=0.8, zorder=0)

    # 极简边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#999999')
    ax.tick_params(left=False)

    return bars_r, bars_m

# 左图：全量
plot_bars(axes[0], y_pos, rescue_weak, misguide_weak, '(a) vs. Weak Baseline (Stats + Time)')

# 右图：去掉 Tabular Meta 那一行（与数据长度匹配）
plot_bars(axes[1], y_pos[1:], rescue_strong[1:], misguide_strong[1:], '(b) vs. Strong Baseline (Stats + Time + Meta)')

# Y 轴标签（只在左边）
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(models_display, fontweight='medium')

# 全局 X 轴标签
fig.text(0.52, 0.02, 'Percentage of Eligible Outliers (%)', ha='center', fontsize=13, fontweight='medium')

# ==================== 5. 注释（Generalization Tax） ====================
ax2 = axes[1]
target_y_index = 4  # V2-Ind 在全局列表中的索引
target_y = y_pos[target_y_index]
target_x = misguide_strong[target_y_index]  # -52.4

ax2.annotate(
    'Generalization Tax\n(Cold-Start Failure)',
    xy=(target_x, target_y), xycoords='data',
    xytext=(-58, target_y - 2.5), textcoords='data',
    arrowprops=dict(arrowstyle="->", color=COLOR_MISGUIDE, lw=2, connectionstyle="arc3,rad=-0.2"),
    color=COLOR_MISGUIDE, fontsize=10, fontweight='bold', ha='center'
)

# ==================== 6. 图例 ====================
legend_elements = [
    mpatches.Patch(facecolor=COLOR_RESCUE, edgecolor=EDGE_COLOR, label='Successful Rescue\n(Error Reduced > $0.5M)'),
    mpatches.Patch(facecolor=COLOR_MISGUIDE, edgecolor=EDGE_COLOR, label='Structural Misguidance\n(Error Increased > $0.5M)')
]
fig.legend(
    handles=legend_elements, loc='upper center',
    bbox_to_anchor=(0.52, 1.08),
    ncol=2, frameon=False, fontsize=12,
    handlelength=1.5, handleheight=1.5
)

plt.subplots_adjust(top=0.85, bottom=0.15)

# ==================== 7. 保存输出 ====================
output_pdf = 'Figure3_TriState_Ultimate_Fixed.pdf'
plt.savefig(output_pdf, bbox_inches='tight', pad_inches=0.1)
print(f"✅ 已生成（百分比数字保证显示 + 避免 Type 3 风险）: {output_pdf}")

# 如果你也想要 PNG（方便预览/插入 Word），取消注释：
# plt.savefig('Figure3_TriState_Ultimate_Fixed.png', dpi=600, bbox_inches='tight', pad_inches=0.1)

# plt.show()