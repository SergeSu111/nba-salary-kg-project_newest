import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from io import StringIO

# ================= DATA LOADING =================
# 将你提供的数据直接粘贴在这里
csv_data = """RefBaseline,Model,Coverage,Success_Cases,Success_Rate,Success_Rate_Pct
Stats+Time,Tabular_OnOff,423,139,0.32860520094562645,32.9%
Stats+Time,RotatE,423,138,0.3262411347517731,32.6%
Stats+Time,Node2Vec,423,121,0.2860520094562648,28.6%
Stats+Time,V1,423,146,0.34515366430260047,34.5%
Stats+Time,V2_Ind,423,109,0.2576832151300236,25.8%
Stats+Time,V2_Trans,423,127,0.30023640661938533,30.0%
Stats+Time,V2_Full,423,113,0.26713947990543735,26.7%
Stats+Time+Meta,RotatE,423,98,0.23167848699763594,23.2%
Stats+Time+Meta,Node2Vec,423,85,0.20094562647754138,20.1%
Stats+Time+Meta,V1,423,109,0.2576832151300236,25.8%
Stats+Time+Meta,V2_Ind,423,69,0.16312056737588654,16.3%
Stats+Time+Meta,V2_Trans,423,103,0.24349881796690306,24.3%
Stats+Time+Meta,V2_Full,423,84,0.19858156028368795,19.9%
"""

# 读取数据
df = pd.read_csv(StringIO(csv_data))

# 为了绘图方便，准备一个数值型的百分比列 (0-100)
df['Success_Rate_Val'] = df['Success_Rate'] * 100

# ================= PLOT STYLING =================
# 设置学术风格的绘图主题
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# 定义自定义调色板：强调 Strong Baseline，区分 Graph Models
# Tabular_OnOff 用深灰色突出，其他图模型用蓝色渐变
custom_palette = {
    "Tabular_OnOff": "#34495e",  # 深灰/黑，代表标杆
    "V1": "#2ecc71",             # 绿色，高亮表现最好的图模型
    "RotatE": "#3498db",         # 标准蓝
    "V2_Trans": "#5dade2",       # 浅一点的蓝
    "Node2Vec": "#85c1e9",
    "V2_Full": "#aed6f1",
    "V2_Ind": "#d6eaf8"          # 最浅的蓝
}

# Order models for consistency across plots
model_order = ["Tabular_OnOff", "V1", "RotatE", "V2_Trans", "Node2Vec", "V2_Full", "V2_Ind"]
# Remove Tabular_OnOff from the second plot list as it's not there
model_order_meta = [m for m in model_order if m != "Tabular_OnOff"]


# ================= PLOTTING MAIN FIGURE =================
# 创建分面图 (Catplot)
g = sns.catplot(
    data=df, kind="bar",
    x="Model", y="Success_Rate_Val", col="RefBaseline",
    palette=custom_palette,
    height=5, aspect=1.3, # 调整图表长宽比
    sharey=True, # 共享 Y 轴以便直观对比
    order=model_order, # 初始排序，第二个面会自动忽略不存在的
    edgecolor=".2" # 给柱子加个细边框更清晰
)

# ================= CUSTOMIZATION & LABELS =================

# 1. 设置标题和轴标签
g.fig.suptitle("Model Rescue Success Rate: Structure vs. Labels", y=1.05, fontweight='bold', fontsize=16)
g.set_axis_labels("", "Rescue Success Rate (%)")
g.set_titles(col_template="vs. {col_name}", fontweight='bold', size=14)

# 2. 调整 Y 轴范围和格式 (0% - 40%)
for ax in g.axes.flat:
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_ylim(0, 40) # 根据数据范围设定，留点空间放数字

# 3. 在每个柱子上添加具体的百分比数值
for ax in g.axes.flat:
    for container in ax.containers:
        # 获取柱子的高度（即百分比值）
        labels = [f'{h:.1f}%' if h > 0 else '' for h in container.datavalues]
        # 标注在柱子上方
        ax.bar_label(container, labels=labels, label_type='edge', padding=3, fontsize=11, fontweight='bold')

# 4. 旋转 X 轴标签以防重叠
for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha('right')

# 5. 添加一条辅助线强调 Strong Baseline 的表现 (仅在左图)
# Tabular_OnOff 在左图是 32.9%
left_ax = g.axes[0,0]
strong_baseline_val = df[(df['RefBaseline']=='Stats+Time') & (df['Model']=='Tabular_OnOff')]['Success_Rate_Val'].values[0]
left_ax.axhline(y=strong_baseline_val, color='#34495e', linestyle='--', linewidth=1.5, alpha=0.7)
left_ax.text(-0.5, strong_baseline_val + 0.5, "Strong Baseline\nLevel", color='#34495e', fontsize=10, ha='left')


# ================= SAVE FIGURE =================
output_path = "Figure_2_Rescue_Rates_Faceted.png"
# 使用 bbox_inches='tight' 确保标题和标签不被裁切
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ 漂亮的学术图表已保存至: {output_path}")
plt.show()