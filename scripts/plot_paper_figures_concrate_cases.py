import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ================= CONFIG =================
# ⚠️ 这里改成你刚刚生成结果的文件夹路径 (和上一步一样)
PRED_DIR = Path("runs/final_eval_strict_v3/20260207_161729/predictions") 

# ================= SETTINGS =================
# 设置论文风格的绘图参数
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300
})

def plot_concrete_examples():
    """图1: 展示具体的救援案例 (Error Reduction)"""
    csv_path = PRED_DIR / "summary_concrete_examples_numeric.csv"
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # 筛选: 每个模型只选 Rescue 最大的 1 个案例画图 (避免图太挤)
    # 或者你可以手动指定几个名字，比如 ['Khris Middleton', 'Fred VanVleet', 'Coby White']
    top_cases = df.sort_values('Rescue_Amount', ascending=False).groupby('Model').head(1)
    
    # 数据变换 (Melt) 以适配 Seaborn
    plot_data = top_cases[['Model', 'Player', 'Base_Err', 'Graph_Err']]
    plot_data = plot_data.melt(id_vars=['Model', 'Player'], var_name='Error_Type', value_name='Error_USD')
    
    # 将金额转换为 Million
    plot_data['Error_Million'] = plot_data['Error_USD'] / 1e6
    plot_data['Error_Type'] = plot_data['Error_Type'].map({
        'Base_Err': 'Baseline Error', 
        'Graph_Err': 'Graph Model Error'
    })

    # 创建标签: "Player (Model)"
    plot_data['Label'] = plot_data['Player'] + "\n(" + plot_data['Model'] + ")"

    # 绘图
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=plot_data,
        x='Label',
        y='Error_Million',
        hue='Error_Type',
        palette={'Baseline Error': '#95a5a6', 'Graph Model Error': '#2ecc71'} # 灰 vs 绿
    )

    plt.title('Valuation Error Reduction: Graph Models vs. Baseline', fontweight='bold')
    plt.ylabel('Absolute Error ($ Million)')
    plt.xlabel('Player Case Study (Model)')
    plt.legend(title=None)
    
    # 在柱子上标数值
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1fM', padding=3)

    plt.tight_layout()
    save_path = PRED_DIR / "figure_rescue_cases.png"
    plt.savefig(save_path)
    print(f"✅ Figure 1 saved to: {save_path}")

def plot_success_rates():
    """图2: 模型成功率对比"""
    csv_path = PRED_DIR / "summary_model_coverage.csv"
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # 排序
    df = df.sort_values('Success_Rate', ascending=False)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df,
        x='Model',
        y='Success_Rate',
        palette="viridis"
    )

    plt.title('Rescue Success Rate by Model', fontweight='bold')
    plt.ylabel('Success Rate (Cases with Rescue > $0.5M)')
    plt.xlabel('Graph Model Variant')
    plt.ylim(0, df['Success_Rate'].max() * 1.2) # 留点空间给标签

    # 标数值 (百分比)
    for i, v in enumerate(df['Success_Rate']):
        ax.text(i, v + 0.005, f"{v*100:.1f}%", ha='center', fontweight='bold')

    plt.tight_layout()
    save_path = PRED_DIR / "figure_success_rates.png"
    plt.savefig(save_path)
    print(f"✅ Figure 2 saved to: {save_path}")

if __name__ == "__main__":
    plot_concrete_examples()
    plot_success_rates()