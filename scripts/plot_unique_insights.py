import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ================= CONFIG =================
PRED_DIR = Path("runs/final_eval_strict_v3/20260207_161729/predictions") 

# ================= SETTINGS =================
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

def plot_unique_supremacy():
    csv_path = PRED_DIR / "summary_unique_insights_numeric.csv"
    if not csv_path.exists():
        print("File not found.")
        return

    df = pd.read_csv(csv_path)
    
    # 策略：每个模型只选 1 个最强案例（Rescue Advantage 最大）来画，避免图太乱
    # 这样图里会有 6 组柱子
    top_cases = df.sort_values('Rescue_Advantage', ascending=False).groupby('Model').head(1)
    
    # 排序：按优势大小排序，让图更好看
    top_cases = top_cases.sort_values('Rescue_Advantage', ascending=False)
    
    # 准备数据
    models = top_cases['Model'].tolist()
    players = top_cases['Player'].tolist()
    my_errs = top_cases['My_Err'] / 1e6
    peer_errs = top_cases['Peer_Best_Err'] / 1e6
    
    x = range(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 画两组柱子
    # 1. 灰色柱子：代表"其他模型里最好的那个"（Peer Best）
    bars2 = ax.bar([i + width/2 for i in x], peer_errs, width, label='Next Best Model Error', color='#95a5a6', alpha=0.6)
    
    # 2. 彩色柱子：代表"我的模型"（My Model）
    # 使用不同颜色区分模型
    colors = sns.color_palette("viridis", n_colors=len(models))
    bars1 = ax.bar([i - width/2 for i in x], my_errs, width, label='My Model Error', color=colors)
    
    # 添加箭头或连线，强调差距
    for i in x:
        # 在两个柱子中间画一个箭头，从 Peer 指向 My (表示 Error 下降)
        ax.annotate(
            "", 
            xy=(i - width/2, my_errs.iloc[i]), 
            xytext=(i + width/2, peer_errs.iloc[i]),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5)
        )
        # 标出优势金额
        gap = peer_errs.iloc[i] - my_errs.iloc[i]
        ax.text(i, peer_errs.iloc[i] + 0.5, f"-${gap:.1f}M", ha='center', color='red', fontweight='bold', fontsize=10)

    # 标签设置
    ax.set_ylabel('Absolute Error ($ Million)')
    ax.set_title('Absolute Supremacy: Unique Insights by Model', fontweight='bold')
    ax.set_xticks(x)
    # X轴标签格式：Player \n (Model)
    ax.set_xticklabels([f"{p}\n({m})" for p, m in zip(players, models)])
    ax.legend()
    
    plt.tight_layout()
    save_path = PRED_DIR / "figure_unique_supremacy.png"
    plt.savefig(save_path)
    print(f"✅ Figure 3 saved to: {save_path}")

if __name__ == "__main__":
    plot_unique_supremacy()