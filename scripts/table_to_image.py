import pandas as pd
import matplotlib.pyplot as plt
import io

# 1. 模拟你提供的数据
data_str = """
setting,model_type,RMSE_mean,RMSE_std,R2_mean,R2_std
Baseline (Stats+Time) RandomForest,RandomForest,0.648691,0.003598,0.688204,0.003459
RotatE + Stats RandomForest,RandomForest,0.623038,0.001353,0.712383,0.001249
Node2Vec + Stats RandomForest,RandomForest,0.625281,0.002971,0.710305,0.002750
V2_Full_SG + Stats RandomForest,RandomForest,0.651647,0.001970,0.685362,0.001905
V2_Full_MG + Stats RandomForest,RandomForest,0.652356,0.002087,0.684677,0.002017
V2_Ind + Stats RandomForest,RandomForest,0.692081,0.002284,0.645105,0.002341
V2_Trans + Stats RandomForest,RandomForest,0.713613,0.002713,0.622677,0.002873
V1 + Stats RandomForest,RandomForest,0.742682,0.001745,0.591313,0.001919
Baseline (Stats+Time) XGBoost,XGBoost,0.625748,0.004755,0.709863,0.004414
Node2Vec + Stats XGBoost,XGBoost,0.626356,0.012644,0.709218,0.011721
RotatE + Stats XGBoost,XGBoost,0.628987,0.002490,0.706862,0.002319
V2_Full_SG + Stats XGBoost,XGBoost,0.640959,0.011934,0.695517,0.011350
V2_Full_MG + Stats XGBoost,XGBoost,0.641311,0.008564,0.695223,0.008205
V1 + Stats XGBoost,XGBoost,0.726946,0.005688,0.608431,0.006132
V2_Trans + Stats XGBoost,XGBoost,0.742913,0.006693,0.591034,0.007359
V2_Ind + Stats XGBoost,XGBoost,0.754069,0.007570,0.578653,0.008465
"""

# 清洗数据
lines = [l.strip() for l in data_str.strip().split('\n')]
# 重新解析，因为原始数据格式有点乱
parsed_data = []
for line in lines[1:]:
    parts = line.split(',')
    # setting 可能是 "Baseline (Stats+Time) RandomForest"
    # model_type 是 RandomForest
    # RMSE 是 0.648691
    setting = parts[0]
    regressor = parts[1]
    rmse = float(parts[2])
    r2 = float(parts[4])
    
    # 提取 Feature Name
    feature = setting.replace(regressor, "").strip()
    parsed_data.append({'Feature': feature, 'Regressor': regressor, 'RMSE': rmse, 'R2': r2})

df = pd.DataFrame(parsed_data)

# Pivot
pivot_rmse = df.pivot(index='Feature', columns='Regressor', values='RMSE')
pivot_r2 = df.pivot(index='Feature', columns='Regressor', values='R2')

final_df = pd.DataFrame(index=pivot_rmse.index)
final_df['XGBoost (RMSE)'] = pivot_rmse['XGBoost']
final_df['XGBoost (R²)'] = pivot_r2['XGBoost']
final_df['Random Forest (RMSE)'] = pivot_rmse['RandomForest']
final_df['Random Forest (R²)'] = pivot_r2['RandomForest']

# 排序
order = [
    "Baseline (Stats+Time)", 
    "RotatE + Stats", 
    "Node2Vec + Stats", 
    "V1 + Stats", 
    "V2_Ind + Stats", 
    "V2_Trans + Stats", 
    "V2_Full_MG + Stats" # 只选 MG 代表 Full
]
final_df = final_df.reindex(order)

# 绘图
fig, ax = plt.subplots(figsize=(11, 5))
ax.axis('off')
ax.axis('tight')

# 表头
cols = ["Feature Set"] + list(final_df.columns)
cell_text = []

# 获取最优值
best_vals = {
    'XGBoost (RMSE)': final_df['XGBoost (RMSE)'].min(),
    'XGBoost (R²)': final_df['XGBoost (R²)'].max(),
    'Random Forest (RMSE)': final_df['Random Forest (RMSE)'].min(),
    'Random Forest (R²)': final_df['Random Forest (R²)'].max()
}

for idx, row in final_df.iterrows():
    row_text = [idx]
    for col in final_df.columns:
        val = row[col]
        is_best = False
        if 'RMSE' in col:
            if abs(val - best_vals[col]) < 1e-6: is_best = True
        else:
            if abs(val - best_vals[col]) < 1e-6: is_best = True
        
        txt = f"{val:.4f}"
        if is_best: txt = f"*{txt}*"
        row_text.append(txt)
    cell_text.append(row_text)

table = ax.table(cellText=cell_text, colLabels=cols, loc='center', cellLoc='center')

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)

# 样式
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#404040')
        cell.set_text_props(color='white', weight='bold')
    else:
        if "*" in cell.get_text().get_text():
            cell.get_text().set_text(cell.get_text().get_text().replace("*", ""))
            cell.get_text().set_weight('bold')
            cell.set_facecolor('#dcedc8') # 浅绿高亮

plt.title("Table 1: Model Performance (RMSE & R²)", fontweight='bold', pad=15)
plt.show()