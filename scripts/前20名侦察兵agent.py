import pandas as pd

# 配置
META_FILE = 'data/processed/training_level1_full.csv'

def check_top_agents():
    # 读取数据
    df = pd.read_csv(META_FILE, low_memory=False)
    
    # 自动修正列名 (沿用之前的逻辑)
    if 'log1p_3PM' in df.columns:
        agent_col = 'log1p_3PM'
    else:
        agent_col = 'agent_name'
    
    print(f"正在读取列: '{agent_col}'...\n")
    
    # 统计排名
    top_agents = df[agent_col].value_counts().head(20)
    
    print("=== 你的数据里排名前 20 的经纪人 ===")
    for name, count in top_agents.items():
        print(f"人数: {count} | 名字: '{name}'")
    print("====================================")

if __name__ == "__main__":
    check_top_agents()