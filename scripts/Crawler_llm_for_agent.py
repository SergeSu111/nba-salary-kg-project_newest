import pandas as pd
from bs4 import BeautifulSoup
import os

# ================= 配置 =================
HTML_FILE = "NBA Player Agent Relationships - RealGM.html"  # 您的本地文件
ID_FILE = "../data/raw_on_court/unique_player_id.csv"
OUTPUT_DIR = "../neo4j/import"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "offcourt_agents_for_kg.csv")

def clean_name(name):
    if not isinstance(name, str): return ""
    name = name.strip()
    # 处理 "Last, First" 格式
    if "," in name:
        parts = name.split(",")
        if len(parts) >= 2:
            return f"{parts[1].strip()} {parts[0].strip()}"
    return name

def parse_html_dynamic(file_path):
    print(f"📖 正在读取: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f: html = f.read()
    except:
        with open(file_path, "r", encoding="iso-8859-1") as f: html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    
    target_table = None
    header_map = {} 

    print(f"🔍 正在寻找包含 Agent 信息的表格...")
    
    for table in tables:
        # 获取所有表头文本 (转小写)
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        
        # 只要包含 "agent" 且包含 "player" 或 "client"，就是我们要的表
        if "agent" in headers and (any(x in headers for x in ["client", "player"])):
            target_table = table
            # 建立列索引映射
            for idx, h in enumerate(headers):
                if h in ["client", "player"]: header_map["player"] = idx
                if h == "agent": header_map["agent"] = idx
            break
            
    if not target_table or "agent" not in header_map:
        print("❌ 失败：未找到正确的表格或表头。")
        return None

    print(f"✅ 锁定表格！列索引: {header_map}")
    
    data = []
    rows = target_table.find("tbody").find_all("tr")
    
    for row in rows:
        cols = row.find_all("td")
        if not cols: continue
        
        try:
            p_idx = header_map["player"]
            ag_idx = header_map["agent"]
            
            player_raw = cols[p_idx].get_text(strip=True)
            # 使用 separator 防止名字粘连 (如果有多个经纪人)
            agent_raw = cols[ag_idx].get_text(separator=" / ", strip=True)
            
            if not player_raw or player_raw.lower() == "player": continue

            data.append({
                "player_name_clean": clean_name(player_raw),
                "agent_name": agent_raw
            })
        except IndexError:
            continue
            
    return pd.DataFrame(data)

def main():
    # 1. 解析 RealGM 数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(script_dir, HTML_FILE)
    
    df_agents = parse_html_dynamic(html_path)
    if df_agents is None or df_agents.empty: return

    print(f"🎉 从网页提取到 {len(df_agents)} 条记录。")

    # 2. 加载您的 ID 表 (主表)
    id_path = os.path.join(script_dir, ID_FILE)
    if not os.path.exists(id_path):
        print("❌ 错误：找不到 ID 文件！")
        return

    print("\n🔄 正在进行全量匹配 (Left Join)...")
    df_ids = pd.read_csv(id_path)
    
    # 统一列名
    if "Player_id" in df_ids.columns:
        df_ids = df_ids.rename(columns={"Player_id": "player_id", "Player": "player_name"})
    
    df_ids['player_name_clean'] = df_ids['player_name'].apply(clean_name)
    
    # 3. 核心步骤：左连接 (Left Join)
    # 以 ID 表为准，保留所有球员。如果 RealGM 里没找到，agent_name 就是 NaN
    df_merged = pd.merge(df_ids[['player_id', 'player_name_clean']], 
                         df_agents, 
                         on='player_name_clean', 
                         how='left')
    
    # 4. 填充缺失值为 "Unknown Agent"
    fill_value = "Unknown Agent"
    missing_count = df_merged['agent_name'].isna().sum()
    df_merged['agent_name'] = df_merged['agent_name'].fillna(fill_value)
    
    # 处理空字符串的情况
    df_merged.loc[df_merged['agent_name'] == '', 'agent_name'] = fill_value

    print(f"✅ 处理完成！")
    print(f"   - 总球员数: {len(df_merged)}")
    print(f"   - 成功匹配经纪人: {len(df_merged) - missing_count}")
    print(f"   - 设为 '{fill_value}': {missing_count} (包含退役/未收录球员)")
    
    # 5. 保存
    final_df = df_merged[['player_id', 'agent_name']]
    output_path = os.path.join(script_dir, OUTPUT_FILE)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    final_df.to_csv(output_path, index=False)
    print(f"\n💾 最终文件已保存: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()