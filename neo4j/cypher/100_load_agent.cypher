// 1. 设置约束
CREATE CONSTRAINT agent_name_unique IF NOT EXISTS FOR (a:Agent) REQUIRE a.name IS UNIQUE;

// 2. 加载 CSV
LOAD CSV WITH HEADERS FROM 'file:///offcourt_agents_for_kg.csv' AS row
WITH row

// 找到球员
MATCH (p:Player {id: toInteger(row.player_id)})

// 创建经纪人节点 (包含 'Unknown Agent')
MERGE (a:Agent {name: row.agent_name})

// 建立关系
MERGE (p)-[:REPRESENTED_BY]->(a);