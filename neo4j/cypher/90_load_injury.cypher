LOAD CSV WITH HEADERS FROM 'file:///offcourt_injury_for_kg.csv' AS row
MATCH (p:Player {player_id: toInteger(row.player_id)})
MERGE (i:InjuryType {name: row.injury_category})
// 创建伤病关系
CREATE (p)-[r:SUFFERED_INJURY]->(i)
SET r.date = date(row.date),
    r.description = row.description,
    r.season = toInteger(row.season); // 这里现在是完美的整数年份