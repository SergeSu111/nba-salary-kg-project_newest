// 1) 建 PlayerSeason 节点 + 连接到 Player / Season
USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM 'file:///oncourt_core_for_kg.csv' AS row
WITH row,
     toInteger(row.player_id) AS pid,
     toInteger(row.season)    AS yr
WHERE pid IS NOT NULL AND yr IS NOT NULL
MERGE (p:Player {player_id: pid})
MERGE (s:Season {year: yr})
MERGE (ps:PlayerSeason {player_id: pid, season: yr})
  ON CREATE SET ps.createdAt = timestamp()
MERGE (p)-[:IN_SEASON]->(ps)
MERGE (ps)-[:OF_SEASON]->(s);

// 2) （可选）示例：如果你的 CSV 里确实有这些列，就解除注释写属性
// USING PERIODIC COMMIT
// LOAD CSV WITH HEADERS FROM 'file:///oncourt_core_for_kg.csv' AS row
// WITH row,
//      toInteger(row.player_id) AS pid,
//      toInteger(row.season)    AS yr
// WHERE pid IS NOT NULL AND yr IS NOT NULL
// MATCH (ps:PlayerSeason {player_id: pid, season: yr})
// FOREACH (_ IN CASE WHEN row.gp IS NOT NULL AND row.gp<>'' THEN [1] ELSE [] END | SET ps.gp = toInteger(row.gp))
// FOREACH (_ IN CASE WHEN row.mp IS NOT NULL AND row.mp<>'' THEN [1] ELSE [] END | SET ps.mp = toFloat(row.mp))
// FOREACH (_ IN CASE WHEN row.per IS NOT NULL AND row.per<>'' THEN [1] ELSE [] END | SET ps.per = toFloat(row.per))
// FOREACH (_ IN CASE WHEN row.ws IS NOT NULL AND row.ws<>'' THEN [1] ELSE [] END | SET ps.ws = toFloat(row.ws))
// // …按你真实存在的列继续加
