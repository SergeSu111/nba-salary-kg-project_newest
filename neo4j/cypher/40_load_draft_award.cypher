// 2) 从 neo4j/import/award_std_fixed.csv 导入
CALL () {
  LOAD CSV WITH HEADERS FROM 'file:///award_std_fixed.csv' AS row

  WITH
    toInteger(row.player_id) AS pid,
    toInteger(row.year)      AS yr,
    trim(row.award)          AS aname,
    CASE
      WHEN row.team IS NULL OR trim(row.team) = '' THEN NULL
      ELSE trim(row.team)
    END                      AS team
  WHERE pid IS NOT NULL AND yr IS NOT NULL AND aname IS NOT NULL

  // 玩家与赛季
  MATCH (p:Player {player_id: pid})
  MERGE (s:Season {year: yr})

  // 奖项节点
  MERGE (a:Award {name: aname})
  MERGE (a)-[:AWARDED_IN]->(s)

  // 关系去重策略：
  // - 非分队奖项（MVP、DPOY 等）：仅按 {year} 合并
  // - 分队奖项（All-NBA/All-Defensive/All-Rookie）：按 {year, team} 合并
  FOREACH (_ IN CASE WHEN team IS NULL THEN [1] ELSE [] END |
    MERGE (p)-[:WON_AWARD {year: yr}]->(a)
  )
  FOREACH (_ IN CASE WHEN team IS NULL THEN [] ELSE [1] END |
    MERGE (p)-[:WON_AWARD {year: yr, team: team}]->(a)
  )
} IN TRANSACTIONS OF 200 ROWS;