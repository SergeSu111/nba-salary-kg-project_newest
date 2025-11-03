CALL () {
  LOAD CSV WITH HEADERS FROM 'file:///offcourt_team_value_for_kg.csv' AS row
  WITH row,
       toInteger(row.team_id) AS tid,
       toInteger(row.year)    AS yr,
       CASE 
         WHEN row.team_value_usd IS NOT NULL AND row.team_value_usd <> '' 
           THEN toFloat(row.team_value_usd)
         ELSE NULL
       END AS val
  WHERE tid IS NOT NULL AND yr IS NOT NULL

  // Team
  MERGE (t:Team {team_id: tid})
    ON CREATE SET t.createdAt = timestamp()

  // 仅当 CSV 里不为空时才更新缩写（避免把空串写进去）
  FOREACH (_ IN CASE WHEN row.team_abbr IS NOT NULL AND row.team_abbr <> '' THEN [1] ELSE [] END |
    SET t.team_abbr = coalesce(t.team_abbr, row.team_abbr)
  )

  // Season
  MERGE (s:Season {year: yr})
    ON CREATE SET s.createdAt = timestamp()

  // TeamValue（复合键）
  MERGE (tv:TeamValue {team_id: tid, year: yr})
    // 写入估值：
    // 选项 A（只首写，不覆盖历史）：ON CREATE SET tv.value = val
    // 选项 B（每次以文件为准覆盖）：SET tv.value = val
    // 按你需求选择其中一个；下面给 B：
    SET tv.value = val

  MERGE (t)-[:HAS_VALUE]->(tv)
  MERGE (t)-[:PLAYS_SEASON]->(s)
  // 建议补一条 TeamValue→Season 的关系，方便按赛季聚合
  MERGE (tv)-[:OF_SEASON]->(s)
} IN TRANSACTIONS OF 1000 ROWS;