CALL () {
  LOAD CSV WITH HEADERS FROM 'file:///offcourt_draft_for_kg.csv' AS row
  WITH row,
       toInteger(row.player_id)   AS pid,
       toInteger(row.team_id)     AS tid,
       toInteger(row.year)        AS yr,
       CASE WHEN row.round         IS NOT NULL AND row.round <> ''         THEN toInteger(row.round)         ELSE NULL END AS rnd,
       CASE WHEN row.round_pick    IS NOT NULL AND row.round_pick <> ''    THEN toInteger(row.round_pick)    ELSE NULL END AS rpk,
       CASE WHEN row.overall_pick  IS NOT NULL AND row.overall_pick <> ''  THEN toInteger(row.overall_pick)  ELSE NULL END AS opk,
       CASE WHEN row.draft_team_abbr IS NOT NULL AND trim(row.draft_team_abbr) <> '' THEN toUpper(trim(row.draft_team_abbr)) ELSE NULL END AS abbr
  WHERE pid IS NOT NULL AND tid IS NOT NULL AND yr IS NOT NULL

  // Player
  MERGE (p:Player {player_id: pid})
    ON CREATE SET p.name = row.player_name, p.createdAt = timestamp()

  // ——字段写入策略——
  // A) 只首写（保留历史）：用 coalesce（你当前做法）
  // B) 总是覆盖（CSV为准）：用 SET（如果未来会修正数据，推荐）
  // 下面保留你的“首写不覆盖”策略，并防空串
  FOREACH (_ IN CASE WHEN yr  IS NOT NULL THEN [1] ELSE [] END | SET p.draft_year      = coalesce(p.draft_year, yr))
  FOREACH (_ IN CASE WHEN abbr IS NOT NULL THEN [1] ELSE [] END | SET p.draft_team_abbr = coalesce(p.draft_team_abbr, abbr))
  FOREACH (_ IN CASE WHEN opk  IS NOT NULL THEN [1] ELSE [] END | SET p.overall_pick    = coalesce(p.overall_pick, opk))
  FOREACH (_ IN CASE WHEN rnd  IS NOT NULL THEN [1] ELSE [] END | SET p.round           = coalesce(p.round, rnd))

  // Team
  MERGE (t:Team {team_id: tid})
  FOREACH (_ IN CASE WHEN abbr IS NOT NULL THEN [1] ELSE [] END | SET t.team_abbr = coalesce(t.team_abbr, abbr))

  // 关系（同一对端点 MERGE 不会重复）
  MERGE (p)-[r:DRAFTED_BY]->(t)
    ON CREATE SET r.year = yr, r.round = rnd, r.round_pick = rpk, r.overall_pick = opk
} IN TRANSACTIONS OF 1000 ROWS;


CALL () {
  LOAD CSV WITH HEADERS FROM 'file:///offcourt_draft_for_kg.csv' AS row
  WITH toInteger(row.player_id) AS pid, row.player_name AS pname
  WHERE pid IS NOT NULL AND pname IS NOT NULL AND trim(pname) <> ''
  MATCH (p:Player {player_id: pid})
  WHERE p.name IS NULL
  SET p.name = pname
} IN TRANSACTIONS OF 1000 ROWS;