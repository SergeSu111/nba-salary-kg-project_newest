USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM 'file:///offcourt_draft_for_kg.csv' AS row
WITH row,
     toInteger(row.player_id) AS pid,
     toInteger(row.team_id)   AS tid,
     toInteger(row.year)      AS yr,
     toInteger(row.round)     AS rnd,
     toInteger(row.round_pick) AS rpk,
     toInteger(row.overall_pick) AS opk
WHERE pid IS NOT NULL AND tid IS NOT NULL AND yr IS NOT NULL
MERGE (p:Player {player_id: pid})
  ON CREATE SET p.name = row.player_name, p.createdAt = timestamp()
SET  p.draft_year = coalesce(p.draft_year, yr),
     p.draft_team_abbr = coalesce(p.draft_team_abbr, row.draft_team_abbr),
     p.overall_pick = coalesce(p.overall_pick, opk),
     p.round = coalesce(p.round, rnd);
MERGE (t:Team {team_id: tid})
MERGE (p)-[r:DRAFTED_BY]->(t)
  ON CREATE SET r.year = yr, r.round = rnd, r.round_pick = rpk, r.overall_pick = opk;
