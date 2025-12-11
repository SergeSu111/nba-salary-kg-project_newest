MATCH (ps:PlayerSeason)
WHERE ps.team_code IS NOT NULL
MATCH (t:Team   {team_abbr: ps.team_code})
MATCH (s:Season {year:    ps.season})
MERGE (ts:TeamSeason {
  team_abbr: t.team_abbr,
  season:    ps.season
})
SET ts.team_id = t.team_id   // 方便以后用
MERGE (ts)-[:OF_TEAM]->(t)
MERGE (ts)-[:IN_SEASON]->(s);


// 给每个球员当季连上对应的 TeamSeason
MATCH (ps:PlayerSeason)
WHERE ps.team_code IS NOT NULL
MATCH (ts:TeamSeason {
  team_abbr: ps.team_code,
  season:    ps.season
})
MERGE (ps)-[:FOR_TEAM]->(ts);
