LOAD CSV WITH HEADERS FROM 'file:///training_oncourt_features_with_team.csv' AS row
MATCH (ps:PlayerSeason {
  player_id: toInteger(row.Player_id),   // 注意这里是 player_id（小写）
  season:    toInteger(row.season)
})
SET ps.team_code = row.Team;             // 比如 "DEN", "LAL"