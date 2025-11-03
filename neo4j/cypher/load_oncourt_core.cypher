CALL (){
  LOAD CSV WITH HEADERS FROM 'file:///oncourt_core_for_kg.csv' AS row
  WITH row, toInteger(row.player_id) AS pid, toInteger(row.season) AS yr
  WHERE pid IS NOT NULL AND yr IS NOT NULL

  MERGE (p:Player {player_id: pid})
  MERGE (s:Season {year: yr})
  MERGE (ps:PlayerSeason {player_id: pid, season: yr})

  MERGE (p)-[:IN_SEASON]->(ps)
  MERGE (ps)-[:OF_SEASON]->(s)

  // ——以下同上，写属性（含百分号兼容）——
  FOREACH (_ IN CASE WHEN row.GP  <> '' THEN [1] ELSE [] END | SET ps.gp          = toInteger(row.GP) )
  FOREACH (_ IN CASE WHEN row.Min <> '' THEN [1] ELSE [] END | SET ps.minutes     = toFloat(row.Min) )
  FOREACH (_ IN CASE WHEN row.PTS <> '' THEN [1] ELSE [] END | SET ps.pts         = toFloat(row.PTS) )
  FOREACH (_ IN CASE WHEN row.FP  <> '' THEN [1] ELSE [] END | SET ps.fp          = toFloat(row.FP) )
  FOREACH (_ IN CASE WHEN row.PTS_per_min <> '' THEN [1] ELSE [] END | SET ps.pts_per_min = toFloat(row.PTS_per_min) )
  FOREACH (_ IN CASE WHEN row.FTA <> '' THEN [1] ELSE [] END | SET ps.fta        = toFloat(row.FTA) )
  FOREACH (_ IN CASE WHEN row.FTM_per_min <> '' THEN [1] ELSE [] END | SET ps.ftm_per_min = toFloat(row.FTM_per_min) )
  FOREACH (_ IN CASE WHEN row.FGA_per_min <> '' THEN [1] ELSE [] END | SET ps.fga_per_min = toFloat(row.FGA_per_min) )
  FOREACH (_ IN CASE WHEN row.`3PA_per_min` <> '' THEN [1] ELSE [] END | SET ps.tpa_per_min = toFloat(row.`3PA_per_min`) )
  FOREACH (_ IN CASE WHEN row.`FG%` <> '' THEN [1] ELSE [] END | SET ps.fg_pct = toFloat(replace(row.`FG%`,'%','')) )
  FOREACH (_ IN CASE WHEN row.`3P%` <> '' THEN [1] ELSE [] END | SET ps.tp3_pct = toFloat(replace(row.`3P%`,'%','')) )
  FOREACH (_ IN CASE WHEN row.`FT%` <> '' THEN [1] ELSE [] END | SET ps.ft_pct = toFloat(replace(row.`FT%`,'%','')) )
  FOREACH (_ IN CASE WHEN row.`TS%_calc` <> '' THEN [1] ELSE [] END | SET ps.ts_pct = toFloat(replace(row.`TS%_calc`,'%','')) )
  FOREACH (_ IN CASE WHEN row.DREB_share <> '' THEN [1] ELSE [] END | SET ps.dreb_share = toFloat(row.DREB_share) )
  FOREACH (_ IN CASE WHEN row.PF_per_min <> '' THEN [1] ELSE [] END | SET ps.pf_per_min = toFloat(row.PF_per_min) )
} IN TRANSACTIONS OF 1000 ROWS;