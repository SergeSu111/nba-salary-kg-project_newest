CALL () {
  LOAD CSV WITH HEADERS FROM 'file:///oncourt_core_for_kg.csv' AS row
  WITH row,
       toInteger(row.player_id) AS pid,
       toInteger(row.season)    AS yr
  WHERE pid IS NOT NULL AND yr IS NOT NULL
  MATCH (ps:PlayerSeason {player_id: pid, season: yr})

  // --- INT 列 ---
  FOREACH (_ IN CASE WHEN row.GP IS NOT NULL AND row.GP <> '' THEN [1] ELSE [] END |
    SET ps.gp = toInteger(row.GP)
  )

  // --- FLOAT 列 ---
  FOREACH (_ IN CASE WHEN row.Min IS NOT NULL AND row.Min <> '' THEN [1] ELSE [] END |
    SET ps.minutes = toFloat(row.Min)
  )
  FOREACH (_ IN CASE WHEN row.PTS IS NOT NULL AND row.PTS <> '' THEN [1] ELSE [] END |
    SET ps.pts = toFloat(row.PTS)
  )
  FOREACH (_ IN CASE WHEN row.FP IS NOT NULL AND row.FP <> '' THEN [1] ELSE [] END |
    SET ps.fp = toFloat(row.FP)
  )
  FOREACH (_ IN CASE WHEN row.PTS_per_min IS NOT NULL AND row.PTS_per_min <> '' THEN [1] ELSE [] END |
    SET ps.pts_per_min = toFloat(row.PTS_per_min)
  )
  FOREACH (_ IN CASE WHEN row.FTA IS NOT NULL AND row.FTA <> '' THEN [1] ELSE [] END |
    SET ps.fta = toFloat(row.FTA)
  )
  FOREACH (_ IN CASE WHEN row.FTM_per_min IS NOT NULL AND row.FTM_per_min <> '' THEN [1] ELSE [] END |
    SET ps.ftm_per_min = toFloat(row.FTM_per_min)
  )
  FOREACH (_ IN CASE WHEN row.FGA_per_min IS NOT NULL AND row.FGA_per_min <> '' THEN [1] ELSE [] END |
    SET ps.fga_per_min = toFloat(row.FGA_per_min)
  )
  // 列名以数字开头 -> 用反引号取列，用安全属性名保存
  FOREACH (_ IN CASE WHEN row.`3PA_per_min` IS NOT NULL AND row.`3PA_per_min` <> '' THEN [1] ELSE [] END |
    SET ps.tpa_per_min = toFloat(row.`3PA_per_min`)
  )
  FOREACH (_ IN CASE WHEN row.`FG%` IS NOT NULL AND row.`FG%` <> '' THEN [1] ELSE [] END |
    SET ps.fg_pct = toFloat(row.`FG%`)
  )
  FOREACH (_ IN CASE WHEN row.`3P%` IS NOT NULL AND row.`3P%` <> '' THEN [1] ELSE [] END |
    SET ps.tp3_pct = toFloat(row.`3P%`)
  )
  FOREACH (_ IN CASE WHEN row.`FT%` IS NOT NULL AND row.`FT%` <> '' THEN [1] ELSE [] END |
    SET ps.ft_pct = toFloat(row.`FT%`)
  )
  FOREACH (_ IN CASE WHEN row.`TS%_calc` IS NOT NULL AND row.`TS%_calc` <> '' THEN [1] ELSE [] END |
    SET ps.ts_pct = toFloat(row.`TS%_calc`)
  )
  FOREACH (_ IN CASE WHEN row.DREB_share IS NOT NULL AND row.DREB_share <> '' THEN [1] ELSE [] END |
    SET ps.dreb_share = toFloat(row.DREB_share)
  )
  FOREACH (_ IN CASE WHEN row.PF_per_min IS NOT NULL AND row.PF_per_min <> '' THEN [1] ELSE [] END |
    SET ps.pf_per_min = toFloat(row.PF_per_min)
  )
} IN TRANSACTIONS OF 1000 ROWS;