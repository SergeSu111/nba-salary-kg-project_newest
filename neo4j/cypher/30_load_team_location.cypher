CALL () {
  LOAD CSV WITH HEADERS FROM 'file:///offcourt_team_location_for_kg.csv' AS row
  WITH row, toInteger(row.team_id) AS tid
  WHERE tid IS NOT NULL

  MERGE (t:Team {team_id: tid})
    ON CREATE SET t.createdAt = timestamp()

  // 仅当 CSV 值存在且非空时再写，避免写入空串；并做标准化
  FOREACH (_ IN CASE WHEN row.team_abbr IS NOT NULL AND trim(row.team_abbr) <> '' THEN [1] ELSE [] END |
    SET t.team_abbr = coalesce(t.team_abbr, toUpper(trim(row.team_abbr)))
  )
  FOREACH (_ IN CASE WHEN row.city IS NOT NULL AND trim(row.city) <> '' THEN [1] ELSE [] END |
    SET t.city = coalesce(t.city, trim(row.city))
  )
  FOREACH (_ IN CASE WHEN row.state IS NOT NULL AND trim(row.state) <> '' THEN [1] ELSE [] END |
    SET t.state = coalesce(t.state, trim(row.state))
  )
} IN TRANSACTIONS OF 1000 ROWS;