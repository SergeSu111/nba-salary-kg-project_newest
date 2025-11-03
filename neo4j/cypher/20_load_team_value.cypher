CALL() {
LOAD CSV WITH HEADERS FROM 'file:///offcourt_team_value_for_kg.csv' AS row
WITH row,
     toInteger(row.team_id) AS tid,
     toInteger(row.year)    AS yr,
     toFloat(row.team_value_usd) AS val
WHERE tid IS NOT NULL AND yr IS NOT NULL
MERGE (t:Team {team_id: tid})
  ON CREATE SET t.createdAt = timestamp()
SET  t.team_abbr = coalesce(t.team_abbr, row.team_abbr)
MERGE (s:Season {year: yr})
//复合键
MERGE (tv:TeamValue {team_id: tid, year: yr})
  ON CREATE SET tv.value = val
MERGE (t)-[:HAS_VALUE]->(tv)
MERGE (t)-[:PLAYS_SEASON]->(s)
} IN TRANSACTIONS OF 1000 ROWS;