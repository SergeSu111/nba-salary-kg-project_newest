CALL () {
  LOAD CSV WITH HEADERS FROM 'file:///award_std.csv' AS row
  WITH
    toInteger(row.player_id) AS pid,
    toInteger(row.year)      AS yr,
    trim(row.award)          AS aname,
    CASE WHEN row.team IS NULL OR trim(row.team) = '' THEN NULL ELSE trim(row.team) END AS team
  WHERE pid IS NOT NULL AND yr IS NOT NULL AND aname IS NOT NULL

  MATCH (p:Player {player_id: pid})
  MATCH (s:Season {year: yr})
  MERGE (a:Award {name: aname})

  // 关系去重靠 MERGE + 关系属性作为“键”
  MERGE (p)-[r:WON_AWARD {year: yr, team: team}]->(a)
  MERGE (a)-[:AWARDED_IN]->(s);
} IN TRANSACTIONS OF 200 ROWS;