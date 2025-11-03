CALL {
  LOAD CSV WITH HEADERS FROM 'file:///oncourt_core_for_kg.csv' AS row
  WITH toInteger(row.player_id) AS pid, row.player_name AS pname
  WHERE pid IS NOT NULL
  MERGE (p:Player {player_id: pid})
    ON CREATE SET p.name = pname,
                  p.createdAt = timestamp()
} IN TRANSACTIONS OF 1000 ROWS;