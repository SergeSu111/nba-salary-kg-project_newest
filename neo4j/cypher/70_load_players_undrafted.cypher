MERGE (u:DraftPool {label:'Undrafted'});

CALL () {
LOAD CSV WITH HEADERS FROM 'file:///offcourt_draft_undrafted_for_kg.csv' AS row
WITH row, toInteger(row.player_id) AS pid
WHERE pid IS NOT NULL
MERGE (p:Player {player_id: pid})
  ON CREATE SET p.name = row.player_name, p.createdAt = timestamp()
SET  p.undrafted_flag = true
MERGE (u:DraftPool {label:'Undrafted'})
MERGE (p)-[:UNDRAFTED]->(u)
} IN TRANSACTIONS OF 1000 ROWS;