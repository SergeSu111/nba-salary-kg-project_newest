MERGE (u:DraftPool {label:'Undrafted'})
  ON CREATE SET u.createdAt = timestamp();

CALL {
  LOAD CSV WITH HEADERS FROM 'file:///offcourt_draft_undrafted_for_kg.csv' AS row
  WITH row,
       toInteger(row.player_id) AS pid,
       CASE WHEN row.year IS NOT NULL AND row.year <> '' THEN toInteger(row.year) ELSE NULL END AS yr
  WHERE pid IS NOT NULL

  MERGE (p:Player {player_id: pid})
    ON CREATE SET p.createdAt = timestamp()

  FOREACH (_ IN CASE WHEN row.player_name IS NOT NULL AND trim(row.player_name) <> '' THEN [1] ELSE [] END |
    SET p.name = coalesce(p.name, row.player_name)
  )

  // （可选）仅当未被选中时打标
  FOREACH (_ IN CASE WHEN NOT (p)-[:DRAFTED_BY]->(:Team) THEN [1] ELSE [] END |
    SET p.undrafted_flag = true
  )

  // ⬇️ 关键：在 FOREACH(更新) 之后，加 WITH 把需要的变量传下去
  WITH p, yr
  MATCH (u:DraftPool {label:'Undrafted'})
  MERGE (p)-[r:UNDRAFTED]->(u)
  FOREACH (_ IN CASE WHEN yr IS NOT NULL THEN [1] ELSE [] END |
    SET r.year = yr
  )
} IN TRANSACTIONS OF 1000 ROWS;



// 从 undrafted 文件补充
CALL (){
  LOAD CSV WITH HEADERS FROM 'file:///offcourt_draft_undrafted_for_kg.csv' AS row
  WITH toInteger(row.player_id) AS pid, row.player_name AS pname
  WHERE pid IS NOT NULL AND pname IS NOT NULL AND trim(pname) <> ''
  MATCH (p:Player {player_id: pid})
  WHERE p.name IS NULL
  SET p.name = pname
} IN TRANSACTIONS OF 1000 ROWS;