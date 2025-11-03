//每处理一定数量的行就提交一次事务 1000行左右 避免内存爆炸
USING PERIODIC COMMIT
//WITH HEADERS 表示第一行是列名 而非数据
LOAD CSV WITH HEADERS FROM 'file:///oncourt_core_for_kg.csv' AS row
WITH toInteger(row.player_id) AS pid, row.player_name AS pname
WHERE pid IS NOT NULL
MERGE (p:Player {player_id: pid})
// 如果节点第一次被创建就执行下面的操作 记录创建时间 
ON CREATE SET p.name = coalesce(pname, p.name), p.createdAt = timestamp();
