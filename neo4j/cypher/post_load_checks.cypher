// 节点/关系数
MATCH (n) RETURN labels(n) AS lbl, count(*) AS cnt ORDER BY cnt DESC;
MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS cnt ORDER BY cnt DESC;

// 球队与赛季覆盖
MATCH (t:Team) RETURN count(t) AS teams;            // 30
MATCH (s:Season) RETURN min(s.year) AS minY, max(s.year) AS maxY;

MATCH (p:Player) RETURN count(p) AS players;
MATCH (ps:PlayerSeason) RETURN count(ps) AS playerSeasons;

MATCH (p:Player)-[:DRAFTED_BY]->(t:Team) RETURN count(*) AS drafted;
MATCH (p:Player)-[:UNDRAFTED]->(:DraftPool) RETURN count(*) AS undrafted;

// 抽样看看是否连通正确
MATCH (p:Player)-[:IN_SEASON]->(ps:PlayerSeason)-[:OF_SEASON]->(s:Season)
RETURN p.player_id, p.name, s.year LIMIT 10;
