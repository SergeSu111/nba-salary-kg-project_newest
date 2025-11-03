// ========== 1) 基本计数 ==========
MATCH (n) RETURN labels(n) AS lbl, count(*) AS cnt ORDER BY cnt DESC;

CALL db.relationshipTypes();          // 关系类型清单（比全表扫描快）
MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS cnt ORDER BY cnt DESC;

// ========== 2) 覆盖度 ==========
MATCH (t:Team)   RETURN count(t) AS teams;                   // 期望 30
MATCH (s:Season) RETURN count(s) AS seasons,
                        min(s.year) AS minY, max(s.year) AS maxY;  // 期望 5, 2020–2024

MATCH (p:Player)        RETURN count(p) AS players;          // 期望 668（若已清理）
MATCH (ps:PlayerSeason) RETURN count(ps) AS playerSeasons;   // 期望 ~2082

// 每年有多少条 PlayerSeason & 唯一球员数
MATCH (:Player)-[:IN_SEASON]->(ps:PlayerSeason)-[:OF_SEASON]->(s:Season)
RETURN s.year AS year,
       count(ps) AS ps_rows,
       count(DISTINCT ps.player_id) AS players_this_year
ORDER BY year;

// ========== 3) 结构连通性 / 孤儿检查 ==========
/* PlayerSeason 必须同时连到 Player 和 Season */
MATCH (ps:PlayerSeason)
WHERE NOT ( (:Player {player_id: ps.player_id})-[:IN_SEASON]->(ps) )
   OR NOT ( (ps)-[:OF_SEASON]->(:Season {year: ps.season}) )
RETURN count(ps) AS ps_orphans;   // 期望 0

/* 有上场记录的球员数（应≈ players） */
MATCH (p:Player)-[:IN_SEASON]->(:PlayerSeason)
RETURN count(DISTINCT p) AS played_players;

// 未上场的球员（若已清理应为 0；否则列出）
MATCH (p:Player)
WHERE NOT (p)-[:IN_SEASON]->(:PlayerSeason)
RETURN count(p) AS players_without_ps;

// ========== 4) Draft / Undrafted 一致性 ==========
MATCH (p:Player)-[:DRAFTED_BY]->(:Team)
RETURN count(DISTINCT p) AS drafted_players;

MATCH (p:Player)-[:UNDRAFTED]->(:DraftPool)
RETURN count(DISTINCT p) AS undrafted_players;

/* 同时被标记为 Drafted + Undrafted 的球员（按业务决定是否允许） */
MATCH (p:Player)
WHERE (p)-[:DRAFTED_BY]->(:Team)
  AND (p)-[:UNDRAFTED]->(:DraftPool)
RETURN count(DISTINCT p) AS both_drafted_and_undrafted;

/* 既没有 drafted 也没有 undrafted 的上场球员（若存在，说明选秀数据缺失） */
MATCH (p:Player)-[:IN_SEASON]->(:PlayerSeason)
WHERE NOT (p)-[:DRAFTED_BY]->(:Team)
  AND NOT (p)-[:UNDRAFTED]->(:DraftPool)
RETURN count(DISTINCT p) AS played_without_draft_info;

// ========== 5) TeamValue 完整性（如你已导入） ==========
MATCH (tv:TeamValue) RETURN count(tv) AS teamvalue_rows;

MATCH (t:Team)-[:HAS_VALUE]->(tv:TeamValue)-[:OF_SEASON]->(s:Season)
RETURN s.year AS year, count(tv) AS tv_rows_that_link_both
ORDER BY year;

/* 每队每年仅一条 TeamValue（若 >1 代表重复） */
MATCH (tv:TeamValue)
WITH tv.team_id AS tid, tv.year AS yr, count(*) AS c
WHERE c > 1
RETURN tid, yr, c
ORDER BY yr, tid;

// ========== 6) 属性质量抽查（可按需扩展） ==========
/* 关键数值字段的空值情况 */
MATCH (ps:PlayerSeason)
RETURN
  count(*)                                        AS rows,
  count { ps.gp           IS NULL }               AS gp_nulls,
  count { ps.minutes      IS NULL }               AS minutes_nulls,
  count { ps.pts          IS NULL }               AS pts_nulls,
  count { ps.fg_pct       IS NULL }               AS fg_pct_nulls,
  count { ps.tp3_pct      IS NULL }               AS tp3_pct_nulls,
  count { ps.ts_pct       IS NULL }               AS ts_pct_nulls;

// ========== 7) 抽样预览 ==========
MATCH (p:Player)-[:IN_SEASON]->(ps:PlayerSeason)-[:OF_SEASON]->(s:Season)
RETURN p.player_id, p.name, s.year, ps.gp, ps.pts, ps.fg_pct
ORDER BY s.year DESC, ps.pts DESC
LIMIT 10;
