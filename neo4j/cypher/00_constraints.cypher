// player
CREATE CONSTRAINT player_id_unique IF NOT EXISTS
FOR (p:Player) REQUIRE p.player_id IS UNIQUE;

CREATE CONSTRAINT player_id_exists IF NOT EXISTS
FOR (p:Player) REQUIRE p.player_id IS NOT NULL;

// 常用查询（可选）
CREATE INDEX player_name_idx IF NOT EXISTS
FOR (p:Player) ON (p.name);

// ---------- Team ----------
CREATE CONSTRAINT team_id_unique IF NOT EXISTS
FOR (t:Team) REQUIRE t.team_id IS UNIQUE;

CREATE CONSTRAINT team_id_exists IF NOT EXISTS
FOR (t:Team) REQUIRE t.team_id IS NOT NULL;

// 常用查询（可选）
CREATE INDEX team_abbr_idx IF NOT EXISTS
FOR (t:Team) ON (t.team_abbr);

// ---------- Season ----------
CREATE CONSTRAINT season_year_unique IF NOT EXISTS
FOR (s:Season) REQUIRE s.year IS UNIQUE;

CREATE CONSTRAINT season_year_exists IF NOT EXISTS
FOR (s:Season) REQUIRE s.year IS NOT NULL;

// ---------- PlayerSeason ----------
// 语义：同一球员在同一赛季仅有一条 PlayerSeason 节点
CREATE CONSTRAINT playerseason_pk_unique IF NOT EXISTS
FOR (ps:PlayerSeason) REQUIRE (ps.player_id, ps.season) IS UNIQUE;

CREATE CONSTRAINT playerseason_player_exists IF NOT EXISTS
FOR (ps:PlayerSeason) REQUIRE ps.player_id IS NOT NULL;

CREATE CONSTRAINT playerseason_season_exists IF NOT EXISTS
FOR (ps:PlayerSeason) REQUIRE ps.season IS NOT NULL;

// 常用查询（可选）
CREATE INDEX playerseason_player_idx IF NOT EXISTS
FOR (ps:PlayerSeason) ON (ps.player_id);

CREATE INDEX playerseason_season_idx IF NOT EXISTS
FOR (ps:PlayerSeason) ON (ps.season);

// ---------- TeamValue ----------
// 语义：同一支球队在同一年仅有一条 TeamValue 记录
CREATE CONSTRAINT teamvalue_pk_unique IF NOT EXISTS
FOR (tv:TeamValue) REQUIRE (tv.team_id, tv.year) IS UNIQUE;

CREATE CONSTRAINT teamvalue_team_exists IF NOT EXISTS
FOR (tv:TeamValue) REQUIRE tv.team_id IS NOT NULL;

CREATE CONSTRAINT teamvalue_year_exists IF NOT EXISTS
FOR (tv:TeamValue) REQUIRE tv.year IS NOT NULL;

// ---------- DraftPool（可选，无强约束） ----------
// 如需：保证命名不为空
// CREATE CONSTRAINT draftpool_label_exists IF NOT EXISTS
// FOR (d:DraftPool) REQUIRE d.label IS NOT NULL;