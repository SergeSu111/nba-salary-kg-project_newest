USING PERIODIC COMMIT
LOAD CSV WITH HEADERS FROM 'file:///offcourt_team_location_for_kg.csv' AS row
WITH row, toInteger(row.team_id) AS tid
WHERE tid IS NOT NULL
MERGE (t:Team {team_id: tid})
SET t.team_abbr = coalesce(t.team_abbr, row.team_abbr),
    t.city      = coalesce(t.city, row.city),
    t.state     = coalesce(t.state, row.state);