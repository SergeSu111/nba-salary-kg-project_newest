import pandas as pd

CANON_MAP = {
    # exact or startswith keys -> (canonical, team)
    'MVP': ('Most Valuable Player', None),
    'Finals MVP': ('Finals MVP', None),
    'Defensive Player of the Year': ('Defensive Player of the Year', None),
    'Rookie of the Year': ('Rookie of the Year', None),
    'Sixth Man of the Year': ('Sixth Man of the Year', None),
    'Most Improved Player': ('Most Improved Player', None),
}

TEAM_PATTERNS = [
    ('All-NBA',       ['All-NBA 1st Team', 'All-NBA First Team'], '1st'),
    ('All-NBA',       ['All-NBA 2nd Team', 'All-NBA Second Team'], '2nd'),
    ('All-NBA',       ['All-NBA 3rd Team', 'All-NBA Third Team'], '3rd'),
    ('All-Defensive', ['All-Defensive 1st Team','All-Defensive First Team'], '1st'),
    # ...同理补 second/third，如有
]

df = pd.read_csv('data/raw_external/award.csv')

# 1) 预清洗
df = df.rename(columns={'Player_id':'player_id','Player':'player','Award':'award','Year':'year'})
df['award'] = df['award'].astype(str).str.strip()
df['team']  = None

# 2) 映射 team 类奖项
for canon, variants, team in TEAM_PATTERNS:
    mask = df['award'].isin(variants)
    df.loc[mask, 'award'] = canon
    df.loc[mask, 'team']  = team

# 3) 其他奖项 canonical
for k,(canon,team) in CANON_MAP.items():
    mask = df['award'].eq(k)
    df.loc[mask,'award'] = canon
    if team: df.loc[mask,'team'] = team

# 4) 过滤年份范围、空值
df = df[df['year'].between(2020, 2024)]
df = df.dropna(subset=['player_id','award','year'])

# 5) 去重
df = df.drop_duplicates(subset=['player_id','award','year','team']).reset_index(drop=True)

# 6) 导出
df[['player_id','award','year','team']].to_csv('award_std.csv', index=False)