# scripts/sql_filters.py

def movie_from_join() -> str:
    return f"""
FROM (
  SELECT *
  FROM (
    SELECT t.*,
           ROW_NUMBER() OVER (
             PARTITION BY LOWER(t.primaryTitle)
             ORDER BY t.numVotes DESC, t.tconst
           ) AS _rn
    FROM titles t
    WHERE {movie_where_clause()}
      AND EXISTS (SELECT 1 FROM title_genres gg WHERE gg.tconst = t.tconst)
  ) d
  WHERE d._rn = 1
) t
INNER JOIN title_genres g ON g.tconst = t.tconst
"""

def movie_where_clause() -> str:
    return (
        "t.startYear IS NOT NULL "
        "AND t.startYear >= 1850 "
        "AND t.averageRating IS NOT NULL "
        "AND t.runtimeMinutes IS NOT NULL "
        "AND t.runtimeMinutes >= 5 "
        "AND t.titleType IN ('movie','tvSeries','tvMovie','tvMiniSeries') "
        "AND t.numVotes >= 10"
    )

def movie_group_by() -> str:
    return "GROUP BY t.tconst"

def movie_having() -> str:
    return "HAVING COUNT(g.genre) > 0"

def people_from_join() -> str:
    return "FROM people p LEFT JOIN people_professions pp ON pp.nconst = p.nconst"

def people_where_clause() -> str:
    return "p.birthYear IS NOT NULL"

def people_group_by() -> str:
    return "GROUP BY p.nconst"

def people_having() -> str:
    return "HAVING COUNT(pp.profession) > 0"
