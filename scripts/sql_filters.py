# scripts/sql_filters.py

# -------------------------------------------------------------------------
# COLUMNS & SELECTS
# -------------------------------------------------------------------------

def movie_select_clause(alias="t", genre_alias="g") -> str:
    """
    Standard columns for the Movie Autoencoder.
    Order: 
    0: tconst
    1: primaryTitle
    2: startYear
    3: endYear
    4: runtimeMinutes
    5: averageRating
    6: numVotes
    7: genres (comma separated)
    8: peopleCount (subquery) -- Renamed from principalCount
    """
    return f"""
    {alias}.tconst,
    {alias}.primaryTitle,
    {alias}.startYear,
    {alias}.endYear,
    {alias}.runtimeMinutes,
    {alias}.averageRating,
    {alias}.numVotes,
    GROUP_CONCAT({genre_alias}.genre, ','),
    (SELECT COUNT(*) FROM principals pr WHERE pr.tconst = {alias}.tconst)
    """

def people_select_clause(alias="p", prof_alias="pp") -> str:
    """
    Standard columns for the Person Autoencoder.
    Order:
    0: primaryName
    1: birthYear
    2: deathYear
    3: professions (comma separated)
    4: titleCount (subquery)
    5: nconst
    """
    return f"""
    {alias}.primaryName,
    {alias}.birthYear,
    {alias}.deathYear,
    GROUP_CONCAT({prof_alias}.profession, ','),
    (SELECT COUNT(*) FROM principals pr WHERE pr.nconst = {alias}.nconst),
    {alias}.nconst
    """

def map_movie_row(r) -> dict:
    """Maps the result of movie_select_clause to a dict."""
    if not r: return {}
    return {
        "tconst": r[0],
        "primaryTitle": r[1],
        "startYear": r[2],
        "endYear": r[3],
        "runtimeMinutes": r[4],
        "averageRating": r[5],
        "numVotes": r[6],
        "genres": r[7].split(",") if r[7] else [],
        "peopleCount": r[8],
    }

def map_person_row(r) -> dict:
    """Maps the result of people_select_clause to a dict."""
    if not r: return {}
    return {
        "primaryName": r[0],
        "birthYear": r[1],
        "deathYear": r[2],
        "professions": r[3].split(",") if r[3] else None,
        "titleCount": r[4],
        "nconst": r[5],
    }

# -------------------------------------------------------------------------
# FILTERS & JOINS
# -------------------------------------------------------------------------

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
        "AND t.runtimeMinutes >= 10 "
        "AND t.titleType IN ('movie','tvSeries','tvMovie','tvMiniSeries') "
        "AND t.numVotes >= 20"
    )

def movie_group_by() -> str:
    return "GROUP BY t.tconst"

def movie_having() -> str:
    return "HAVING COUNT(g.genre) > 0"

def people_from_join() -> str:
    return "FROM people p LEFT JOIN people_professions pp ON pp.nconst = p.nconst"

def people_where_clause() -> str:
    # Added check for birthYear >= 1800
    return "p.birthYear IS NOT NULL AND p.birthYear >= 1800"

def people_group_by() -> str:
    return "GROUP BY p.nconst"

def people_having() -> str:
    return "HAVING COUNT(pp.profession) > 0"