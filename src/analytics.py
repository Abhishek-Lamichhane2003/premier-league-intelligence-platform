import sqlite3
import pandas as pd


def team_rows(matches):
    home = pd.DataFrame({
        "Season": matches["Season"],
        "Date": matches["Date"],
        "Team": matches["HomeTeam"],
        "Opponent": matches["AwayTeam"],
        "Venue": "Home",
        "GF": matches["FTHG"],
        "GA": matches["FTAG"],
        "RawResult": matches["FTR"],
    })

    away = pd.DataFrame({
        "Season": matches["Season"],
        "Date": matches["Date"],
        "Team": matches["AwayTeam"],
        "Opponent": matches["HomeTeam"],
        "Venue": "Away",
        "GF": matches["FTAG"],
        "GA": matches["FTHG"],
        "RawResult": matches["FTR"],
    })

    home["Result"] = home["RawResult"].map({"H": "Win", "D": "Draw", "A": "Loss"})
    away["Result"] = away["RawResult"].map({"H": "Loss", "D": "Draw", "A": "Win"})

    out = pd.concat([home, away], ignore_index=True)
    out["Points"] = out["Result"].map({"Win": 3, "Draw": 1, "Loss": 0})
    return out.drop(columns="RawResult")


def league_table(matches):
    rows = team_rows(matches)
    conn = sqlite3.connect(":memory:")
    rows.to_sql("team_matches", conn, index=False, if_exists="replace")

    query = """
        SELECT
            Team,
            COUNT(*) AS Played,
            SUM(CASE WHEN Result = 'Win' THEN 1 ELSE 0 END) AS Won,
            SUM(CASE WHEN Result = 'Draw' THEN 1 ELSE 0 END) AS Drawn,
            SUM(CASE WHEN Result = 'Loss' THEN 1 ELSE 0 END) AS Lost,
            CAST(SUM(GF) AS INTEGER) AS GF,
            CAST(SUM(GA) AS INTEGER) AS GA,
            CAST(SUM(GF) - SUM(GA) AS INTEGER) AS GD,
            CAST(SUM(Points) AS INTEGER) AS Points
        FROM team_matches
        GROUP BY Team
        ORDER BY Points DESC, GD DESC, GF DESC, Team ASC
    """

    table = pd.read_sql_query(query, conn)
    conn.close()
    table.insert(0, "Position", range(1, len(table) + 1))
    return table


def team_match_log(matches, team):
    rows = team_rows(matches)
    rows = rows[rows["Team"] == team].copy()
    rows = rows.sort_values("Date").reset_index(drop=True)
    rows["CumulativePoints"] = rows["Points"].cumsum()
    return rows


def team_summary(log):
    played = len(log)
    wins = int((log["Result"] == "Win").sum())
    draws = int((log["Result"] == "Draw").sum())
    losses = int((log["Result"] == "Loss").sum())
    gf = int(log["GF"].sum()) if played else 0
    ga = int(log["GA"].sum()) if played else 0
    points = int(log["Points"].sum()) if played else 0

    return {
        "played": played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "gf": gf,
        "ga": ga,
        "gd": gf - ga,
        "points": points,
        "points_per_match": points / played if played else 0,
        "win_rate": 100 * wins / played if played else 0,
    }


def season_team_summary(matches):
    rows = team_rows(matches)
    out = (
        rows.groupby(["Season", "Team"], as_index=False)
        .agg(
            Played=("Team", "size"),
            Wins=("Result", lambda s: (s == "Win").sum()),
            Draws=("Result", lambda s: (s == "Draw").sum()),
            Losses=("Result", lambda s: (s == "Loss").sum()),
            GF=("GF", "sum"),
            GA=("GA", "sum"),
            Points=("Points", "sum"),
        )
    )
    out["GD"] = out["GF"] - out["GA"]
    out["PPG"] = out["Points"] / out["Played"]
    return out
