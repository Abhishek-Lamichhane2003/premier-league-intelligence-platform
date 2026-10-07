import pandas as pd
from src.analytics import team_rows, league_table
from src.features import build_ml_dataset


def sample_matches():
    return pd.DataFrame({
        "Season": ["2025-26", "2025-26", "2025-26"],
        "Date": pd.to_datetime(["2025-08-01", "2025-08-08", "2025-08-15"]),
        "HomeTeam": ["Alpha", "Beta", "Alpha"],
        "AwayTeam": ["Beta", "Alpha", "Beta"],
        "FTHG": [2, 0, 1],
        "FTAG": [0, 0, 1],
        "FTR": ["H", "D", "D"],
    })


def test_team_rows_has_two_rows_per_match():
    rows = team_rows(sample_matches())
    assert len(rows) == 6


def test_league_table_points():
    table = league_table(sample_matches())
    alpha = table[table["Team"] == "Alpha"].iloc[0]
    assert alpha["Points"] == 5


def test_ml_dataset_created():
    data = build_ml_dataset(sample_matches(), window=2)
    assert "FormDifference" in data.columns
    assert len(data) >= 1
