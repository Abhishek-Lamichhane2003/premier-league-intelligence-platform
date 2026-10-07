from pathlib import Path
import pandas as pd
import requests
import certifi

BASE_URL = "https://www.football-data.co.uk/mmz4281"
SEASON_URLS = {
    "2021-22": f"{BASE_URL}/2122/E0.csv",
    "2022-23": f"{BASE_URL}/2223/E0.csv",
    "2023-24": f"{BASE_URL}/2324/E0.csv",
    "2024-25": f"{BASE_URL}/2425/E0.csv",
    "2025-26": f"{BASE_URL}/2526/E0.csv",
}

REQUIRED = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
NUMERIC = [
    "FTHG", "FTAG", "HTHG", "HTAG", "HS", "AS", "HST", "AST",
    "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"
]


def project_root():
    return Path(__file__).resolve().parents[1]


def raw_data_dir():
    path = project_root() / "data" / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_season(season, url):
    output = raw_data_dir() / f"{season}.csv"
    if output.exists():
        return output

    response = requests.get(url, timeout=30, verify=certifi.where())
    response.raise_for_status()
    output.write_bytes(response.content)
    return output


def load_all_seasons():
    frames = []

    for season, url in SEASON_URLS.items():
        path = download_season(season, url)
        df = pd.read_csv(path)

        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{season} is missing required columns: {missing}")

        df = df.copy()
        df["Season"] = season
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

        for col in NUMERIC:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=REQUIRED + ["Season"])
        frames.append(df)

    all_matches = pd.concat(frames, ignore_index=True)
    return all_matches.sort_values("Date").reset_index(drop=True)
