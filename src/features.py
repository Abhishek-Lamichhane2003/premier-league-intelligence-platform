import pandas as pd


FEATURE_COLUMNS = [
    "HomeFormPoints",
    "AwayFormPoints",
    "HomeGoalsForAvg",
    "AwayGoalsForAvg",
    "HomeGoalsAgainstAvg",
    "AwayGoalsAgainstAvg",
    "FormDifference",
    "AttackDifference",
    "DefenseDifference",
]


def _team_history(matches):
    home = pd.DataFrame({
        "Date": matches["Date"],
        "Team": matches["HomeTeam"],
        "GF": matches["FTHG"],
        "GA": matches["FTAG"],
        "Points": matches["FTR"].map({
            "H": 3,
            "D": 1,
            "A": 0,
        }),
    })

    away = pd.DataFrame({
        "Date": matches["Date"],
        "Team": matches["AwayTeam"],
        "GF": matches["FTAG"],
        "GA": matches["FTHG"],
        "Points": matches["FTR"].map({
            "H": 0,
            "D": 1,
            "A": 3,
        }),
    })

    return (
        pd.concat([home, away], ignore_index=True)
        .sort_values(["Team", "Date"])
        .reset_index(drop=True)
    )


def build_ml_dataset(matches, window=5):
    """
    Create pre-match statistics using only matches
    that happened before the match being predicted.
    """

    matches = (
        matches
        .sort_values("Date")
        .reset_index(drop=True)
        .copy()
    )

    history = _team_history(matches)

    grouped = history.groupby(
        "Team",
        group_keys=False,
    )

    history["FormPoints"] = grouped["Points"].transform(
        lambda values:
        values.shift(1)
        .rolling(
            window,
            min_periods=1,
        )
        .mean()
    )

    history["GoalsForAvg"] = grouped["GF"].transform(
        lambda values:
        values.shift(1)
        .rolling(
            window,
            min_periods=1,
        )
        .mean()
    )

    history["GoalsAgainstAvg"] = grouped["GA"].transform(
        lambda values:
        values.shift(1)
        .rolling(
            window,
            min_periods=1,
        )
        .mean()
    )

    history["MatchesSeen"] = (
        grouped.cumcount()
    )

    home_features = history.rename(
        columns={
            "Team": "HomeTeam",
            "FormPoints": "HomeFormPoints",
            "GoalsForAvg": "HomeGoalsForAvg",
            "GoalsAgainstAvg": "HomeGoalsAgainstAvg",
            "MatchesSeen": "HomeMatchesSeen",
        }
    )[
        [
            "Date",
            "HomeTeam",
            "HomeFormPoints",
            "HomeGoalsForAvg",
            "HomeGoalsAgainstAvg",
            "HomeMatchesSeen",
        ]
    ]

    away_features = history.rename(
        columns={
            "Team": "AwayTeam",
            "FormPoints": "AwayFormPoints",
            "GoalsForAvg": "AwayGoalsForAvg",
            "GoalsAgainstAvg": "AwayGoalsAgainstAvg",
            "MatchesSeen": "AwayMatchesSeen",
        }
    )[
        [
            "Date",
            "AwayTeam",
            "AwayFormPoints",
            "AwayGoalsForAvg",
            "AwayGoalsAgainstAvg",
            "AwayMatchesSeen",
        ]
    ]

    data = matches.merge(
        home_features,
        on=[
            "Date",
            "HomeTeam",
        ],
        how="left",
    )

    data = data.merge(
        away_features,
        on=[
            "Date",
            "AwayTeam",
        ],
        how="left",
    )

    rolling_columns = [
        "HomeFormPoints",
        "HomeGoalsForAvg",
        "HomeGoalsAgainstAvg",
        "AwayFormPoints",
        "AwayGoalsForAvg",
        "AwayGoalsAgainstAvg",
    ]

    for column in rolling_columns:
        data[column] = (
            data[column]
            .fillna(0.0)
        )

    data["FormDifference"] = (
        data["HomeFormPoints"]
        -
        data["AwayFormPoints"]
    )

    data["AttackDifference"] = (
        data["HomeGoalsForAvg"]
        -
        data["AwayGoalsForAvg"]
    )

    data["DefenseDifference"] = (
        data["AwayGoalsAgainstAvg"]
        -
        data["HomeGoalsAgainstAvg"]
    )

    data = data[
        (data["HomeMatchesSeen"] > 0)
        |
        (data["AwayMatchesSeen"] > 0)
    ].copy()

    return data


def latest_team_features(
    matches,
    team,
    window=5,
):
    """
    Calculate a team's recent form from its
    latest completed matches.
    """

    history = _team_history(matches)

    team_history = (
        history[
            history["Team"] == team
        ]
        .sort_values("Date")
    )

    if team_history.empty:
        raise ValueError(
            f"No historical matches found for {team}."
        )

    recent = team_history.tail(
        window
    )

    return {
        "form_points":
            float(
                recent["Points"].mean()
            ),

        "goals_for":
            float(
                recent["GF"].mean()
            ),

        "goals_against":
            float(
                recent["GA"].mean()
            ),

        "matches_used":
            int(len(recent)),

        "last_match_date":
            recent["Date"].max(),
    }


def build_prediction_row(
    matches,
    home_team,
    away_team,
    window=5,
):
    """
    Create one row of data that can be sent
    to the trained model.
    """

    if home_team == away_team:
        raise ValueError(
            "Home and away teams must be different."
        )

    home = latest_team_features(
        matches,
        home_team,
        window,
    )

    away = latest_team_features(
        matches,
        away_team,
        window,
    )

    row = {
        "HomeTeam":
            home_team,

        "AwayTeam":
            away_team,

        "HomeFormPoints":
            home["form_points"],

        "AwayFormPoints":
            away["form_points"],

        "HomeGoalsForAvg":
            home["goals_for"],

        "AwayGoalsForAvg":
            away["goals_for"],

        "HomeGoalsAgainstAvg":
            home["goals_against"],

        "AwayGoalsAgainstAvg":
            away["goals_against"],

        "FormDifference":
            home["form_points"]
            -
            away["form_points"],

        "AttackDifference":
            home["goals_for"]
            -
            away["goals_for"],

        "DefenseDifference":
            away["goals_against"]
            -
            home["goals_against"],
    }

    return (
        pd.DataFrame([row]),
        home,
        away,
    )
