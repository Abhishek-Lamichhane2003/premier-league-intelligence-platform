from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from .features import (
    FEATURE_COLUMNS,
    build_ml_dataset,
    build_prediction_row,
)


def artifacts_dir():

    path = (
        Path(__file__)
        .resolve()
        .parents[1]
        /
        "artifacts"
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def train_and_compare(matches):

    data = build_ml_dataset(
        matches
    )

    data = (
        data
        .sort_values("Date")
        .reset_index(drop=True)
    )


    # ---------------------------------------------
    # TIME-BASED TRAIN / TEST SPLIT
    # ---------------------------------------------

    cutoff = int(
        len(data) * 0.8
    )


    train = (
        data
        .iloc[:cutoff]
        .copy()
    )


    test = (
        data
        .iloc[cutoff:]
        .copy()
    )


    numeric = FEATURE_COLUMNS


    categorical = [
        "HomeTeam",
        "AwayTeam",
    ]


    features = (
        numeric
        +
        categorical
    )


    # ---------------------------------------------
    # DATA PREPROCESSING
    # ---------------------------------------------

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                StandardScaler(),
                numeric,
            ),

            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical,
            ),
        ]
    )


    # ---------------------------------------------
    # MODELS
    # ---------------------------------------------

    models = {

        "logistic_regression":
            LogisticRegression(
                max_iter=1000
            ),


        "random_forest":
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                min_samples_leaf=3,
                class_weight=
                    "balanced_subsample",
            ),
    }


    results = {}

    trained = {}


    labels = [
        "H",
        "D",
        "A",
    ]


    # ---------------------------------------------
    # TRAIN BOTH MODELS
    # ---------------------------------------------

    for name, model in models.items():

        pipeline = Pipeline(
            [
                (
                    "prep",
                    preprocessor,
                ),

                (
                    "model",
                    model,
                ),
            ]
        )


        pipeline.fit(
            train[features],
            train["FTR"],
        )


        predictions = (
            pipeline.predict(
                test[features]
            )
        )


        results[name] = {

            "accuracy":
                float(
                    accuracy_score(
                        test["FTR"],
                        predictions,
                    )
                ),


            "confusion_matrix":
                confusion_matrix(
                    test["FTR"],
                    predictions,
                    labels=labels,
                ).tolist(),


            "classification_report":
                classification_report(
                    test["FTR"],
                    predictions,
                    labels=labels,
                    output_dict=True,
                    zero_division=0,
                ),


            "test_rows":
                int(
                    len(test)
                ),
        }


        trained[name] = (
            pipeline
        )


    # ---------------------------------------------
    # SIMPLE BASELINE
    # ---------------------------------------------

    baseline_class = (
        train["FTR"]
        .mode()
        .iloc[0]
    )


    baseline_predictions = (
        pd.Series(
            baseline_class,
            index=test.index,
        )
    )


    baseline_accuracy = float(
        accuracy_score(
            test["FTR"],
            baseline_predictions,
        )
    )


    # ---------------------------------------------
    # SELECT BEST MODEL
    # ---------------------------------------------

    best_name = max(
        results,
        key=lambda name:
        results[name]["accuracy"],
    )


    best_model = (
        trained[best_name]
    )


    # ---------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------

    joblib.dump(
        best_model,
        artifacts_dir()
        /
        "match_model.joblib",
    )


    metadata = {

        "best_model":
            best_name,


        "features":
            features,


        "results":
            results,


        "baseline_class":
            baseline_class,


        "baseline_accuracy":
            baseline_accuracy,


        "train_rows":
            int(
                len(train)
            ),


        "test_rows":
            int(
                len(test)
            ),


        "test_start_date":
            str(
                test["Date"]
                .min()
                .date()
            ),


        "test_end_date":
            str(
                test["Date"]
                .max()
                .date()
            ),
    }


    metrics_file = (
        artifacts_dir()
        /
        "model_metrics.json"
    )


    metrics_file.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )


    return metadata


def load_model_and_metrics():

    model_path = (
        artifacts_dir()
        /
        "match_model.joblib"
    )


    metrics_path = (
        artifacts_dir()
        /
        "model_metrics.json"
    )


    if (
        not model_path.exists()
        or
        not metrics_path.exists()
    ):

        return None, None


    model = joblib.load(
        model_path
    )


    metrics = json.loads(
        metrics_path.read_text()
    )


    return (
        model,
        metrics,
    )


def predict_match(
    model,
    matches,
    home_team,
    away_team,
    window=5,
):

    row, home_summary, away_summary = (
        build_prediction_row(
            matches,
            home_team,
            away_team,
            window,
        )
    )


    features = (
        FEATURE_COLUMNS
        +
        [
            "HomeTeam",
            "AwayTeam",
        ]
    )


    predicted_class = (
        model.predict(
            row[features]
        )[0]
    )


    probabilities = (
        model.predict_proba(
            row[features]
        )[0]
    )


    classes = model.classes_


    probability_map = {

        label:
            float(probability)

        for label, probability
        in zip(
            classes,
            probabilities,
        )
    }


    return {

        "predicted_class":
            predicted_class,


        "probabilities":
            probability_map,


        "home_summary":
            home_summary,


        "away_summary":
            away_summary,


        "feature_row":
            row,
    }
