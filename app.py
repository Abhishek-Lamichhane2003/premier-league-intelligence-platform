import base64
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data import load_all_seasons

from src.analytics import (
    league_table,
    team_match_log,
    team_summary,
    season_team_summary,
)

from src.model import (
    load_model_and_metrics,
    predict_match,
)


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Premier League Intelligence Platform",
    page_icon="⚽",
    layout="wide",
)


# =========================================================
# BACKGROUND IMAGE
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

BACKGROUND_FILE = (
    BASE_DIR
    / "assets"
    / "stadium_bg.png"
)


def get_background_image():

    if not BACKGROUND_FILE.exists():
        return None

    image_bytes = (
        BACKGROUND_FILE.read_bytes()
    )

    return base64.b64encode(
        image_bytes
    ).decode()


background_image = (
    get_background_image()
)


if background_image:

    background_style = f"""
    background-image:
        linear-gradient(
            rgba(1, 10, 8, 0.82),
            rgba(1, 10, 8, 0.94)
        ),
        url(
            "data:image/png;base64,{background_image}"
        );

    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    """

else:

    background_style = """
    background:
        linear-gradient(
            135deg,
            #02110c,
            #062319,
            #010906
        );
    """


# =========================================================
# CSS / VISUAL DESIGN
# =========================================================

css = """
<style>


/* MAIN BACKGROUND */

[data-testid="stAppViewContainer"] {
    __BACKGROUND__
}


/* STREAMLIT HEADER */

[data-testid="stHeader"] {
    background: transparent;
}


/* MAIN PAGE WIDTH */

.block-container {

    max-width: 1450px;

    padding-top: 2rem;

    padding-bottom: 4rem;
}


/* SIDEBAR */

[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            rgba(1, 20, 14, 0.97),
            rgba(2, 40, 27, 0.97)
        );

    border-right:
        1px solid
        rgba(74, 222, 128, 0.30);
}


/* SIDEBAR LABELS */

[data-testid="stSidebar"] label {

    color: #ecfdf5;

    font-weight: 600;
}


/* METRIC CARDS */

[data-testid="stMetric"] {

    background:
        linear-gradient(
            145deg,
            rgba(4, 47, 31, 0.93),
            rgba(2, 25, 18, 0.91)
        );

    border:
        1px solid
        rgba(74, 222, 128, 0.35);

    border-radius: 16px;

    padding: 20px;

    box-shadow:
        0 8px 25px
        rgba(0, 0, 0, 0.28);

    backdrop-filter:
        blur(10px);
}


/* METRIC LABEL */

[data-testid="stMetricLabel"] {

    color: #bbf7d0;

    font-weight: 600;
}


/* METRIC VALUE */

[data-testid="stMetricValue"] {

    color: white;

    font-weight: 800;
}


/* HEADINGS */

h1 {

    color: white;

    font-weight: 800;
}


h2,
h3 {

    color: #f0fdf4;
}


/* DROPDOWN */

[data-baseweb="select"] > div {

    background:
        rgba(4, 25, 18, 0.95);

    border:
        1px solid
        rgba(74, 222, 128, 0.30);

    border-radius: 10px;
}


/* TABLE */

[data-testid="stDataFrame"] {

    border:
        1px solid
        rgba(74, 222, 128, 0.24);

    border-radius: 12px;

    overflow: hidden;
}


/* ALERTS */

[data-testid="stAlert"] {

    border-radius: 12px;
}


/* DIVIDER */

hr {

    border-color:
        rgba(74, 222, 128, 0.20);
}


/* BUTTON */

.stButton > button {

    border-radius: 10px;

    border:
        1px solid
        rgba(74, 222, 128, 0.60);

    font-weight: 600;
}


</style>
"""


css = css.replace(
    "__BACKGROUND__",
    background_style,
)


st.markdown(
    css,
    unsafe_allow_html=True,
)


# =========================================================
# CHART STYLE
# =========================================================

def style_chart(fig):

    fig.update_layout(

        template="plotly_dark",

        paper_bgcolor=
            "rgba(0,0,0,0)",

        plot_bgcolor=
            "rgba(1,15,11,0.72)",

        font=dict(
            color="#e5e7eb"
        ),

        title_font=dict(
            color="#f0fdf4",
            size=20,
        ),

        xaxis=dict(
            gridcolor=
                "rgba(148,163,184,0.15)"
        ),

        yaxis=dict(
            gridcolor=
                "rgba(148,163,184,0.15)"
        ),

        margin=dict(
            l=40,
            r=25,
            t=60,
            b=40,
        ),
    )

    return fig


# =========================================================
# HEADER
# =========================================================

st.title(
    "⚽ Premier League Intelligence Platform"
)


st.caption(
    "Five seasons of Premier League analytics "
    "and machine-learning match analysis"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data(show_spinner=True)
def get_matches():

    return load_all_seasons()


try:

    matches = get_matches()

except Exception as error:

    st.error(
        "The Premier League dataset "
        "could not be loaded."
    )

    st.exception(
        error
    )

    st.stop()


seasons = sorted(
    matches[
        "Season"
    ]
    .dropna()
    .unique()
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "⚽ Football Intelligence"
)


st.sidebar.caption(
    "Premier League Analytics"
)


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "👥 Team Explorer",
        "🤖 ML Model",
    ],
)


# ---------------------------------------------------------
# HIDE SEASON FILTER ON ML PAGE
# ---------------------------------------------------------

if page == "🤖 ML Model":

    season = "All Seasons"

    latest_season = sorted(
        matches[
            "Season"
        ]
        .dropna()
        .unique()
    )[-1]


    st.sidebar.caption(
        f"Prediction teams: {latest_season}"
    )


    st.sidebar.caption(
        "Predictions use each team's "
        "latest 5 completed matches."
    )


else:

    season = st.sidebar.selectbox(
        "Season",
        ["All Seasons"] + seasons,
    )


# =========================================================
# FILTER DATA
# =========================================================

if season == "All Seasons":

    season_matches = (
        matches.copy()
    )

else:

    season_matches = (
        matches[
            matches[
                "Season"
            ]
            ==
            season
        ]
        .copy()
    )


available_teams = sorted(

    set(
        season_matches[
            "HomeTeam"
        ]
        .dropna()
    )

    .union(

        set(
            season_matches[
                "AwayTeam"
            ]
            .dropna()
        )

    )

)


# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "🏠 Overview":

    st.subheader(
        "📊 League Overview"
    )


    total_matches = len(
        season_matches
    )


    total_goals = int(

        season_matches[
            "FTHG"
        ].sum()

        +

        season_matches[
            "FTAG"
        ].sum()
    )


    goals_per_match = (

        total_goals
        /
        total_matches

        if total_matches

        else 0
    )


    home_win = (

        season_matches[
            "FTR"
        ]
        .eq("H")
        .mean()

        *
        100
    )


    draw = (

        season_matches[
            "FTR"
        ]
        .eq("D")
        .mean()

        *
        100
    )


    away_win = (

        season_matches[
            "FTR"
        ]
        .eq("A")
        .mean()

        *
        100
    )


    # -----------------------------------------------------
    # TOP METRICS
    # -----------------------------------------------------

    col1, col2, col3 = (
        st.columns(3)
    )


    col1.metric(
        "⚽ Matches",
        f"{total_matches:,}",
    )


    col2.metric(
        "🥅 Goals",
        f"{total_goals:,}",
    )


    col3.metric(
        "📈 Goals / Match",
        f"{goals_per_match:.2f}",
    )


    col4, col5, col6 = (
        st.columns(3)
    )


    col4.metric(
        "🏠 Home Win",
        f"{home_win:.1f}%",
    )


    col5.metric(
        "🤝 Draw",
        f"{draw:.1f}%",
    )


    col6.metric(
        "✈️ Away Win",
        f"{away_win:.1f}%",
    )


    st.divider()


    # =====================================================
    # SPECIFIC SEASON
    # =====================================================

    if season != "All Seasons":

        st.subheader(
            f"🏆 {season} League Table"
        )


        table = league_table(
            season_matches
        )


        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
        )


        st.subheader(
            "⚽ Goals Scored by Club"
        )


        fig = px.bar(

            table.sort_values(
                "GF",
                ascending=False,
            ),

            x="Team",

            y="GF",

            labels={
                "GF":
                    "Goals Scored",

                "Team":
                    "Club",
            },
        )


        fig.update_traces(
            marker_color=
                "#4ade80"
        )


        fig.update_layout(
            xaxis_tickangle=-45
        )


        fig = style_chart(
            fig
        )


        st.plotly_chart(
            fig,
            width="stretch",
        )


    # =====================================================
    # ALL FIVE SEASONS
    # =====================================================

    else:

        st.subheader(
            "📈 Five-Season Overview"
        )


        by_season = (

            matches.assign(

                TotalGoals=
                    matches["FTHG"]
                    +
                    matches["FTAG"]

            )

            .groupby(
                "Season",
                as_index=False,
            )

            .agg(

                Matches=(
                    "FTR",
                    "size",
                ),

                Goals=(
                    "TotalGoals",
                    "sum",
                ),

            )

        )


        by_season[
            "Goals / Match"
        ] = (

            by_season[
                "Goals"
            ]

            /

            by_season[
                "Matches"
            ]

        ).round(2)


        st.dataframe(
            by_season,
            hide_index=True,
            width="stretch",
        )


        # -------------------------------------------------
        # RESULT TREND CALCULATIONS
        # -------------------------------------------------

        result_trends = (

            matches

            .groupby(
                "Season"
            )

            .agg(

                Matches=(
                    "FTR",
                    "size",
                ),

                HomeWins=(
                    "FTR",
                    lambda values:
                        (
                            values
                            ==
                            "H"
                        ).sum(),
                ),

                Draws=(
                    "FTR",
                    lambda values:
                        (
                            values
                            ==
                            "D"
                        ).sum(),
                ),

                AwayWins=(
                    "FTR",
                    lambda values:
                        (
                            values
                            ==
                            "A"
                        ).sum(),
                ),

            )

            .reset_index()

        )


        result_trends[
            "Home Win %"
        ] = (

            result_trends[
                "HomeWins"
            ]

            /

            result_trends[
                "Matches"
            ]

            *
            100
        )


        result_trends[
            "Draw %"
        ] = (

            result_trends[
                "Draws"
            ]

            /

            result_trends[
                "Matches"
            ]

            *
            100
        )


        result_trends[
            "Away Win %"
        ] = (

            result_trends[
                "AwayWins"
            ]

            /

            result_trends[
                "Matches"
            ]

            *
            100
        )


        result_long = (
            result_trends.melt(

                id_vars=
                    "Season",

                value_vars=[
                    "Home Win %",
                    "Draw %",
                    "Away Win %",
                ],

                var_name=
                    "Result",

                value_name=
                    "Percentage",
            )
        )


        left, right = (
            st.columns(2)
        )


        # -------------------------------------------------
        # GOALS BY SEASON
        # -------------------------------------------------

        with left:

            fig = px.line(

                by_season,

                x="Season",

                y="Goals",

                markers=True,

                title=
                    "⚽ Goals by Season",
            )


            fig.update_traces(

                line=dict(
                    width=3,
                    color="#4ade80",
                ),

                marker=dict(
                    size=9,
                    color="#4ade80",
                ),

            )


            fig = style_chart(
                fig
            )


            st.plotly_chart(
                fig,
                width="stretch",
            )


        # -------------------------------------------------
        # RESULT TRENDS
        # -------------------------------------------------

        with right:

            fig = px.line(

                result_long,

                x="Season",

                y="Percentage",

                color="Result",

                markers=True,

                title=
                    "📊 Match Result Trends",

                color_discrete_map={

                    "Home Win %":
                        "#4ade80",

                    "Draw %":
                        "#facc15",

                    "Away Win %":
                        "#60a5fa",

                },

            )


            fig.update_traces(

                line=dict(
                    width=3
                ),

                marker=dict(
                    size=8
                ),

            )


            fig.update_yaxes(
                ticksuffix="%"
            )


            fig = style_chart(
                fig
            )


            st.plotly_chart(
                fig,
                width="stretch",
            )


# =========================================================
# TEAM EXPLORER
# =========================================================

elif page == "👥 Team Explorer":

    team = (
        st.sidebar.selectbox(
            "Team",
            available_teams,
        )
    )


    st.subheader(
        f"⚽ {team}"
    )


    log = team_match_log(
        season_matches,
        team,
    )


    if log.empty:

        st.warning(
            "No matches were found "
            "for this team."
        )

        st.stop()


    summary = team_summary(
        log
    )


    # -----------------------------------------------------
    # TEAM METRICS
    # -----------------------------------------------------

    col1, col2, col3 = (
        st.columns(3)
    )


    col1.metric(
        "🎮 Played",
        summary["played"],
    )


    col2.metric(
        "✅ Wins",
        summary["wins"],
    )


    col3.metric(
        "🏆 Points",
        summary["points"],
    )


    col4, col5 = (
        st.columns(2)
    )


    col4.metric(
        "⚖️ Goal Difference",
        summary["gd"],
    )


    col5.metric(
        "📈 Win Rate",
        f"{summary['win_rate']:.1f}%",
    )


    st.divider()


    # -----------------------------------------------------
    # FIVE SEASON TREND
    # -----------------------------------------------------

    if season == "All Seasons":

        st.subheader(
            "📈 Five-Season Performance"
        )


        trend = (
            season_team_summary(
                matches
            )
        )


        trend = (
            trend[
                trend["Team"]
                ==
                team
            ]
            .copy()
        )


        if not trend.empty:

            fig = px.line(

                trend,

                x="Season",

                y="Points",

                markers=True,

                title=
                    f"{team}: Points by Season",
            )


            fig.update_traces(

                line=dict(
                    width=3,
                    color="#4ade80",
                ),

                marker=dict(
                    size=10,
                    color="#4ade80",
                ),

            )


            fig = style_chart(
                fig
            )


            st.plotly_chart(
                fig,
                width="stretch",
            )


    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    st.subheader(
        "⚽ Results"
    )


    result_counts = (

        log[
            "Result"
        ]

        .value_counts()

        .rename_axis(
            "Result"
        )

        .reset_index(
            name="Matches"
        )

    )


    fig = px.bar(

        result_counts,

        x="Result",

        y="Matches",

        color="Result",

        color_discrete_map={

            "Win":
                "#4ade80",

            "Draw":
                "#facc15",

            "Loss":
                "#f87171",

        },

    )


    fig = style_chart(
        fig
    )


    st.plotly_chart(
        fig,
        width="stretch",
    )


    # -----------------------------------------------------
    # MATCH LOG
    # -----------------------------------------------------

    st.subheader(
        "📋 Match Log"
    )

    display_log = log.sort_values(
        "Date",
        ascending=False,
    ).copy()

    display_log["Date"] = (
        pd.to_datetime(
            display_log["Date"]
        )
        .dt.strftime("%b %d, %Y")
    )

    st.dataframe(
        display_log,
        hide_index=True,
        width="stretch",
    )


# =========================================================
# MACHINE LEARNING PAGE
# =========================================================

else:

    st.subheader(
        "🤖 Machine-Learning Match Predictor"
    )


    st.caption(
        "Model evaluation and interactive "
        "Premier League match prediction"
    )


    model, metrics = (
        load_model_and_metrics()
    )


    if model is None or metrics is None:

        st.warning(
            "The model has not been trained yet."
        )


        st.code(
            "python3 scripts/train_model.py"
        )


        st.stop()


    # =====================================================
    # MODEL SUMMARY
    # =====================================================

    best_model = (
        metrics[
            "best_model"
        ]
    )


    best_accuracy = (

        metrics[
            "results"
        ]

        [
            best_model
        ]

        [
            "accuracy"
        ]

    )


    baseline_accuracy = (
        metrics.get(
            "baseline_accuracy",
            0,
        )
    )


    improvement = (
        best_accuracy
        -
        baseline_accuracy
    )


    col1, col2, col3 = (
        st.columns(3)
    )


    col1.metric(

        "🏆 Best Model",

        best_model
        .replace(
            "_",
            " ",
        )
        .title(),

    )


    col2.metric(

        "🎯 Test Accuracy",

        f"{best_accuracy:.1%}",

        delta=
            f"{improvement:.1%} vs baseline",

    )


    col3.metric(

        "📏 Baseline Accuracy",

        f"{baseline_accuracy:.1%}",

    )


    st.caption(
        "The baseline is a simple predictor "
        "that always chooses the most common "
        "match outcome."
    )


    st.divider()


    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    st.subheader(
        "📊 Model Comparison"
    )


    rows = []


    for name, result in (
        metrics[
            "results"
        ].items()
    ):

        rows.append(

            {

                "Model":
                    name
                    .replace(
                        "_",
                        " ",
                    )
                    .title(),

                "Accuracy":
                    result[
                        "accuracy"
                    ],

            }

        )


    rows.append(

        {

            "Model":
                "Simple Baseline",

            "Accuracy":
                baseline_accuracy,

        }

    )


    score_df = (
        pd.DataFrame(
            rows
        )
    )


    fig = px.bar(

        score_df,

        x="Model",

        y="Accuracy",

        text_auto=".1%",

        title=
            "Accuracy Comparison",

    )


    fig.update_traces(
        marker_color=
            "#4ade80"
    )


    fig.update_yaxes(
        tickformat=".0%"
    )


    fig = style_chart(
        fig
    )


    st.plotly_chart(
        fig,
        width="stretch",
    )


    st.divider()


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.subheader(
        "🧩 Confusion Matrix"
    )


    confusion = (

        metrics[
            "results"
        ]

        [
            best_model
        ]

        [
            "confusion_matrix"
        ]

    )


    confusion_df = (
        pd.DataFrame(

            confusion,

            index=[

                "Actual Home Win",

                "Actual Draw",

                "Actual Away Win",

            ],

            columns=[

                "Predicted Home Win",

                "Predicted Draw",

                "Predicted Away Win",

            ],

        )
    )


    fig = px.imshow(

        confusion_df,

        text_auto=True,

        aspect="auto",

        color_continuous_scale=
            "Greens",

        title=
            "Actual vs Predicted Results",

        labels={

            "x":
                "Model Prediction",

            "y":
                "Actual Result",

            "color":
                "Matches",

        },

    )


    fig = style_chart(
        fig
    )


    st.plotly_chart(
        fig,
        width="stretch",
    )


    st.caption(
        "Diagonal cells are correct predictions. "
        "Other cells show prediction mistakes."
    )


    st.info(
        "The confusion matrix shows that some "
        "match outcomes are harder to predict. "
        "Draws are particularly difficult "
        "for this model."
    )


    st.divider()


    # =====================================================
    # INTERACTIVE MATCH PREDICTOR
    # =====================================================

    st.subheader(
        "⚽ Predict a Match"
    )


    latest_season = sorted(

        matches[
            "Season"
        ]
        .dropna()
        .unique()

    )[-1]


    latest_matches = (

        matches[
            matches[
                "Season"
            ]
            ==
            latest_season
        ]

    )


    prediction_teams = sorted(

        set(
            latest_matches[
                "HomeTeam"
            ]
            .dropna()
        )

        .union(

            set(
                latest_matches[
                    "AwayTeam"
                ]
                .dropna()
            )

        )

    )


    st.caption(
        f"Available teams: {latest_season}. "
        "Predictions use recent historical form."
    )


    left, right = (
        st.columns(2)
    )


    with left:

        home_team = (
            st.selectbox(

                "Home Team",

                prediction_teams,

                key=
                    "prediction_home",

            )
        )


    with right:

        away_options = [

            team

            for team
            in prediction_teams

            if team
            != home_team

        ]


        away_team = (
            st.selectbox(

                "Away Team",

                away_options,

                key=
                    "prediction_away",

            )
        )


    if st.button(
        "🔮 Predict Match",
        type="primary",
    ):

        try:

            prediction = (
                predict_match(

                    model,

                    matches,

                    home_team,

                    away_team,

                )
            )


            probabilities = (
                prediction[
                    "probabilities"
                ]
            )


            home_probability = (
                probabilities.get(
                    "H",
                    0,
                )
            )


            draw_probability = (
                probabilities.get(
                    "D",
                    0,
                )
            )


            away_probability = (
                probabilities.get(
                    "A",
                    0,
                )
            )


            result_names = {

                "H":
                    f"{home_team} Win",

                "D":
                    "Draw",

                "A":
                    f"{away_team} Win",

            }


            predicted_result = (

                result_names[

                    prediction[
                        "predicted_class"
                    ]

                ]

            )


            st.success(
                "Predicted result: "
                f"{predicted_result}"
            )


            # =============================================
            # PROBABILITY CARDS
            # =============================================

            p1, p2, p3 = (
                st.columns(3)
            )


            p1.metric(

                f"🏠 {home_team} Win",

                f"{home_probability:.1%}",

            )


            p2.metric(

                "🤝 Draw",

                f"{draw_probability:.1%}",

            )


            p3.metric(

                f"✈️ {away_team} Win",

                f"{away_probability:.1%}",

            )


            st.divider()


            # =============================================
            # RECENT FORM
            # =============================================

            st.subheader(
                "📈 Recent Form Used by the Model"
            )


            home_form = (
                prediction[
                    "home_summary"
                ]
            )


            away_form = (
                prediction[
                    "away_summary"
                ]
            )


            form_table = (
                pd.DataFrame(

                    [

                        {

                            "Team":
                                home_team,

                            "Recent Points / Match":
                                home_form[
                                    "form_points"
                                ],

                            "Goals Scored / Match":
                                home_form[
                                    "goals_for"
                                ],

                            "Goals Conceded / Match":
                                home_form[
                                    "goals_against"
                                ],

                            "Matches Used":
                                home_form[
                                    "matches_used"
                                ],

                        },


                        {

                            "Team":
                                away_team,

                            "Recent Points / Match":
                                away_form[
                                    "form_points"
                                ],

                            "Goals Scored / Match":
                                away_form[
                                    "goals_for"
                                ],

                            "Goals Conceded / Match":
                                away_form[
                                    "goals_against"
                                ],

                            "Matches Used":
                                away_form[
                                    "matches_used"
                                ],

                        },

                    ]

                )
            )


            numeric_columns = [

                "Recent Points / Match",

                "Goals Scored / Match",

                "Goals Conceded / Match",

            ]


            form_table[
                numeric_columns
            ] = (

                form_table[
                    numeric_columns
                ]

                .round(2)

            )


            st.dataframe(

                form_table,

                hide_index=True,

                width="stretch",

            )


            # =============================================
            # FORM CHART
            # =============================================

            form_chart = (

                form_table.melt(

                    id_vars=
                        "Team",

                    value_vars=[

                        "Recent Points / Match",

                        "Goals Scored / Match",

                        "Goals Conceded / Match",

                    ],

                    var_name=
                        "Metric",

                    value_name=
                        "Value",

                )

            )


            fig = px.bar(

                form_chart,

                x="Metric",

                y="Value",

                color="Team",

                barmode="group",

                title=
                    "Recent Team Form Comparison",

            )


            fig = style_chart(
                fig
            )


            st.plotly_chart(
                fig,
                width="stretch",
            )


            home_date = (
                home_form[
                    "last_match_date"
                ]
            )


            away_date = (
                away_form[
                    "last_match_date"
                ]
            )


            st.caption(
                "Prediction uses each team's "
                "five most recent completed matches."
            )


            st.caption(

                f"{home_team} latest data: "
                f"{home_date.date()} | "

                f"{away_team} latest data: "
                f"{away_date.date()}"

            )


        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


    st.divider()


    # =====================================================
    # MODEL TEST INFORMATION
    # =====================================================

    st.caption(

        f"Model test period: "
        f"{metrics['test_start_date']} "
        f"to "
        f"{metrics['test_end_date']} "
        f"— "
        f"{metrics['test_rows']} matches."

    )
