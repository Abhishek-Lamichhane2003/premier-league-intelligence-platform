# Start Here

This project is intentionally written in a straightforward style so you can learn it file by file.

Learn it in this order:

1. `src/data.py` - where the CSV files come from and how they are cleaned.
2. `src/analytics.py` - how one match becomes home and away team rows, and how SQL builds the league table.
3. `src/features.py` - how rolling pre-match features are created without using future information.
4. `src/model.py` - how the models are trained, evaluated, compared, and saved.
5. `app.py` - how Streamlit turns the analysis into a web app.
6. `tests/test_features.py` - how a few important assumptions are checked automatically.

Before an interview, be able to explain: why the split is chronological, what leakage means, why rolling features use `shift(1)`, and what accuracy does and does not tell you.
