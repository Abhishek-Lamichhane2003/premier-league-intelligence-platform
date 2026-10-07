# Project 1 Testing Checklist

Use this after installing the dependencies.

## 1. Check the Python files

```bash
python3 -m py_compile app.py src/*.py scripts/train_model.py
```

No output means the syntax check passed.

## 2. Run automated tests

```bash
pytest -q
```

Expected result: 3 tests pass.

## 3. Launch the dashboard

```bash
streamlit run app.py
```

Check:
- Overview loads without a red error box.
- All Seasons shows five seasons.
- A specific season shows a league table.
- Team Explorer changes when a different team is selected.
- Match Log displays matches for the selected team.
- ML Model page says the model has not been trained yet before training.

## 4. Train the models

Stop Streamlit with Control+C, then run:

```bash
python3 scripts/train_model.py
```

The script should print both model accuracies and the name of the better model.
It should also create:

- `artifacts/match_model.joblib`
- `artifacts/model_metrics.json`

## 5. Reopen the dashboard

```bash
streamlit run app.py
```

Open the ML Model section and check that:
- The saved best model is shown.
- Logistic Regression and Random Forest accuracies are displayed.
- The time-based test period is displayed.

## 6. Final manual checks

- Try at least three different seasons.
- Try at least three different teams.
- Confirm league positions, points, and goals look reasonable.
- Confirm no chart is blank unexpectedly.
- Confirm no red Streamlit error appears.
- Save the exact model metrics before writing resume claims.
