# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T12:38:23.329825+00:00
Scope: `new`

## Source Files

- `artifacts/btc/hourly/history.csv`
- `artifacts/btc/daily/history.csv`
- `artifacts/btc/market_hours/history.csv`
- `artifacts/btc/market_hours_daily/history.csv`
- `artifacts/consolidated/history.csv`

## Coverage Metadata

| Variation | Source File | Date Range | Rows | Validated | Missing | Failed |
| --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 128 | 68 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 164 | 104 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 183 | 92 | 91 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 183 | 92 | 91 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T01:00:00+00:00 | 72 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T01:00:00+00:00 | 72 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T01:00:00+00:00 | 72 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-25T01:00:00+00:00 | 73 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 92 | 51 | 41 | 55.43% | 55.43% | 55.43% | 5.43 pp | 10 | 8 | 1.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 72 | 39 | 33 | 54.17% | 54.17% | 54.17% | 4.17 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 72 | 39 | 33 | 54.17% | 54.17% | 54.17% | 4.17 pp | 6 | 8 | 0.75 |
| BTC Hourly | transformer | Transformer | 68 | 35 | 33 | 51.47% | 51.47% | 51.47% | 1.47 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 72 | 38 | 34 | 52.78% | 52.78% | 52.78% | 2.78 pp | 4 | 8 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 72 | 38 | 34 | 52.78% | 52.78% | 52.78% | 2.78 pp | 4 | 8 | 0.50 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 92 | 48 | 44 | 52.17% | 52.17% | 52.17% | 2.17 pp | 4 | 8 | 0.50 |
| BTC Market Hours | rf | RandomForest | 92 | 46 | 46 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Hourly | lstm | LSTM | 72 | 36 | 36 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 72 | 36 | 36 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Hourly | nn | NN | 68 | 34 | 34 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 92 | 45 | 47 | 48.91% | 48.91% | 48.91% | 1.09 pp | -2 | 8 | -0.25 |
| BTC Market Hours Daily | rf | RandomForest | 92 | 43 | 49 | 46.74% | 46.74% | 46.74% | 3.26 pp | -6 | 8 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 92 | 43 | 49 | 46.74% | 46.74% | 46.74% | 3.26 pp | -6 | 8 | -0.75 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 94 | 44 | 50 | 46.81% | 46.81% | 46.81% | 3.19 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 94 | 44 | 50 | 46.81% | 46.81% | 46.81% | 3.19 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 8 | -1.25 |
| BTC Market Hours Daily | nn | NN | 92 | 40 | 52 | 43.48% | 43.48% | 43.48% | 6.52 pp | -12 | 8 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 8 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 8 | -1.50 |
| BTC Market Hours | lstm | LSTM | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 8 | -1.75 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 68 | 31 | 37 | 45.59% | 45.59% | 45.59% | 4.41 pp | -6 | 3 | -2.00 |
| BTC Daily | transformer | Transformer | 94 | 42 | 52 | 44.68% | 44.68% | 44.68% | 5.32 pp | -10 | 5 | -2.00 |
| BTC Market Hours | transformer | Transformer | 92 | 37 | 55 | 40.22% | 40.22% | 40.22% | 9.78 pp | -18 | 8 | -2.25 |
| BTC Market Hours Daily | lstm | LSTM | 92 | 37 | 55 | 40.22% | 40.22% | 40.22% | 9.78 pp | -18 | 8 | -2.25 |
| Consolidated Hourly | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 8 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 8 | -2.25 |
| BTC Market Hours | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 8 | -2.50 |
| BTC Market Hours Daily | xgb | XGBoost | 92 | 35 | 57 | 38.04% | 38.04% | 38.04% | 11.96 pp | -22 | 8 | -2.75 |
| BTC Hourly | rf | RandomForest | 68 | 28 | 40 | 41.18% | 41.18% | 41.18% | 8.82 pp | -12 | 3 | -4.00 |
| BTC Daily | rf | RandomForest | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 5 | -4.40 |
| BTC Daily | lstm | LSTM | 94 | 34 | 60 | 36.17% | 36.17% | 36.17% | 13.83 pp | -26 | 5 | -5.20 |
| BTC Hourly | lstm | LSTM | 68 | 25 | 43 | 36.76% | 36.76% | 36.76% | 13.24 pp | -18 | 3 | -6.00 |
| BTC Daily | xgb | XGBoost | 104 | 34 | 70 | 32.69% | 32.69% | 32.69% | 17.31 pp | -36 | 6 | -6.00 |
| BTC Hourly | xgb | XGBoost | 68 | 21 | 47 | 30.88% | 30.88% | 30.88% | 19.12 pp | -26 | 3 | -8.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 68 | 35 | 33 | 51.47% | 51.47% | 51.47% | 1.47 pp | 2 | 3 | 0.67 |
| BTC Hourly | nn | NN | 68 | 34 | 34 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 68 | 31 | 37 | 45.59% | 45.59% | 45.59% | 4.41 pp | -6 | 3 | -2.00 |
| BTC Hourly | rf | RandomForest | 68 | 28 | 40 | 41.18% | 41.18% | 41.18% | 8.82 pp | -12 | 3 | -4.00 |
| BTC Hourly | lstm | LSTM | 68 | 25 | 43 | 36.76% | 36.76% | 36.76% | 13.24 pp | -18 | 3 | -6.00 |
| BTC Hourly | xgb | XGBoost | 68 | 21 | 47 | 30.88% | 30.88% | 30.88% | 19.12 pp | -26 | 3 | -8.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 94 | 44 | 50 | 46.81% | 46.81% | 46.81% | 3.19 pp | -6 | 5 | -1.20 |
| BTC Daily | nn | NN | 94 | 44 | 50 | 46.81% | 46.81% | 46.81% | 3.19 pp | -6 | 5 | -1.20 |
| BTC Daily | transformer | Transformer | 94 | 42 | 52 | 44.68% | 44.68% | 44.68% | 5.32 pp | -10 | 5 | -2.00 |
| BTC Daily | rf | RandomForest | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 5 | -4.40 |
| BTC Daily | lstm | LSTM | 94 | 34 | 60 | 36.17% | 36.17% | 36.17% | 13.83 pp | -26 | 5 | -5.20 |
| BTC Daily | xgb | XGBoost | 104 | 34 | 70 | 32.69% | 32.69% | 32.69% | 17.31 pp | -36 | 6 | -6.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 92 | 51 | 41 | 55.43% | 55.43% | 55.43% | 5.43 pp | 10 | 8 | 1.25 |
| BTC Market Hours | rf | RandomForest | 92 | 46 | 46 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 92 | 45 | 47 | 48.91% | 48.91% | 48.91% | 1.09 pp | -2 | 8 | -0.25 |
| BTC Market Hours | lstm | LSTM | 92 | 39 | 53 | 42.39% | 42.39% | 42.39% | 7.61 pp | -14 | 8 | -1.75 |
| BTC Market Hours | transformer | Transformer | 92 | 37 | 55 | 40.22% | 40.22% | 40.22% | 9.78 pp | -18 | 8 | -2.25 |
| BTC Market Hours | xgb | XGBoost | 92 | 36 | 56 | 39.13% | 39.13% | 39.13% | 10.87 pp | -20 | 8 | -2.50 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 92 | 48 | 44 | 52.17% | 52.17% | 52.17% | 2.17 pp | 4 | 8 | 0.50 |
| BTC Market Hours Daily | rf | RandomForest | 92 | 43 | 49 | 46.74% | 46.74% | 46.74% | 3.26 pp | -6 | 8 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 92 | 43 | 49 | 46.74% | 46.74% | 46.74% | 3.26 pp | -6 | 8 | -0.75 |
| BTC Market Hours Daily | nn | NN | 92 | 40 | 52 | 43.48% | 43.48% | 43.48% | 6.52 pp | -12 | 8 | -1.50 |
| BTC Market Hours Daily | lstm | LSTM | 92 | 37 | 55 | 40.22% | 40.22% | 40.22% | 9.78 pp | -18 | 8 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 92 | 35 | 57 | 38.04% | 38.04% | 38.04% | 11.96 pp | -22 | 8 | -2.75 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 72 | 39 | 33 | 54.17% | 54.17% | 54.17% | 4.17 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 72 | 38 | 34 | 52.78% | 52.78% | 52.78% | 2.78 pp | 4 | 8 | 0.50 |
| Consolidated Hourly | lstm | LSTM | 72 | 36 | 36 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Hourly | xgb | XGBoost | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 8 | -1.50 |
| Consolidated Hourly | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 8 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 72 | 39 | 33 | 54.17% | 54.17% | 54.17% | 4.17 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 72 | 38 | 34 | 52.78% | 52.78% | 52.78% | 2.78 pp | 4 | 8 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 72 | 36 | 36 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 72 | 31 | 41 | 43.06% | 43.06% | 43.06% | 6.94 pp | -10 | 8 | -1.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 72 | 30 | 42 | 41.67% | 41.67% | 41.67% | 8.33 pp | -12 | 8 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 72 | 27 | 45 | 37.50% | 37.50% | 37.50% | 12.50 pp | -18 | 8 | -2.25 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
