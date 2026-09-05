# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T09:47:08.575225+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 225 | 165 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 260 | 200 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 357 | 188 | 169 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 00:00:00+00:00 | 357 | 188 | 169 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 14:00:00+00:00 | 161 | 161 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 14:00:00+00:00 | 161 | 161 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 14:00:00+00:00 | 161 | 43 | 118 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 14:00:00+00:00 | 161 | 43 | 118 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 188 | 99 | 89 | 52.66% | 52.66% | 52.66% | 2.66 pp | 10 | 16 | 0.62 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 165 | 84 | 81 | 50.91% | 50.91% | 50.91% | 0.91 pp | 3 | 7 | 0.43 |
| BTC Market Hours | nn | NN | 188 | 96 | 92 | 51.06% | 51.06% | 51.06% | 1.06 pp | 4 | 15 | 0.27 |
| Consolidated Hourly | rf | RandomForest | 161 | 80 | 81 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 161 | 80 | 81 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| BTC Market Hours | transformer | Transformer | 188 | 93 | 95 | 49.47% | 49.47% | 49.47% | 0.53 pp | -2 | 15 | -0.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 161 | 78 | 83 | 48.45% | 48.45% | 48.45% | 1.55 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 161 | 78 | 83 | 48.45% | 48.45% | 48.45% | 1.55 pp | -5 | 12 | -0.42 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 188 | 90 | 98 | 47.87% | 47.87% | 47.87% | 2.13 pp | -8 | 16 | -0.50 |
| BTC Hourly | transformer | Transformer | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 7 | -0.71 |
| BTC Market Hours Daily | nn | NN | 188 | 88 | 100 | 46.81% | 46.81% | 46.81% | 3.19 pp | -12 | 16 | -0.75 |
| Consolidated Market Hours | xgb | XGBoost | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 4 | -0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 188 | 87 | 101 | 46.28% | 46.28% | 46.28% | 3.72 pp | -14 | 15 | -0.93 |
| BTC Market Hours | rf | RandomForest | 188 | 87 | 101 | 46.28% | 46.28% | 46.28% | 3.72 pp | -14 | 15 | -0.93 |
| BTC Daily | mlp_sklearn | MLPClassifier | 190 | 90 | 100 | 47.37% | 47.37% | 47.37% | 2.63 pp | -10 | 9 | -1.11 |
| Consolidated Hourly | lstm | LSTM | 161 | 73 | 88 | 45.34% | 45.34% | 45.34% | 4.66 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 161 | 73 | 88 | 45.34% | 45.34% | 45.34% | 4.66 pp | -15 | 12 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 188 | 84 | 104 | 44.68% | 44.68% | 44.68% | 5.32 pp | -20 | 16 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 161 | 70 | 91 | 43.48% | 43.48% | 43.48% | 6.52 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 161 | 70 | 91 | 43.48% | 43.48% | 43.48% | 6.52 pp | -21 | 12 | -1.75 |
| Consolidated Market Hours | lstm | LSTM | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | lstm | LSTM | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 4 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 188 | 80 | 108 | 42.55% | 42.55% | 42.55% | 7.45 pp | -28 | 15 | -1.87 |
| Consolidated Hourly | transformer | Transformer | 161 | 69 | 92 | 42.86% | 42.86% | 42.86% | 7.14 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 161 | 69 | 92 | 42.86% | 42.86% | 42.86% | 7.14 pp | -23 | 12 | -1.92 |
| BTC Market Hours | lstm | LSTM | 188 | 79 | 109 | 42.02% | 42.02% | 42.02% | 7.98 pp | -30 | 15 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 188 | 78 | 110 | 41.49% | 41.49% | 41.49% | 8.51 pp | -32 | 16 | -2.00 |
| Consolidated Hourly | nn | NN | 161 | 67 | 94 | 41.61% | 41.61% | 41.61% | 8.39 pp | -27 | 12 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 161 | 67 | 94 | 41.61% | 41.61% | 41.61% | 8.39 pp | -27 | 12 | -2.25 |
| BTC Market Hours Daily | lstm | LSTM | 188 | 76 | 112 | 40.43% | 40.43% | 40.43% | 9.57 pp | -36 | 16 | -2.25 |
| BTC Daily | nn | NN | 190 | 83 | 107 | 43.68% | 43.68% | 43.68% | 6.32 pp | -24 | 9 | -2.67 |
| Consolidated Market Hours | nn | NN | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | transformer | Transformer | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | transformer | Transformer | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| BTC Daily | transformer | Transformer | 190 | 82 | 108 | 43.16% | 43.16% | 43.16% | 6.84 pp | -26 | 9 | -2.89 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 4 | -3.25 |
| BTC Hourly | nn | NN | 165 | 70 | 95 | 42.42% | 42.42% | 42.42% | 7.58 pp | -25 | 7 | -3.57 |
| BTC Hourly | rf | RandomForest | 165 | 69 | 96 | 41.82% | 41.82% | 41.82% | 8.18 pp | -27 | 7 | -3.86 |
| BTC Daily | rf | RandomForest | 190 | 73 | 117 | 38.42% | 38.42% | 38.42% | 11.58 pp | -44 | 9 | -4.89 |
| BTC Daily | xgb | XGBoost | 200 | 72 | 128 | 36.00% | 36.00% | 36.00% | 14.00 pp | -56 | 10 | -5.60 |
| BTC Hourly | lstm | LSTM | 165 | 60 | 105 | 36.36% | 36.36% | 36.36% | 13.64 pp | -45 | 7 | -6.43 |
| BTC Daily | lstm | LSTM | 190 | 66 | 124 | 34.74% | 34.74% | 34.74% | 15.26 pp | -58 | 9 | -6.44 |
| BTC Hourly | xgb | XGBoost | 165 | 58 | 107 | 35.15% | 35.15% | 35.15% | 14.85 pp | -49 | 7 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 165 | 84 | 81 | 50.91% | 50.91% | 50.91% | 0.91 pp | 3 | 7 | 0.43 |
| BTC Hourly | transformer | Transformer | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 7 | -0.71 |
| BTC Hourly | nn | NN | 165 | 70 | 95 | 42.42% | 42.42% | 42.42% | 7.58 pp | -25 | 7 | -3.57 |
| BTC Hourly | rf | RandomForest | 165 | 69 | 96 | 41.82% | 41.82% | 41.82% | 8.18 pp | -27 | 7 | -3.86 |
| BTC Hourly | lstm | LSTM | 165 | 60 | 105 | 36.36% | 36.36% | 36.36% | 13.64 pp | -45 | 7 | -6.43 |
| BTC Hourly | xgb | XGBoost | 165 | 58 | 107 | 35.15% | 35.15% | 35.15% | 14.85 pp | -49 | 7 | -7.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 190 | 90 | 100 | 47.37% | 47.37% | 47.37% | 2.63 pp | -10 | 9 | -1.11 |
| BTC Daily | nn | NN | 190 | 83 | 107 | 43.68% | 43.68% | 43.68% | 6.32 pp | -24 | 9 | -2.67 |
| BTC Daily | transformer | Transformer | 190 | 82 | 108 | 43.16% | 43.16% | 43.16% | 6.84 pp | -26 | 9 | -2.89 |
| BTC Daily | rf | RandomForest | 190 | 73 | 117 | 38.42% | 38.42% | 38.42% | 11.58 pp | -44 | 9 | -4.89 |
| BTC Daily | xgb | XGBoost | 200 | 72 | 128 | 36.00% | 36.00% | 36.00% | 14.00 pp | -56 | 10 | -5.60 |
| BTC Daily | lstm | LSTM | 190 | 66 | 124 | 34.74% | 34.74% | 34.74% | 15.26 pp | -58 | 9 | -6.44 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 188 | 96 | 92 | 51.06% | 51.06% | 51.06% | 1.06 pp | 4 | 15 | 0.27 |
| BTC Market Hours | transformer | Transformer | 188 | 93 | 95 | 49.47% | 49.47% | 49.47% | 0.53 pp | -2 | 15 | -0.13 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 188 | 87 | 101 | 46.28% | 46.28% | 46.28% | 3.72 pp | -14 | 15 | -0.93 |
| BTC Market Hours | rf | RandomForest | 188 | 87 | 101 | 46.28% | 46.28% | 46.28% | 3.72 pp | -14 | 15 | -0.93 |
| BTC Market Hours | xgb | XGBoost | 188 | 80 | 108 | 42.55% | 42.55% | 42.55% | 7.45 pp | -28 | 15 | -1.87 |
| BTC Market Hours | lstm | LSTM | 188 | 79 | 109 | 42.02% | 42.02% | 42.02% | 7.98 pp | -30 | 15 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 188 | 99 | 89 | 52.66% | 52.66% | 52.66% | 2.66 pp | 10 | 16 | 0.62 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 188 | 90 | 98 | 47.87% | 47.87% | 47.87% | 2.13 pp | -8 | 16 | -0.50 |
| BTC Market Hours Daily | nn | NN | 188 | 88 | 100 | 46.81% | 46.81% | 46.81% | 3.19 pp | -12 | 16 | -0.75 |
| BTC Market Hours Daily | rf | RandomForest | 188 | 84 | 104 | 44.68% | 44.68% | 44.68% | 5.32 pp | -20 | 16 | -1.25 |
| BTC Market Hours Daily | xgb | XGBoost | 188 | 78 | 110 | 41.49% | 41.49% | 41.49% | 8.51 pp | -32 | 16 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 188 | 76 | 112 | 40.43% | 40.43% | 40.43% | 9.57 pp | -36 | 16 | -2.25 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 161 | 80 | 81 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 161 | 78 | 83 | 48.45% | 48.45% | 48.45% | 1.55 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | lstm | LSTM | 161 | 73 | 88 | 45.34% | 45.34% | 45.34% | 4.66 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 161 | 70 | 91 | 43.48% | 43.48% | 43.48% | 6.52 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | transformer | Transformer | 161 | 69 | 92 | 42.86% | 42.86% | 42.86% | 7.14 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | nn | NN | 161 | 67 | 94 | 41.61% | 41.61% | 41.61% | 8.39 pp | -27 | 12 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 161 | 80 | 81 | 49.69% | 49.69% | 49.69% | 0.31 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 161 | 78 | 83 | 48.45% | 48.45% | 48.45% | 1.55 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 161 | 73 | 88 | 45.34% | 45.34% | 45.34% | 4.66 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 161 | 70 | 91 | 43.48% | 43.48% | 43.48% | 6.52 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 161 | 69 | 92 | 42.86% | 42.86% | 42.86% | 7.14 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 161 | 67 | 94 | 41.61% | 41.61% | 41.61% | 8.39 pp | -27 | 12 | -2.25 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours | rf | RandomForest | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours | lstm | LSTM | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours | nn | NN | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | transformer | Transformer | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 4 | -3.25 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 43 | 20 | 23 | 46.51% | 46.51% | 46.51% | 3.49 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 43 | 19 | 24 | 44.19% | 44.19% | 44.19% | 5.81 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 43 | 18 | 25 | 41.86% | 41.86% | 41.86% | 8.14 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | nn | NN | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | transformer | Transformer | 43 | 16 | 27 | 37.21% | 37.21% | 37.21% | 12.79 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 43 | 15 | 28 | 34.88% | 34.88% | 34.88% | 15.12 pp | -13 | 4 | -3.25 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
