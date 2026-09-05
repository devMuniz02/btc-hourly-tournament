# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T22:31:51.622357+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 233 | 173 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 269 | 209 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 21:00:00+00:00 | 376 | 197 | 179 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 21:00:00+00:00 | 376 | 197 | 179 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T18:00:00+00:00 | 169 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T18:00:00+00:00 | 169 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T18:00:00+00:00 | 169 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T18:00:00+00:00 | 170 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 197 | 103 | 94 | 52.28% | 52.28% | 52.28% | 2.28 pp | 9 | 17 | 0.53 |
| BTC Market Hours | nn | NN | 197 | 101 | 96 | 51.27% | 51.27% | 51.27% | 1.27 pp | 5 | 16 | 0.31 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 173 | 87 | 86 | 50.29% | 50.29% | 50.29% | 0.29 pp | 1 | 8 | 0.12 |
| BTC Market Hours | transformer | Transformer | 197 | 97 | 100 | 49.24% | 49.24% | 49.24% | 0.76 pp | -3 | 16 | -0.19 |
| Consolidated Hourly | rf | RandomForest | 169 | 83 | 86 | 49.11% | 49.11% | 49.11% | 0.89 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 169 | 83 | 86 | 49.11% | 49.11% | 49.11% | 0.89 pp | -3 | 12 | -0.25 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 197 | 95 | 102 | 48.22% | 48.22% | 48.22% | 1.78 pp | -7 | 17 | -0.41 |
| BTC Market Hours Daily | nn | NN | 197 | 95 | 102 | 48.22% | 48.22% | 48.22% | 1.78 pp | -7 | 17 | -0.41 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 169 | 82 | 87 | 48.52% | 48.52% | 48.52% | 1.48 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 169 | 82 | 87 | 48.52% | 48.52% | 48.52% | 1.48 pp | -5 | 12 | -0.42 |
| Consolidated Market Hours | xgb | XGBoost | 47 | 22 | 25 | 46.81% | 46.81% | 46.81% | 3.19 pp | -3 | 4 | -0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 199 | 96 | 103 | 48.24% | 48.24% | 48.24% | 1.76 pp | -7 | 9 | -0.78 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 197 | 92 | 105 | 46.70% | 46.70% | 46.70% | 3.30 pp | -13 | 16 | -0.81 |
| Consolidated Market Hours Daily | xgb | XGBoost | 48 | 22 | 26 | 45.83% | 45.83% | 45.83% | 4.17 pp | -4 | 4 | -1.00 |
| BTC Market Hours | rf | RandomForest | 197 | 90 | 107 | 45.69% | 45.69% | 45.69% | 4.31 pp | -17 | 16 | -1.06 |
| Consolidated Hourly | xgb | XGBoost | 169 | 77 | 92 | 45.56% | 45.56% | 45.56% | 4.44 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 169 | 77 | 92 | 45.56% | 45.56% | 45.56% | 4.44 pp | -15 | 12 | -1.25 |
| Consolidated Market Hours | lstm | LSTM | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 17 | -1.35 |
| BTC Hourly | transformer | Transformer | 173 | 81 | 92 | 46.82% | 46.82% | 46.82% | 3.18 pp | -11 | 8 | -1.38 |
| Consolidated Hourly | lstm | LSTM | 169 | 76 | 93 | 44.97% | 44.97% | 44.97% | 5.03 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | nn | NN | 169 | 76 | 93 | 44.97% | 44.97% | 44.97% | 5.03 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 169 | 76 | 93 | 44.97% | 44.97% | 44.97% | 5.03 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | nn | NN | 169 | 76 | 93 | 44.97% | 44.97% | 44.97% | 5.03 pp | -17 | 12 | -1.42 |
| Consolidated Market Hours Daily | lstm | LSTM | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 197 | 86 | 111 | 43.65% | 43.65% | 43.65% | 6.35 pp | -25 | 16 | -1.56 |
| Consolidated Market Hours | rf | RandomForest | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 4 | -1.75 |
| BTC Market Hours | lstm | LSTM | 197 | 84 | 113 | 42.64% | 42.64% | 42.64% | 7.36 pp | -29 | 16 | -1.81 |
| BTC Market Hours Daily | xgb | XGBoost | 197 | 82 | 115 | 41.62% | 41.62% | 41.62% | 8.38 pp | -33 | 17 | -1.94 |
| Consolidated Market Hours Daily | rf | RandomForest | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 4 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 197 | 81 | 116 | 41.12% | 41.12% | 41.12% | 8.88 pp | -35 | 17 | -2.06 |
| Consolidated Hourly | transformer | Transformer | 169 | 72 | 97 | 42.60% | 42.60% | 42.60% | 7.40 pp | -25 | 12 | -2.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 169 | 72 | 97 | 42.60% | 42.60% | 42.60% | 7.40 pp | -25 | 12 | -2.08 |
| BTC Daily | nn | NN | 199 | 90 | 109 | 45.23% | 45.23% | 45.23% | 4.77 pp | -19 | 9 | -2.11 |
| Consolidated Market Hours Daily | transformer | Transformer | 48 | 19 | 29 | 39.58% | 39.58% | 39.58% | 10.42 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | transformer | Transformer | 47 | 18 | 29 | 38.30% | 38.30% | 38.30% | 11.70 pp | -11 | 4 | -2.75 |
| BTC Hourly | rf | RandomForest | 173 | 74 | 99 | 42.77% | 42.77% | 42.77% | 7.23 pp | -25 | 8 | -3.12 |
| BTC Daily | transformer | Transformer | 199 | 85 | 114 | 42.71% | 42.71% | 42.71% | 7.29 pp | -29 | 9 | -3.22 |
| Consolidated Market Hours | nn | NN | 47 | 17 | 30 | 36.17% | 36.17% | 36.17% | 13.83 pp | -13 | 4 | -3.25 |
| BTC Hourly | nn | NN | 173 | 73 | 100 | 42.20% | 42.20% | 42.20% | 7.80 pp | -27 | 8 | -3.38 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 4 | -3.75 |
| BTC Daily | rf | RandomForest | 199 | 77 | 122 | 38.69% | 38.69% | 38.69% | 11.31 pp | -45 | 9 | -5.00 |
| BTC Hourly | lstm | LSTM | 173 | 63 | 110 | 36.42% | 36.42% | 36.42% | 13.58 pp | -47 | 8 | -5.88 |
| BTC Daily | xgb | XGBoost | 209 | 74 | 135 | 35.41% | 35.41% | 35.41% | 14.59 pp | -61 | 10 | -6.10 |
| BTC Hourly | xgb | XGBoost | 173 | 62 | 111 | 35.84% | 35.84% | 35.84% | 14.16 pp | -49 | 8 | -6.12 |
| BTC Daily | lstm | LSTM | 199 | 67 | 132 | 33.67% | 33.67% | 33.67% | 16.33 pp | -65 | 9 | -7.22 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 173 | 87 | 86 | 50.29% | 50.29% | 50.29% | 0.29 pp | 1 | 8 | 0.12 |
| BTC Hourly | transformer | Transformer | 173 | 81 | 92 | 46.82% | 46.82% | 46.82% | 3.18 pp | -11 | 8 | -1.38 |
| BTC Hourly | rf | RandomForest | 173 | 74 | 99 | 42.77% | 42.77% | 42.77% | 7.23 pp | -25 | 8 | -3.12 |
| BTC Hourly | nn | NN | 173 | 73 | 100 | 42.20% | 42.20% | 42.20% | 7.80 pp | -27 | 8 | -3.38 |
| BTC Hourly | lstm | LSTM | 173 | 63 | 110 | 36.42% | 36.42% | 36.42% | 13.58 pp | -47 | 8 | -5.88 |
| BTC Hourly | xgb | XGBoost | 173 | 62 | 111 | 35.84% | 35.84% | 35.84% | 14.16 pp | -49 | 8 | -6.12 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 199 | 96 | 103 | 48.24% | 48.24% | 48.24% | 1.76 pp | -7 | 9 | -0.78 |
| BTC Daily | nn | NN | 199 | 90 | 109 | 45.23% | 45.23% | 45.23% | 4.77 pp | -19 | 9 | -2.11 |
| BTC Daily | transformer | Transformer | 199 | 85 | 114 | 42.71% | 42.71% | 42.71% | 7.29 pp | -29 | 9 | -3.22 |
| BTC Daily | rf | RandomForest | 199 | 77 | 122 | 38.69% | 38.69% | 38.69% | 11.31 pp | -45 | 9 | -5.00 |
| BTC Daily | xgb | XGBoost | 209 | 74 | 135 | 35.41% | 35.41% | 35.41% | 14.59 pp | -61 | 10 | -6.10 |
| BTC Daily | lstm | LSTM | 199 | 67 | 132 | 33.67% | 33.67% | 33.67% | 16.33 pp | -65 | 9 | -7.22 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 197 | 101 | 96 | 51.27% | 51.27% | 51.27% | 1.27 pp | 5 | 16 | 0.31 |
| BTC Market Hours | transformer | Transformer | 197 | 97 | 100 | 49.24% | 49.24% | 49.24% | 0.76 pp | -3 | 16 | -0.19 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 197 | 92 | 105 | 46.70% | 46.70% | 46.70% | 3.30 pp | -13 | 16 | -0.81 |
| BTC Market Hours | rf | RandomForest | 197 | 90 | 107 | 45.69% | 45.69% | 45.69% | 4.31 pp | -17 | 16 | -1.06 |
| BTC Market Hours | xgb | XGBoost | 197 | 86 | 111 | 43.65% | 43.65% | 43.65% | 6.35 pp | -25 | 16 | -1.56 |
| BTC Market Hours | lstm | LSTM | 197 | 84 | 113 | 42.64% | 42.64% | 42.64% | 7.36 pp | -29 | 16 | -1.81 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 197 | 103 | 94 | 52.28% | 52.28% | 52.28% | 2.28 pp | 9 | 17 | 0.53 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 197 | 95 | 102 | 48.22% | 48.22% | 48.22% | 1.78 pp | -7 | 17 | -0.41 |
| BTC Market Hours Daily | nn | NN | 197 | 95 | 102 | 48.22% | 48.22% | 48.22% | 1.78 pp | -7 | 17 | -0.41 |
| BTC Market Hours Daily | rf | RandomForest | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 17 | -1.35 |
| BTC Market Hours Daily | xgb | XGBoost | 197 | 82 | 115 | 41.62% | 41.62% | 41.62% | 8.38 pp | -33 | 17 | -1.94 |
| BTC Market Hours Daily | lstm | LSTM | 197 | 81 | 116 | 41.12% | 41.12% | 41.12% | 8.88 pp | -35 | 17 | -2.06 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 169 | 83 | 86 | 49.11% | 49.11% | 49.11% | 0.89 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 169 | 82 | 87 | 48.52% | 48.52% | 48.52% | 1.48 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | xgb | XGBoost | 169 | 77 | 92 | 45.56% | 45.56% | 45.56% | 4.44 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | lstm | LSTM | 169 | 76 | 93 | 44.97% | 44.97% | 44.97% | 5.03 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | nn | NN | 169 | 76 | 93 | 44.97% | 44.97% | 44.97% | 5.03 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | transformer | Transformer | 169 | 72 | 97 | 42.60% | 42.60% | 42.60% | 7.40 pp | -25 | 12 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 169 | 83 | 86 | 49.11% | 49.11% | 49.11% | 0.89 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 169 | 82 | 87 | 48.52% | 48.52% | 48.52% | 1.48 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 169 | 77 | 92 | 45.56% | 45.56% | 45.56% | 4.44 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 169 | 76 | 93 | 44.97% | 44.97% | 44.97% | 5.03 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | nn | NN | 169 | 76 | 93 | 44.97% | 44.97% | 44.97% | 5.03 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 169 | 72 | 97 | 42.60% | 42.60% | 42.60% | 7.40 pp | -25 | 12 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 47 | 22 | 25 | 46.81% | 46.81% | 46.81% | 3.19 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours | lstm | LSTM | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours | transformer | Transformer | 47 | 18 | 29 | 38.30% | 38.30% | 38.30% | 11.70 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | nn | NN | 47 | 17 | 30 | 36.17% | 36.17% | 36.17% | 13.83 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 47 | 16 | 31 | 34.04% | 34.04% | 34.04% | 15.96 pp | -15 | 4 | -3.75 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 48 | 22 | 26 | 45.83% | 45.83% | 45.83% | 4.17 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 48 | 19 | 29 | 39.58% | 39.58% | 39.58% | 10.42 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 4 | -3.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
