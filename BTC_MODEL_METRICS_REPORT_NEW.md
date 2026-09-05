# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T18:01:44.724210+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 230 | 170 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 266 | 206 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 17:00:00+00:00 | 369 | 194 | 175 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 17:00:00+00:00 | 369 | 194 | 175 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T17:00:00+00:00 | 167 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T17:00:00+00:00 | 167 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T17:00:00+00:00 | 167 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T17:00:00+00:00 | 168 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 194 | 102 | 92 | 52.58% | 52.58% | 52.58% | 2.58 pp | 10 | 16 | 0.62 |
| BTC Market Hours | nn | NN | 194 | 100 | 94 | 51.55% | 51.55% | 51.55% | 1.55 pp | 6 | 15 | 0.40 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 170 | 86 | 84 | 50.59% | 50.59% | 50.59% | 0.59 pp | 2 | 8 | 0.25 |
| BTC Market Hours | transformer | Transformer | 194 | 96 | 98 | 49.48% | 49.48% | 49.48% | 0.52 pp | -2 | 15 | -0.13 |
| Consolidated Hourly | rf | RandomForest | 167 | 82 | 85 | 49.10% | 49.10% | 49.10% | 0.90 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 167 | 82 | 85 | 49.10% | 49.10% | 49.10% | 0.90 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 194 | 93 | 101 | 47.94% | 47.94% | 47.94% | 2.06 pp | -8 | 16 | -0.50 |
| BTC Market Hours Daily | nn | NN | 194 | 93 | 101 | 47.94% | 47.94% | 47.94% | 2.06 pp | -8 | 16 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 196 | 95 | 101 | 48.47% | 48.47% | 48.47% | 1.53 pp | -6 | 9 | -0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 47 | 22 | 25 | 46.81% | 46.81% | 46.81% | 3.19 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 47 | 22 | 25 | 46.81% | 46.81% | 46.81% | 3.19 pp | -3 | 4 | -0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 194 | 91 | 103 | 46.91% | 46.91% | 46.91% | 3.09 pp | -12 | 15 | -0.80 |
| BTC Market Hours | rf | RandomForest | 194 | 90 | 104 | 46.39% | 46.39% | 46.39% | 3.61 pp | -14 | 15 | -0.93 |
| BTC Hourly | transformer | Transformer | 170 | 81 | 89 | 47.65% | 47.65% | 47.65% | 2.35 pp | -8 | 8 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| BTC Market Hours Daily | rf | RandomForest | 194 | 87 | 107 | 44.85% | 44.85% | 44.85% | 5.15 pp | -20 | 16 | -1.25 |
| Consolidated Hourly | lstm | LSTM | 167 | 75 | 92 | 44.91% | 44.91% | 44.91% | 5.09 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | nn | NN | 167 | 75 | 92 | 44.91% | 44.91% | 44.91% | 5.09 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 167 | 75 | 92 | 44.91% | 44.91% | 44.91% | 5.09 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | nn | NN | 167 | 75 | 92 | 44.91% | 44.91% | 44.91% | 5.09 pp | -17 | 12 | -1.42 |
| Consolidated Market Hours | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 4 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 194 | 84 | 110 | 43.30% | 43.30% | 43.30% | 6.70 pp | -26 | 15 | -1.73 |
| Consolidated Market Hours Daily | rf | RandomForest | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 4 | -1.75 |
| BTC Market Hours | lstm | LSTM | 194 | 83 | 111 | 42.78% | 42.78% | 42.78% | 7.22 pp | -28 | 15 | -1.87 |
| Consolidated Hourly | transformer | Transformer | 167 | 72 | 95 | 43.11% | 43.11% | 43.11% | 6.89 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 167 | 72 | 95 | 43.11% | 43.11% | 43.11% | 6.89 pp | -23 | 12 | -1.92 |
| BTC Daily | nn | NN | 196 | 89 | 107 | 45.41% | 45.41% | 45.41% | 4.59 pp | -18 | 9 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 194 | 80 | 114 | 41.24% | 41.24% | 41.24% | 8.76 pp | -34 | 16 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 194 | 80 | 114 | 41.24% | 41.24% | 41.24% | 8.76 pp | -34 | 16 | -2.12 |
| BTC Hourly | nn | NN | 170 | 73 | 97 | 42.94% | 42.94% | 42.94% | 7.06 pp | -24 | 8 | -3.00 |
| BTC Hourly | rf | RandomForest | 170 | 73 | 97 | 42.94% | 42.94% | 42.94% | 7.06 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | transformer | Transformer | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| BTC Daily | transformer | Transformer | 196 | 84 | 112 | 42.86% | 42.86% | 42.86% | 7.14 pp | -28 | 9 | -3.11 |
| Consolidated Market Hours Daily | nn | NN | 47 | 17 | 30 | 36.17% | 36.17% | 36.17% | 13.83 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 47 | 17 | 30 | 36.17% | 36.17% | 36.17% | 13.83 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 47 | 15 | 32 | 31.91% | 31.91% | 31.91% | 18.09 pp | -17 | 4 | -4.25 |
| BTC Daily | rf | RandomForest | 196 | 76 | 120 | 38.78% | 38.78% | 38.78% | 11.22 pp | -44 | 9 | -4.89 |
| BTC Hourly | lstm | LSTM | 170 | 63 | 107 | 37.06% | 37.06% | 37.06% | 12.94 pp | -44 | 8 | -5.50 |
| BTC Daily | xgb | XGBoost | 206 | 74 | 132 | 35.92% | 35.92% | 35.92% | 14.08 pp | -58 | 10 | -5.80 |
| BTC Hourly | xgb | XGBoost | 170 | 61 | 109 | 35.88% | 35.88% | 35.88% | 14.12 pp | -48 | 8 | -6.00 |
| BTC Daily | lstm | LSTM | 196 | 67 | 129 | 34.18% | 34.18% | 34.18% | 15.82 pp | -62 | 9 | -6.89 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 170 | 86 | 84 | 50.59% | 50.59% | 50.59% | 0.59 pp | 2 | 8 | 0.25 |
| BTC Hourly | transformer | Transformer | 170 | 81 | 89 | 47.65% | 47.65% | 47.65% | 2.35 pp | -8 | 8 | -1.00 |
| BTC Hourly | nn | NN | 170 | 73 | 97 | 42.94% | 42.94% | 42.94% | 7.06 pp | -24 | 8 | -3.00 |
| BTC Hourly | rf | RandomForest | 170 | 73 | 97 | 42.94% | 42.94% | 42.94% | 7.06 pp | -24 | 8 | -3.00 |
| BTC Hourly | lstm | LSTM | 170 | 63 | 107 | 37.06% | 37.06% | 37.06% | 12.94 pp | -44 | 8 | -5.50 |
| BTC Hourly | xgb | XGBoost | 170 | 61 | 109 | 35.88% | 35.88% | 35.88% | 14.12 pp | -48 | 8 | -6.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 196 | 95 | 101 | 48.47% | 48.47% | 48.47% | 1.53 pp | -6 | 9 | -0.67 |
| BTC Daily | nn | NN | 196 | 89 | 107 | 45.41% | 45.41% | 45.41% | 4.59 pp | -18 | 9 | -2.00 |
| BTC Daily | transformer | Transformer | 196 | 84 | 112 | 42.86% | 42.86% | 42.86% | 7.14 pp | -28 | 9 | -3.11 |
| BTC Daily | rf | RandomForest | 196 | 76 | 120 | 38.78% | 38.78% | 38.78% | 11.22 pp | -44 | 9 | -4.89 |
| BTC Daily | xgb | XGBoost | 206 | 74 | 132 | 35.92% | 35.92% | 35.92% | 14.08 pp | -58 | 10 | -5.80 |
| BTC Daily | lstm | LSTM | 196 | 67 | 129 | 34.18% | 34.18% | 34.18% | 15.82 pp | -62 | 9 | -6.89 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 194 | 100 | 94 | 51.55% | 51.55% | 51.55% | 1.55 pp | 6 | 15 | 0.40 |
| BTC Market Hours | transformer | Transformer | 194 | 96 | 98 | 49.48% | 49.48% | 49.48% | 0.52 pp | -2 | 15 | -0.13 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 194 | 91 | 103 | 46.91% | 46.91% | 46.91% | 3.09 pp | -12 | 15 | -0.80 |
| BTC Market Hours | rf | RandomForest | 194 | 90 | 104 | 46.39% | 46.39% | 46.39% | 3.61 pp | -14 | 15 | -0.93 |
| BTC Market Hours | xgb | XGBoost | 194 | 84 | 110 | 43.30% | 43.30% | 43.30% | 6.70 pp | -26 | 15 | -1.73 |
| BTC Market Hours | lstm | LSTM | 194 | 83 | 111 | 42.78% | 42.78% | 42.78% | 7.22 pp | -28 | 15 | -1.87 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 194 | 102 | 92 | 52.58% | 52.58% | 52.58% | 2.58 pp | 10 | 16 | 0.62 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 194 | 93 | 101 | 47.94% | 47.94% | 47.94% | 2.06 pp | -8 | 16 | -0.50 |
| BTC Market Hours Daily | nn | NN | 194 | 93 | 101 | 47.94% | 47.94% | 47.94% | 2.06 pp | -8 | 16 | -0.50 |
| BTC Market Hours Daily | rf | RandomForest | 194 | 87 | 107 | 44.85% | 44.85% | 44.85% | 5.15 pp | -20 | 16 | -1.25 |
| BTC Market Hours Daily | lstm | LSTM | 194 | 80 | 114 | 41.24% | 41.24% | 41.24% | 8.76 pp | -34 | 16 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 194 | 80 | 114 | 41.24% | 41.24% | 41.24% | 8.76 pp | -34 | 16 | -2.12 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 167 | 82 | 85 | 49.10% | 49.10% | 49.10% | 0.90 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | xgb | XGBoost | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | lstm | LSTM | 167 | 75 | 92 | 44.91% | 44.91% | 44.91% | 5.09 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | nn | NN | 167 | 75 | 92 | 44.91% | 44.91% | 44.91% | 5.09 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | transformer | Transformer | 167 | 72 | 95 | 43.11% | 43.11% | 43.11% | 6.89 pp | -23 | 12 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 167 | 82 | 85 | 49.10% | 49.10% | 49.10% | 0.90 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 167 | 75 | 92 | 44.91% | 44.91% | 44.91% | 5.09 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | nn | NN | 167 | 75 | 92 | 44.91% | 44.91% | 44.91% | 5.09 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 167 | 72 | 95 | 43.11% | 43.11% | 43.11% | 6.89 pp | -23 | 12 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | transformer | Transformer | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 47 | 22 | 25 | 46.81% | 46.81% | 46.81% | 3.19 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 47 | 22 | 25 | 46.81% | 46.81% | 46.81% | 3.19 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 47 | 20 | 27 | 42.55% | 42.55% | 42.55% | 7.45 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | nn | NN | 47 | 17 | 30 | 36.17% | 36.17% | 36.17% | 13.83 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 47 | 17 | 30 | 36.17% | 36.17% | 36.17% | 13.83 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 47 | 15 | 32 | 31.91% | 31.91% | 31.91% | 18.09 pp | -17 | 4 | -4.25 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
