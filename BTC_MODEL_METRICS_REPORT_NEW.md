# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T15:13:26.592616+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 228 | 168 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 264 | 204 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 14:00:00+00:00 | 364 | 192 | 172 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 14:00:00+00:00 | 364 | 192 | 172 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T16:00:00+00:00 | 165 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T16:00:00+00:00 | 165 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T16:00:00+00:00 | 165 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T16:00:00+00:00 | 166 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 192 | 101 | 91 | 52.60% | 52.60% | 52.60% | 2.60 pp | 10 | 16 | 0.62 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 168 | 86 | 82 | 51.19% | 51.19% | 51.19% | 1.19 pp | 4 | 7 | 0.57 |
| BTC Market Hours | nn | NN | 192 | 99 | 93 | 51.56% | 51.56% | 51.56% | 1.56 pp | 6 | 15 | 0.40 |
| BTC Market Hours | transformer | Transformer | 192 | 96 | 96 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 15 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | rf | RandomForest | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 192 | 92 | 100 | 47.92% | 47.92% | 47.92% | 2.08 pp | -8 | 16 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | nn | NN | 192 | 91 | 101 | 47.40% | 47.40% | 47.40% | 2.60 pp | -10 | 16 | -0.62 |
| Consolidated Market Hours | xgb | XGBoost | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 4 | -0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 192 | 90 | 102 | 46.88% | 46.88% | 46.88% | 3.12 pp | -12 | 15 | -0.80 |
| BTC Hourly | transformer | Transformer | 168 | 81 | 87 | 48.21% | 48.21% | 48.21% | 1.79 pp | -6 | 7 | -0.86 |
| BTC Daily | mlp_sklearn | MLPClassifier | 194 | 93 | 101 | 47.94% | 47.94% | 47.94% | 2.06 pp | -8 | 9 | -0.89 |
| BTC Market Hours | rf | RandomForest | 192 | 89 | 103 | 46.35% | 46.35% | 46.35% | 3.65 pp | -14 | 15 | -0.93 |
| Consolidated Market Hours Daily | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 165 | 75 | 90 | 45.45% | 45.45% | 45.45% | 4.55 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 165 | 75 | 90 | 45.45% | 45.45% | 45.45% | 4.55 pp | -15 | 12 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 192 | 86 | 106 | 44.79% | 44.79% | 44.79% | 5.21 pp | -20 | 16 | -1.25 |
| Consolidated Market Hours | lstm | LSTM | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | lstm | LSTM | 165 | 74 | 91 | 44.85% | 44.85% | 44.85% | 5.15 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | nn | NN | 165 | 74 | 91 | 44.85% | 44.85% | 44.85% | 5.15 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 165 | 74 | 91 | 44.85% | 44.85% | 44.85% | 5.15 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | nn | NN | 165 | 74 | 91 | 44.85% | 44.85% | 44.85% | 5.15 pp | -17 | 12 | -1.42 |
| BTC Market Hours | xgb | XGBoost | 192 | 83 | 109 | 43.23% | 43.23% | 43.23% | 6.77 pp | -26 | 15 | -1.73 |
| Consolidated Market Hours | rf | RandomForest | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 4 | -1.75 |
| BTC Market Hours | lstm | LSTM | 192 | 82 | 110 | 42.71% | 42.71% | 42.71% | 7.29 pp | -28 | 15 | -1.87 |
| Consolidated Market Hours Daily | rf | RandomForest | 46 | 19 | 27 | 41.30% | 41.30% | 41.30% | 8.70 pp | -8 | 4 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 165 | 70 | 95 | 42.42% | 42.42% | 42.42% | 7.58 pp | -25 | 12 | -2.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 165 | 70 | 95 | 42.42% | 42.42% | 42.42% | 7.58 pp | -25 | 12 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 192 | 79 | 113 | 41.15% | 41.15% | 41.15% | 8.85 pp | -34 | 16 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 192 | 79 | 113 | 41.15% | 41.15% | 41.15% | 8.85 pp | -34 | 16 | -2.12 |
| BTC Daily | nn | NN | 194 | 87 | 107 | 44.85% | 44.85% | 44.85% | 5.15 pp | -20 | 9 | -2.22 |
| Consolidated Market Hours Daily | nn | NN | 46 | 18 | 28 | 39.13% | 39.13% | 39.13% | 10.87 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours | nn | NN | 45 | 17 | 28 | 37.78% | 37.78% | 37.78% | 12.22 pp | -11 | 4 | -2.75 |
| BTC Daily | transformer | Transformer | 194 | 84 | 110 | 43.30% | 43.30% | 43.30% | 6.70 pp | -26 | 9 | -2.89 |
| BTC Hourly | nn | NN | 168 | 73 | 95 | 43.45% | 43.45% | 43.45% | 6.55 pp | -22 | 7 | -3.14 |
| Consolidated Market Hours | transformer | Transformer | 45 | 16 | 29 | 35.56% | 35.56% | 35.56% | 14.44 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 46 | 16 | 30 | 34.78% | 34.78% | 34.78% | 15.22 pp | -14 | 4 | -3.50 |
| BTC Hourly | rf | RandomForest | 168 | 71 | 97 | 42.26% | 42.26% | 42.26% | 7.74 pp | -26 | 7 | -3.71 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 45 | 15 | 30 | 33.33% | 33.33% | 33.33% | 16.67 pp | -15 | 4 | -3.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |
| BTC Daily | rf | RandomForest | 194 | 76 | 118 | 39.18% | 39.18% | 39.18% | 10.82 pp | -42 | 9 | -4.67 |
| BTC Daily | xgb | XGBoost | 204 | 74 | 130 | 36.27% | 36.27% | 36.27% | 13.73 pp | -56 | 10 | -5.60 |
| BTC Hourly | lstm | LSTM | 168 | 61 | 107 | 36.31% | 36.31% | 36.31% | 13.69 pp | -46 | 7 | -6.57 |
| BTC Daily | lstm | LSTM | 194 | 67 | 127 | 34.54% | 34.54% | 34.54% | 15.46 pp | -60 | 9 | -6.67 |
| BTC Hourly | xgb | XGBoost | 168 | 60 | 108 | 35.71% | 35.71% | 35.71% | 14.29 pp | -48 | 7 | -6.86 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 168 | 86 | 82 | 51.19% | 51.19% | 51.19% | 1.19 pp | 4 | 7 | 0.57 |
| BTC Hourly | transformer | Transformer | 168 | 81 | 87 | 48.21% | 48.21% | 48.21% | 1.79 pp | -6 | 7 | -0.86 |
| BTC Hourly | nn | NN | 168 | 73 | 95 | 43.45% | 43.45% | 43.45% | 6.55 pp | -22 | 7 | -3.14 |
| BTC Hourly | rf | RandomForest | 168 | 71 | 97 | 42.26% | 42.26% | 42.26% | 7.74 pp | -26 | 7 | -3.71 |
| BTC Hourly | lstm | LSTM | 168 | 61 | 107 | 36.31% | 36.31% | 36.31% | 13.69 pp | -46 | 7 | -6.57 |
| BTC Hourly | xgb | XGBoost | 168 | 60 | 108 | 35.71% | 35.71% | 35.71% | 14.29 pp | -48 | 7 | -6.86 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 194 | 93 | 101 | 47.94% | 47.94% | 47.94% | 2.06 pp | -8 | 9 | -0.89 |
| BTC Daily | nn | NN | 194 | 87 | 107 | 44.85% | 44.85% | 44.85% | 5.15 pp | -20 | 9 | -2.22 |
| BTC Daily | transformer | Transformer | 194 | 84 | 110 | 43.30% | 43.30% | 43.30% | 6.70 pp | -26 | 9 | -2.89 |
| BTC Daily | rf | RandomForest | 194 | 76 | 118 | 39.18% | 39.18% | 39.18% | 10.82 pp | -42 | 9 | -4.67 |
| BTC Daily | xgb | XGBoost | 204 | 74 | 130 | 36.27% | 36.27% | 36.27% | 13.73 pp | -56 | 10 | -5.60 |
| BTC Daily | lstm | LSTM | 194 | 67 | 127 | 34.54% | 34.54% | 34.54% | 15.46 pp | -60 | 9 | -6.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 192 | 99 | 93 | 51.56% | 51.56% | 51.56% | 1.56 pp | 6 | 15 | 0.40 |
| BTC Market Hours | transformer | Transformer | 192 | 96 | 96 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 15 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 192 | 90 | 102 | 46.88% | 46.88% | 46.88% | 3.12 pp | -12 | 15 | -0.80 |
| BTC Market Hours | rf | RandomForest | 192 | 89 | 103 | 46.35% | 46.35% | 46.35% | 3.65 pp | -14 | 15 | -0.93 |
| BTC Market Hours | xgb | XGBoost | 192 | 83 | 109 | 43.23% | 43.23% | 43.23% | 6.77 pp | -26 | 15 | -1.73 |
| BTC Market Hours | lstm | LSTM | 192 | 82 | 110 | 42.71% | 42.71% | 42.71% | 7.29 pp | -28 | 15 | -1.87 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 192 | 101 | 91 | 52.60% | 52.60% | 52.60% | 2.60 pp | 10 | 16 | 0.62 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 192 | 92 | 100 | 47.92% | 47.92% | 47.92% | 2.08 pp | -8 | 16 | -0.50 |
| BTC Market Hours Daily | nn | NN | 192 | 91 | 101 | 47.40% | 47.40% | 47.40% | 2.60 pp | -10 | 16 | -0.62 |
| BTC Market Hours Daily | rf | RandomForest | 192 | 86 | 106 | 44.79% | 44.79% | 44.79% | 5.21 pp | -20 | 16 | -1.25 |
| BTC Market Hours Daily | lstm | LSTM | 192 | 79 | 113 | 41.15% | 41.15% | 41.15% | 8.85 pp | -34 | 16 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 192 | 79 | 113 | 41.15% | 41.15% | 41.15% | 8.85 pp | -34 | 16 | -2.12 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | rf | RandomForest | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | xgb | XGBoost | 165 | 75 | 90 | 45.45% | 45.45% | 45.45% | 4.55 pp | -15 | 12 | -1.25 |
| Consolidated Hourly | lstm | LSTM | 165 | 74 | 91 | 44.85% | 44.85% | 44.85% | 5.15 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | nn | NN | 165 | 74 | 91 | 44.85% | 44.85% | 44.85% | 5.15 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | transformer | Transformer | 165 | 70 | 95 | 42.42% | 42.42% | 42.42% | 7.58 pp | -25 | 12 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 165 | 75 | 90 | 45.45% | 45.45% | 45.45% | 4.55 pp | -15 | 12 | -1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 165 | 74 | 91 | 44.85% | 44.85% | 44.85% | 5.15 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | nn | NN | 165 | 74 | 91 | 44.85% | 44.85% | 44.85% | 5.15 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 165 | 70 | 95 | 42.42% | 42.42% | 42.42% | 7.58 pp | -25 | 12 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours | lstm | LSTM | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours | nn | NN | 45 | 17 | 28 | 37.78% | 37.78% | 37.78% | 12.22 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | transformer | Transformer | 45 | 16 | 29 | 35.56% | 35.56% | 35.56% | 14.44 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 45 | 15 | 30 | 33.33% | 33.33% | 33.33% | 16.67 pp | -15 | 4 | -3.75 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 46 | 19 | 27 | 41.30% | 41.30% | 41.30% | 8.70 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 46 | 18 | 28 | 39.13% | 39.13% | 39.13% | 10.87 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 46 | 16 | 30 | 34.78% | 34.78% | 34.78% | 15.22 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
