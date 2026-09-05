# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T17:39:55.242068+00:00
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
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 16:00:00+00:00 | 368 | 194 | 174 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 16:00:00+00:00 | 368 | 194 | 174 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T16:00:00+00:00 | 166 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T16:00:00+00:00 | 166 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T16:00:00+00:00 | 166 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T16:00:00+00:00 | 167 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 194 | 102 | 92 | 52.58% | 52.58% | 52.58% | 2.58 pp | 10 | 16 | 0.62 |
| BTC Market Hours | nn | NN | 194 | 100 | 94 | 51.55% | 51.55% | 51.55% | 1.55 pp | 6 | 15 | 0.40 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 170 | 86 | 84 | 50.59% | 50.59% | 50.59% | 0.59 pp | 2 | 8 | 0.25 |
| BTC Market Hours | transformer | Transformer | 194 | 96 | 98 | 49.48% | 49.48% | 49.48% | 0.52 pp | -2 | 15 | -0.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 12 | -0.33 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 194 | 93 | 101 | 47.94% | 47.94% | 47.94% | 2.06 pp | -8 | 16 | -0.50 |
| BTC Market Hours Daily | nn | NN | 194 | 93 | 101 | 47.94% | 47.94% | 47.94% | 2.06 pp | -8 | 16 | -0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 196 | 95 | 101 | 48.47% | 48.47% | 48.47% | 1.53 pp | -6 | 9 | -0.67 |
| Consolidated Market Hours | xgb | XGBoost | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 4 | -0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 194 | 91 | 103 | 46.91% | 46.91% | 46.91% | 3.09 pp | -12 | 15 | -0.80 |
| BTC Market Hours | rf | RandomForest | 194 | 90 | 104 | 46.39% | 46.39% | 46.39% | 3.61 pp | -14 | 15 | -0.93 |
| BTC Hourly | transformer | Transformer | 170 | 81 | 89 | 47.65% | 47.65% | 47.65% | 2.35 pp | -8 | 8 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 166 | 76 | 90 | 45.78% | 45.78% | 45.78% | 4.22 pp | -14 | 12 | -1.17 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 166 | 76 | 90 | 45.78% | 45.78% | 45.78% | 4.22 pp | -14 | 12 | -1.17 |
| BTC Market Hours Daily | rf | RandomForest | 194 | 87 | 107 | 44.85% | 44.85% | 44.85% | 5.15 pp | -20 | 16 | -1.25 |
| Consolidated Market Hours | lstm | LSTM | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | nn | NN | 166 | 75 | 91 | 45.18% | 45.18% | 45.18% | 4.82 pp | -16 | 12 | -1.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 166 | 75 | 91 | 45.18% | 45.18% | 45.18% | 4.82 pp | -16 | 12 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 166 | 74 | 92 | 44.58% | 44.58% | 44.58% | 5.42 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 166 | 74 | 92 | 44.58% | 44.58% | 44.58% | 5.42 pp | -18 | 12 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 194 | 84 | 110 | 43.30% | 43.30% | 43.30% | 6.70 pp | -26 | 15 | -1.73 |
| Consolidated Market Hours | rf | RandomForest | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 4 | -1.75 |
| BTC Market Hours | lstm | LSTM | 194 | 83 | 111 | 42.78% | 42.78% | 42.78% | 7.22 pp | -28 | 15 | -1.87 |
| BTC Daily | nn | NN | 196 | 89 | 107 | 45.41% | 45.41% | 45.41% | 4.59 pp | -18 | 9 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 166 | 71 | 95 | 42.77% | 42.77% | 42.77% | 7.23 pp | -24 | 12 | -2.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 166 | 71 | 95 | 42.77% | 42.77% | 42.77% | 7.23 pp | -24 | 12 | -2.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 46 | 19 | 27 | 41.30% | 41.30% | 41.30% | 8.70 pp | -8 | 4 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 194 | 80 | 114 | 41.24% | 41.24% | 41.24% | 8.76 pp | -34 | 16 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 194 | 80 | 114 | 41.24% | 41.24% | 41.24% | 8.76 pp | -34 | 16 | -2.12 |
| Consolidated Market Hours | nn | NN | 45 | 17 | 28 | 37.78% | 37.78% | 37.78% | 12.22 pp | -11 | 4 | -2.75 |
| BTC Hourly | nn | NN | 170 | 73 | 97 | 42.94% | 42.94% | 42.94% | 7.06 pp | -24 | 8 | -3.00 |
| BTC Hourly | rf | RandomForest | 170 | 73 | 97 | 42.94% | 42.94% | 42.94% | 7.06 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| BTC Daily | transformer | Transformer | 196 | 84 | 112 | 42.86% | 42.86% | 42.86% | 7.14 pp | -28 | 9 | -3.11 |
| Consolidated Market Hours | transformer | Transformer | 45 | 16 | 29 | 35.56% | 35.56% | 35.56% | 14.44 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 46 | 16 | 30 | 34.78% | 34.78% | 34.78% | 15.22 pp | -14 | 4 | -3.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 45 | 15 | 30 | 33.33% | 33.33% | 33.33% | 16.67 pp | -15 | 4 | -3.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |
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
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | xgb | XGBoost | 166 | 76 | 90 | 45.78% | 45.78% | 45.78% | 4.22 pp | -14 | 12 | -1.17 |
| Consolidated Hourly | nn | NN | 166 | 75 | 91 | 45.18% | 45.18% | 45.18% | 4.82 pp | -16 | 12 | -1.33 |
| Consolidated Hourly | lstm | LSTM | 166 | 74 | 92 | 44.58% | 44.58% | 44.58% | 5.42 pp | -18 | 12 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 166 | 71 | 95 | 42.77% | 42.77% | 42.77% | 7.23 pp | -24 | 12 | -2.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 166 | 81 | 85 | 48.80% | 48.80% | 48.80% | 1.20 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 166 | 76 | 90 | 45.78% | 45.78% | 45.78% | 4.22 pp | -14 | 12 | -1.17 |
| Consolidated Daily/Hourly Refresh | nn | NN | 166 | 75 | 91 | 45.18% | 45.18% | 45.18% | 4.82 pp | -16 | 12 | -1.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 166 | 74 | 92 | 44.58% | 44.58% | 44.58% | 5.42 pp | -18 | 12 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 166 | 71 | 95 | 42.77% | 42.77% | 42.77% | 7.23 pp | -24 | 12 | -2.00 |

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
| Consolidated Market Hours Daily | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 46 | 19 | 27 | 41.30% | 41.30% | 41.30% | 8.70 pp | -8 | 4 | -2.00 |
| Consolidated Market Hours Daily | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
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
