# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T17:20:42.125251+00:00
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
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 16:00:00+00:00 | 367 | 193 | 174 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 16:00:00+00:00 | 165 | 165 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 16:00:00+00:00 | 165 | 165 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 16:00:00+00:00 | 165 | 45 | 120 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 16:00:00+00:00 | 165 | 45 | 120 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 193 | 102 | 91 | 52.85% | 52.85% | 52.85% | 2.85 pp | 11 | 16 | 0.69 |
| BTC Market Hours | nn | NN | 194 | 100 | 94 | 51.55% | 51.55% | 51.55% | 1.55 pp | 6 | 15 | 0.40 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 170 | 86 | 84 | 50.59% | 50.59% | 50.59% | 0.59 pp | 2 | 8 | 0.25 |
| Consolidated Hourly | rf | RandomForest | 165 | 82 | 83 | 49.70% | 49.70% | 49.70% | 0.30 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 165 | 82 | 83 | 49.70% | 49.70% | 49.70% | 0.30 pp | -1 | 12 | -0.08 |
| BTC Market Hours | transformer | Transformer | 194 | 96 | 98 | 49.48% | 49.48% | 49.48% | 0.52 pp | -2 | 15 | -0.13 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 193 | 92 | 101 | 47.67% | 47.67% | 47.67% | 2.33 pp | -9 | 16 | -0.56 |
| BTC Market Hours Daily | nn | NN | 193 | 92 | 101 | 47.67% | 47.67% | 47.67% | 2.33 pp | -9 | 16 | -0.56 |
| BTC Daily | mlp_sklearn | MLPClassifier | 196 | 95 | 101 | 48.47% | 48.47% | 48.47% | 1.53 pp | -6 | 9 | -0.67 |
| Consolidated Market Hours | xgb | XGBoost | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 4 | -0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 194 | 91 | 103 | 46.91% | 46.91% | 46.91% | 3.09 pp | -12 | 15 | -0.80 |
| BTC Market Hours | rf | RandomForest | 194 | 90 | 104 | 46.39% | 46.39% | 46.39% | 3.61 pp | -14 | 15 | -0.93 |
| BTC Hourly | transformer | Transformer | 170 | 81 | 89 | 47.65% | 47.65% | 47.65% | 2.35 pp | -8 | 8 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 165 | 76 | 89 | 46.06% | 46.06% | 46.06% | 3.94 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 165 | 76 | 89 | 46.06% | 46.06% | 46.06% | 3.94 pp | -13 | 12 | -1.08 |
| Consolidated Market Hours | lstm | LSTM | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 193 | 86 | 107 | 44.56% | 44.56% | 44.56% | 5.44 pp | -21 | 16 | -1.31 |
| BTC Market Hours | xgb | XGBoost | 194 | 84 | 110 | 43.30% | 43.30% | 43.30% | 6.70 pp | -26 | 15 | -1.73 |
| Consolidated Hourly | transformer | Transformer | 165 | 72 | 93 | 43.64% | 43.64% | 43.64% | 6.36 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 165 | 72 | 93 | 43.64% | 43.64% | 43.64% | 6.36 pp | -21 | 12 | -1.75 |
| Consolidated Market Hours | rf | RandomForest | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 4 | -1.75 |
| BTC Market Hours | lstm | LSTM | 194 | 83 | 111 | 42.78% | 42.78% | 42.78% | 7.22 pp | -28 | 15 | -1.87 |
| Consolidated Hourly | xgb | XGBoost | 165 | 71 | 94 | 43.03% | 43.03% | 43.03% | 6.97 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 165 | 71 | 94 | 43.03% | 43.03% | 43.03% | 6.97 pp | -23 | 12 | -1.92 |
| BTC Daily | nn | NN | 196 | 89 | 107 | 45.41% | 45.41% | 45.41% | 4.59 pp | -18 | 9 | -2.00 |
| Consolidated Hourly | nn | NN | 165 | 70 | 95 | 42.42% | 42.42% | 42.42% | 7.58 pp | -25 | 12 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 165 | 70 | 95 | 42.42% | 42.42% | 42.42% | 7.58 pp | -25 | 12 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 193 | 79 | 114 | 40.93% | 40.93% | 40.93% | 9.07 pp | -35 | 16 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 193 | 79 | 114 | 40.93% | 40.93% | 40.93% | 9.07 pp | -35 | 16 | -2.19 |
| Consolidated Market Hours | nn | NN | 45 | 17 | 28 | 37.78% | 37.78% | 37.78% | 12.22 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 45 | 17 | 28 | 37.78% | 37.78% | 37.78% | 12.22 pp | -11 | 4 | -2.75 |
| BTC Hourly | nn | NN | 170 | 73 | 97 | 42.94% | 42.94% | 42.94% | 7.06 pp | -24 | 8 | -3.00 |
| BTC Hourly | rf | RandomForest | 170 | 73 | 97 | 42.94% | 42.94% | 42.94% | 7.06 pp | -24 | 8 | -3.00 |
| BTC Daily | transformer | Transformer | 196 | 84 | 112 | 42.86% | 42.86% | 42.86% | 7.14 pp | -28 | 9 | -3.11 |
| Consolidated Market Hours | transformer | Transformer | 45 | 16 | 29 | 35.56% | 35.56% | 35.56% | 14.44 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 45 | 16 | 29 | 35.56% | 35.56% | 35.56% | 14.44 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 45 | 15 | 30 | 33.33% | 33.33% | 33.33% | 16.67 pp | -15 | 4 | -3.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 45 | 15 | 30 | 33.33% | 33.33% | 33.33% | 16.67 pp | -15 | 4 | -3.75 |
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
| BTC Market Hours Daily | transformer | Transformer | 193 | 102 | 91 | 52.85% | 52.85% | 52.85% | 2.85 pp | 11 | 16 | 0.69 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 193 | 92 | 101 | 47.67% | 47.67% | 47.67% | 2.33 pp | -9 | 16 | -0.56 |
| BTC Market Hours Daily | nn | NN | 193 | 92 | 101 | 47.67% | 47.67% | 47.67% | 2.33 pp | -9 | 16 | -0.56 |
| BTC Market Hours Daily | rf | RandomForest | 193 | 86 | 107 | 44.56% | 44.56% | 44.56% | 5.44 pp | -21 | 16 | -1.31 |
| BTC Market Hours Daily | lstm | LSTM | 193 | 79 | 114 | 40.93% | 40.93% | 40.93% | 9.07 pp | -35 | 16 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 193 | 79 | 114 | 40.93% | 40.93% | 40.93% | 9.07 pp | -35 | 16 | -2.19 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 165 | 82 | 83 | 49.70% | 49.70% | 49.70% | 0.30 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | lstm | LSTM | 165 | 76 | 89 | 46.06% | 46.06% | 46.06% | 3.94 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | transformer | Transformer | 165 | 72 | 93 | 43.64% | 43.64% | 43.64% | 6.36 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | xgb | XGBoost | 165 | 71 | 94 | 43.03% | 43.03% | 43.03% | 6.97 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | nn | NN | 165 | 70 | 95 | 42.42% | 42.42% | 42.42% | 7.58 pp | -25 | 12 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 165 | 82 | 83 | 49.70% | 49.70% | 49.70% | 0.30 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 165 | 76 | 89 | 46.06% | 46.06% | 46.06% | 3.94 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 165 | 72 | 93 | 43.64% | 43.64% | 43.64% | 6.36 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 165 | 71 | 94 | 43.03% | 43.03% | 43.03% | 6.97 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 165 | 70 | 95 | 42.42% | 42.42% | 42.42% | 7.58 pp | -25 | 12 | -2.08 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | lstm | LSTM | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | nn | NN | 45 | 17 | 28 | 37.78% | 37.78% | 37.78% | 12.22 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | transformer | Transformer | 45 | 16 | 29 | 35.56% | 35.56% | 35.56% | 14.44 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 45 | 15 | 30 | 33.33% | 33.33% | 33.33% | 16.67 pp | -15 | 4 | -3.75 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
