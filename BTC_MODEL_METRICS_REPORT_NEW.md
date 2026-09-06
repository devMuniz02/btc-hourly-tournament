# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-06T15:30:13.559485+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 245 | 185 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 281 | 221 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-06 14:00:00+00:00 | 394 | 209 | 185 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-06 14:00:00+00:00 | 394 | 209 | 185 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T00:00:00+00:00 | 179 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T00:00:00+00:00 | 179 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T00:00:00+00:00 | 179 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T00:00:00+00:00 | 180 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 209 | 108 | 101 | 51.67% | 51.67% | 51.67% | 1.67 pp | 7 | 17 | 0.41 |
| BTC Market Hours Daily | transformer | Transformer | 209 | 108 | 101 | 51.67% | 51.67% | 51.67% | 1.67 pp | 7 | 17 | 0.41 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 185 | 93 | 92 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 8 | 0.12 |
| BTC Market Hours | transformer | Transformer | 209 | 103 | 106 | 49.28% | 49.28% | 49.28% | 0.72 pp | -3 | 17 | -0.18 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 209 | 103 | 106 | 49.28% | 49.28% | 49.28% | 0.72 pp | -3 | 17 | -0.18 |
| Consolidated Market Hours | xgb | XGBoost | 53 | 26 | 27 | 49.06% | 49.06% | 49.06% | 0.94 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 179 | 87 | 92 | 48.60% | 48.60% | 48.60% | 1.40 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 179 | 87 | 92 | 48.60% | 48.60% | 48.60% | 1.40 pp | -5 | 13 | -0.38 |
| Consolidated Market Hours Daily | xgb | XGBoost | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | nn | NN | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 17 | -0.53 |
| Consolidated Hourly | rf | RandomForest | 179 | 86 | 93 | 48.04% | 48.04% | 48.04% | 1.96 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 179 | 86 | 93 | 48.04% | 48.04% | 48.04% | 1.96 pp | -7 | 13 | -0.54 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 209 | 98 | 111 | 46.89% | 46.89% | 46.89% | 3.11 pp | -13 | 17 | -0.76 |
| BTC Daily | mlp_sklearn | MLPClassifier | 211 | 101 | 110 | 47.87% | 47.87% | 47.87% | 2.13 pp | -9 | 10 | -0.90 |
| BTC Market Hours | rf | RandomForest | 209 | 96 | 113 | 45.93% | 45.93% | 45.93% | 4.07 pp | -17 | 17 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 53 | 24 | 29 | 45.28% | 45.28% | 45.28% | 4.72 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 179 | 82 | 97 | 45.81% | 45.81% | 45.81% | 4.19 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 179 | 82 | 97 | 45.81% | 45.81% | 45.81% | 4.19 pp | -15 | 13 | -1.15 |
| Consolidated Market Hours Daily | lstm | LSTM | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | lstm | LSTM | 179 | 81 | 98 | 45.25% | 45.25% | 45.25% | 4.75 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 179 | 81 | 98 | 45.25% | 45.25% | 45.25% | 4.75 pp | -17 | 13 | -1.31 |
| BTC Market Hours | xgb | XGBoost | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 17 | -1.47 |
| BTC Market Hours Daily | rf | RandomForest | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 17 | -1.47 |
| Consolidated Hourly | nn | NN | 179 | 79 | 100 | 44.13% | 44.13% | 44.13% | 5.87 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | nn | NN | 179 | 79 | 100 | 44.13% | 44.13% | 44.13% | 5.87 pp | -21 | 13 | -1.62 |
| BTC Hourly | transformer | Transformer | 185 | 86 | 99 | 46.49% | 46.49% | 46.49% | 3.51 pp | -13 | 8 | -1.62 |
| Consolidated Market Hours | rf | RandomForest | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 5 | -1.80 |
| BTC Daily | nn | NN | 211 | 96 | 115 | 45.50% | 45.50% | 45.50% | 4.50 pp | -19 | 10 | -1.90 |
| Consolidated Market Hours Daily | rf | RandomForest | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 5 | -2.00 |
| BTC Market Hours | lstm | LSTM | 209 | 87 | 122 | 41.63% | 41.63% | 41.63% | 8.37 pp | -35 | 17 | -2.06 |
| Consolidated Hourly | transformer | Transformer | 179 | 76 | 103 | 42.46% | 42.46% | 42.46% | 7.54 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 179 | 76 | 103 | 42.46% | 42.46% | 42.46% | 7.54 pp | -27 | 13 | -2.08 |
| BTC Market Hours Daily | xgb | XGBoost | 209 | 86 | 123 | 41.15% | 41.15% | 41.15% | 8.85 pp | -37 | 17 | -2.18 |
| Consolidated Market Hours | transformer | Transformer | 53 | 21 | 32 | 39.62% | 39.62% | 39.62% | 10.38 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 5 | -2.40 |
| BTC Market Hours Daily | lstm | LSTM | 209 | 84 | 125 | 40.19% | 40.19% | 40.19% | 9.81 pp | -41 | 17 | -2.41 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |
| Consolidated Market Hours | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |
| Consolidated Market Hours Daily | nn | NN | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |
| BTC Hourly | nn | NN | 185 | 79 | 106 | 42.70% | 42.70% | 42.70% | 7.30 pp | -27 | 8 | -3.38 |
| BTC Daily | transformer | Transformer | 211 | 88 | 123 | 41.71% | 41.71% | 41.71% | 8.29 pp | -35 | 10 | -3.50 |
| BTC Hourly | rf | RandomForest | 185 | 78 | 107 | 42.16% | 42.16% | 42.16% | 7.84 pp | -29 | 8 | -3.62 |
| BTC Daily | rf | RandomForest | 211 | 81 | 130 | 38.39% | 38.39% | 38.39% | 11.61 pp | -49 | 10 | -4.90 |
| BTC Daily | xgb | XGBoost | 221 | 80 | 141 | 36.20% | 36.20% | 36.20% | 13.80 pp | -61 | 11 | -5.55 |
| BTC Hourly | xgb | XGBoost | 185 | 69 | 116 | 37.30% | 37.30% | 37.30% | 12.70 pp | -47 | 8 | -5.88 |
| BTC Hourly | lstm | LSTM | 185 | 68 | 117 | 36.76% | 36.76% | 36.76% | 13.24 pp | -49 | 8 | -6.12 |
| BTC Daily | lstm | LSTM | 211 | 71 | 140 | 33.65% | 33.65% | 33.65% | 16.35 pp | -69 | 10 | -6.90 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 185 | 93 | 92 | 50.27% | 50.27% | 50.27% | 0.27 pp | 1 | 8 | 0.12 |
| BTC Hourly | transformer | Transformer | 185 | 86 | 99 | 46.49% | 46.49% | 46.49% | 3.51 pp | -13 | 8 | -1.62 |
| BTC Hourly | nn | NN | 185 | 79 | 106 | 42.70% | 42.70% | 42.70% | 7.30 pp | -27 | 8 | -3.38 |
| BTC Hourly | rf | RandomForest | 185 | 78 | 107 | 42.16% | 42.16% | 42.16% | 7.84 pp | -29 | 8 | -3.62 |
| BTC Hourly | xgb | XGBoost | 185 | 69 | 116 | 37.30% | 37.30% | 37.30% | 12.70 pp | -47 | 8 | -5.88 |
| BTC Hourly | lstm | LSTM | 185 | 68 | 117 | 36.76% | 36.76% | 36.76% | 13.24 pp | -49 | 8 | -6.12 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 211 | 101 | 110 | 47.87% | 47.87% | 47.87% | 2.13 pp | -9 | 10 | -0.90 |
| BTC Daily | nn | NN | 211 | 96 | 115 | 45.50% | 45.50% | 45.50% | 4.50 pp | -19 | 10 | -1.90 |
| BTC Daily | transformer | Transformer | 211 | 88 | 123 | 41.71% | 41.71% | 41.71% | 8.29 pp | -35 | 10 | -3.50 |
| BTC Daily | rf | RandomForest | 211 | 81 | 130 | 38.39% | 38.39% | 38.39% | 11.61 pp | -49 | 10 | -4.90 |
| BTC Daily | xgb | XGBoost | 221 | 80 | 141 | 36.20% | 36.20% | 36.20% | 13.80 pp | -61 | 11 | -5.55 |
| BTC Daily | lstm | LSTM | 211 | 71 | 140 | 33.65% | 33.65% | 33.65% | 16.35 pp | -69 | 10 | -6.90 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 209 | 108 | 101 | 51.67% | 51.67% | 51.67% | 1.67 pp | 7 | 17 | 0.41 |
| BTC Market Hours | transformer | Transformer | 209 | 103 | 106 | 49.28% | 49.28% | 49.28% | 0.72 pp | -3 | 17 | -0.18 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 209 | 98 | 111 | 46.89% | 46.89% | 46.89% | 3.11 pp | -13 | 17 | -0.76 |
| BTC Market Hours | rf | RandomForest | 209 | 96 | 113 | 45.93% | 45.93% | 45.93% | 4.07 pp | -17 | 17 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 17 | -1.47 |
| BTC Market Hours | lstm | LSTM | 209 | 87 | 122 | 41.63% | 41.63% | 41.63% | 8.37 pp | -35 | 17 | -2.06 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 209 | 108 | 101 | 51.67% | 51.67% | 51.67% | 1.67 pp | 7 | 17 | 0.41 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 209 | 103 | 106 | 49.28% | 49.28% | 49.28% | 0.72 pp | -3 | 17 | -0.18 |
| BTC Market Hours Daily | nn | NN | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 17 | -0.53 |
| BTC Market Hours Daily | rf | RandomForest | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 17 | -1.47 |
| BTC Market Hours Daily | xgb | XGBoost | 209 | 86 | 123 | 41.15% | 41.15% | 41.15% | 8.85 pp | -37 | 17 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 209 | 84 | 125 | 40.19% | 40.19% | 40.19% | 9.81 pp | -41 | 17 | -2.41 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 179 | 87 | 92 | 48.60% | 48.60% | 48.60% | 1.40 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | rf | RandomForest | 179 | 86 | 93 | 48.04% | 48.04% | 48.04% | 1.96 pp | -7 | 13 | -0.54 |
| Consolidated Hourly | xgb | XGBoost | 179 | 82 | 97 | 45.81% | 45.81% | 45.81% | 4.19 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | lstm | LSTM | 179 | 81 | 98 | 45.25% | 45.25% | 45.25% | 4.75 pp | -17 | 13 | -1.31 |
| Consolidated Hourly | nn | NN | 179 | 79 | 100 | 44.13% | 44.13% | 44.13% | 5.87 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | transformer | Transformer | 179 | 76 | 103 | 42.46% | 42.46% | 42.46% | 7.54 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 179 | 87 | 92 | 48.60% | 48.60% | 48.60% | 1.40 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 179 | 86 | 93 | 48.04% | 48.04% | 48.04% | 1.96 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 179 | 82 | 97 | 45.81% | 45.81% | 45.81% | 4.19 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 179 | 81 | 98 | 45.25% | 45.25% | 45.25% | 4.75 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | nn | NN | 179 | 79 | 100 | 44.13% | 44.13% | 44.13% | 5.87 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 179 | 76 | 103 | 42.46% | 42.46% | 42.46% | 7.54 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 53 | 26 | 27 | 49.06% | 49.06% | 49.06% | 0.94 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours | lstm | LSTM | 53 | 24 | 29 | 45.28% | 45.28% | 45.28% | 4.72 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | transformer | Transformer | 53 | 21 | 32 | 39.62% | 39.62% | 39.62% | 10.38 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |
| Consolidated Market Hours | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 54 | 26 | 28 | 48.15% | 48.15% | 48.15% | 1.85 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | lstm | LSTM | 54 | 24 | 30 | 44.44% | 44.44% | 44.44% | 5.56 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 54 | 21 | 33 | 38.89% | 38.89% | 38.89% | 11.11 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |
| Consolidated Market Hours Daily | nn | NN | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 5 | -3.20 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
