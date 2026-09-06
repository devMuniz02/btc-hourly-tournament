# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-06T16:28:15.580832+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 246 | 186 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 281 | 221 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-06 15:00:00+00:00 | 395 | 209 | 186 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-06 15:00:00+00:00 | 395 | 209 | 186 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 00:00:00+00:00 | 179 | 179 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 00:00:00+00:00 | 179 | 179 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 00:00:00+00:00 | 179 | 53 | 126 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 00:00:00+00:00 | 179 | 53 | 126 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 209 | 108 | 101 | 51.67% | 51.67% | 51.67% | 1.67 pp | 7 | 17 | 0.41 |
| BTC Market Hours Daily | transformer | Transformer | 209 | 108 | 101 | 51.67% | 51.67% | 51.67% | 1.67 pp | 7 | 17 | 0.41 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 186 | 93 | 93 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Market Hours | transformer | Transformer | 209 | 103 | 106 | 49.28% | 49.28% | 49.28% | 0.72 pp | -3 | 17 | -0.18 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 209 | 103 | 106 | 49.28% | 49.28% | 49.28% | 0.72 pp | -3 | 17 | -0.18 |
| Consolidated Market Hours | xgb | XGBoost | 53 | 26 | 27 | 49.06% | 49.06% | 49.06% | 0.94 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | xgb | XGBoost | 53 | 26 | 27 | 49.06% | 49.06% | 49.06% | 0.94 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | rf | RandomForest | 179 | 88 | 91 | 49.16% | 49.16% | 49.16% | 0.84 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 179 | 88 | 91 | 49.16% | 49.16% | 49.16% | 0.84 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 179 | 87 | 92 | 48.60% | 48.60% | 48.60% | 1.40 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 179 | 87 | 92 | 48.60% | 48.60% | 48.60% | 1.40 pp | -5 | 13 | -0.38 |
| BTC Market Hours Daily | nn | NN | 209 | 100 | 109 | 47.85% | 47.85% | 47.85% | 2.15 pp | -9 | 17 | -0.53 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 209 | 98 | 111 | 46.89% | 46.89% | 46.89% | 3.11 pp | -13 | 17 | -0.76 |
| BTC Daily | mlp_sklearn | MLPClassifier | 211 | 101 | 110 | 47.87% | 47.87% | 47.87% | 2.13 pp | -9 | 10 | -0.90 |
| Consolidated Hourly | lstm | LSTM | 179 | 83 | 96 | 46.37% | 46.37% | 46.37% | 3.63 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 179 | 83 | 96 | 46.37% | 46.37% | 46.37% | 3.63 pp | -13 | 13 | -1.00 |
| BTC Market Hours | rf | RandomForest | 209 | 96 | 113 | 45.93% | 45.93% | 45.93% | 4.07 pp | -17 | 17 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 53 | 24 | 29 | 45.28% | 45.28% | 45.28% | 4.72 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 53 | 24 | 29 | 45.28% | 45.28% | 45.28% | 4.72 pp | -5 | 5 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 17 | -1.47 |
| BTC Market Hours Daily | rf | RandomForest | 209 | 92 | 117 | 44.02% | 44.02% | 44.02% | 5.98 pp | -25 | 17 | -1.47 |
| BTC Hourly | transformer | Transformer | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 8 | -1.75 |
| Consolidated Hourly | transformer | Transformer | 179 | 78 | 101 | 43.58% | 43.58% | 43.58% | 6.42 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 179 | 78 | 101 | 43.58% | 43.58% | 43.58% | 6.42 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 179 | 78 | 101 | 43.58% | 43.58% | 43.58% | 6.42 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 179 | 78 | 101 | 43.58% | 43.58% | 43.58% | 6.42 pp | -23 | 13 | -1.77 |
| Consolidated Market Hours | rf | RandomForest | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | rf | RandomForest | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 5 | -1.80 |
| BTC Market Hours | lstm | LSTM | 209 | 87 | 122 | 41.63% | 41.63% | 41.63% | 8.37 pp | -35 | 17 | -2.06 |
| BTC Daily | nn | NN | 211 | 95 | 116 | 45.02% | 45.02% | 45.02% | 4.98 pp | -21 | 10 | -2.10 |
| BTC Market Hours Daily | xgb | XGBoost | 209 | 86 | 123 | 41.15% | 41.15% | 41.15% | 8.85 pp | -37 | 17 | -2.18 |
| Consolidated Market Hours | transformer | Transformer | 53 | 21 | 32 | 39.62% | 39.62% | 39.62% | 10.38 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 53 | 21 | 32 | 39.62% | 39.62% | 39.62% | 10.38 pp | -11 | 5 | -2.20 |
| Consolidated Hourly | nn | NN | 179 | 75 | 104 | 41.90% | 41.90% | 41.90% | 8.10 pp | -29 | 13 | -2.23 |
| Consolidated Daily/Hourly Refresh | nn | NN | 179 | 75 | 104 | 41.90% | 41.90% | 41.90% | 8.10 pp | -29 | 13 | -2.23 |
| BTC Market Hours Daily | lstm | LSTM | 209 | 84 | 125 | 40.19% | 40.19% | 40.19% | 9.81 pp | -41 | 17 | -2.41 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |
| Consolidated Market Hours | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |
| BTC Hourly | nn | NN | 186 | 80 | 106 | 43.01% | 43.01% | 43.01% | 6.99 pp | -26 | 8 | -3.25 |
| BTC Daily | transformer | Transformer | 211 | 88 | 123 | 41.71% | 41.71% | 41.71% | 8.29 pp | -35 | 10 | -3.50 |
| BTC Hourly | rf | RandomForest | 186 | 78 | 108 | 41.94% | 41.94% | 41.94% | 8.06 pp | -30 | 8 | -3.75 |
| BTC Daily | rf | RandomForest | 211 | 80 | 131 | 37.91% | 37.91% | 37.91% | 12.09 pp | -51 | 10 | -5.10 |
| BTC Daily | xgb | XGBoost | 221 | 79 | 142 | 35.75% | 35.75% | 35.75% | 14.25 pp | -63 | 11 | -5.73 |
| BTC Hourly | xgb | XGBoost | 186 | 69 | 117 | 37.10% | 37.10% | 37.10% | 12.90 pp | -48 | 8 | -6.00 |
| BTC Hourly | lstm | LSTM | 186 | 68 | 118 | 36.56% | 36.56% | 36.56% | 13.44 pp | -50 | 8 | -6.25 |
| BTC Daily | lstm | LSTM | 211 | 71 | 140 | 33.65% | 33.65% | 33.65% | 16.35 pp | -69 | 10 | -6.90 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 186 | 93 | 93 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 8 | 0.00 |
| BTC Hourly | transformer | Transformer | 186 | 86 | 100 | 46.24% | 46.24% | 46.24% | 3.76 pp | -14 | 8 | -1.75 |
| BTC Hourly | nn | NN | 186 | 80 | 106 | 43.01% | 43.01% | 43.01% | 6.99 pp | -26 | 8 | -3.25 |
| BTC Hourly | rf | RandomForest | 186 | 78 | 108 | 41.94% | 41.94% | 41.94% | 8.06 pp | -30 | 8 | -3.75 |
| BTC Hourly | xgb | XGBoost | 186 | 69 | 117 | 37.10% | 37.10% | 37.10% | 12.90 pp | -48 | 8 | -6.00 |
| BTC Hourly | lstm | LSTM | 186 | 68 | 118 | 36.56% | 36.56% | 36.56% | 13.44 pp | -50 | 8 | -6.25 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 211 | 101 | 110 | 47.87% | 47.87% | 47.87% | 2.13 pp | -9 | 10 | -0.90 |
| BTC Daily | nn | NN | 211 | 95 | 116 | 45.02% | 45.02% | 45.02% | 4.98 pp | -21 | 10 | -2.10 |
| BTC Daily | transformer | Transformer | 211 | 88 | 123 | 41.71% | 41.71% | 41.71% | 8.29 pp | -35 | 10 | -3.50 |
| BTC Daily | rf | RandomForest | 211 | 80 | 131 | 37.91% | 37.91% | 37.91% | 12.09 pp | -51 | 10 | -5.10 |
| BTC Daily | xgb | XGBoost | 221 | 79 | 142 | 35.75% | 35.75% | 35.75% | 14.25 pp | -63 | 11 | -5.73 |
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
| Consolidated Hourly | rf | RandomForest | 179 | 88 | 91 | 49.16% | 49.16% | 49.16% | 0.84 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 179 | 87 | 92 | 48.60% | 48.60% | 48.60% | 1.40 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | lstm | LSTM | 179 | 83 | 96 | 46.37% | 46.37% | 46.37% | 3.63 pp | -13 | 13 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 179 | 78 | 101 | 43.58% | 43.58% | 43.58% | 6.42 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 179 | 78 | 101 | 43.58% | 43.58% | 43.58% | 6.42 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 179 | 75 | 104 | 41.90% | 41.90% | 41.90% | 8.10 pp | -29 | 13 | -2.23 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 179 | 88 | 91 | 49.16% | 49.16% | 49.16% | 0.84 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 179 | 87 | 92 | 48.60% | 48.60% | 48.60% | 1.40 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 179 | 83 | 96 | 46.37% | 46.37% | 46.37% | 3.63 pp | -13 | 13 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 179 | 78 | 101 | 43.58% | 43.58% | 43.58% | 6.42 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 179 | 78 | 101 | 43.58% | 43.58% | 43.58% | 6.42 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 179 | 75 | 104 | 41.90% | 41.90% | 41.90% | 8.10 pp | -29 | 13 | -2.23 |

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
| Consolidated Market Hours Daily | xgb | XGBoost | 53 | 26 | 27 | 49.06% | 49.06% | 49.06% | 0.94 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 53 | 24 | 29 | 45.28% | 45.28% | 45.28% | 4.72 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 53 | 22 | 31 | 41.51% | 41.51% | 41.51% | 8.49 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours Daily | transformer | Transformer | 53 | 21 | 32 | 39.62% | 39.62% | 39.62% | 10.38 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 53 | 19 | 34 | 35.85% | 35.85% | 35.85% | 14.15 pp | -15 | 5 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
