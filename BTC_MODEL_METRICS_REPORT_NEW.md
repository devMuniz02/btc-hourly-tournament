# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T09:30:22.955735+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 257 | 197 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 293 | 233 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 416 | 221 | 195 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 00:00:00+00:00 | 416 | 221 | 195 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T17:00:00+00:00 | 191 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T17:00:00+00:00 | 191 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T17:00:00+00:00 | 191 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T17:00:00+00:00 | 192 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 221 | 115 | 106 | 52.04% | 52.04% | 52.04% | 2.04 pp | 9 | 17 | 0.53 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 197 | 99 | 98 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Market Hours | xgb | XGBoost | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 5 | -0.20 |
| BTC Market Hours Daily | nn | NN | 221 | 108 | 113 | 48.87% | 48.87% | 48.87% | 1.13 pp | -5 | 18 | -0.28 |
| Consolidated Hourly | rf | RandomForest | 191 | 93 | 98 | 48.69% | 48.69% | 48.69% | 1.31 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 191 | 93 | 98 | 48.69% | 48.69% | 48.69% | 1.31 pp | -5 | 13 | -0.38 |
| BTC Market Hours Daily | transformer | Transformer | 221 | 107 | 114 | 48.42% | 48.42% | 48.42% | 1.58 pp | -7 | 18 | -0.39 |
| Consolidated Market Hours Daily | xgb | XGBoost | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 221 | 105 | 116 | 47.51% | 47.51% | 47.51% | 2.49 pp | -11 | 18 | -0.61 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 221 | 105 | 116 | 47.51% | 47.51% | 47.51% | 2.49 pp | -11 | 17 | -0.65 |
| BTC Market Hours | rf | RandomForest | 221 | 105 | 116 | 47.51% | 47.51% | 47.51% | 2.49 pp | -11 | 17 | -0.65 |
| BTC Market Hours Daily | rf | RandomForest | 221 | 102 | 119 | 46.15% | 46.15% | 46.15% | 3.85 pp | -17 | 18 | -0.94 |
| BTC Market Hours | transformer | Transformer | 221 | 101 | 120 | 45.70% | 45.70% | 45.70% | 4.30 pp | -19 | 17 | -1.12 |
| Consolidated Hourly | xgb | XGBoost | 191 | 88 | 103 | 46.07% | 46.07% | 46.07% | 3.93 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 191 | 88 | 103 | 46.07% | 46.07% | 46.07% | 3.93 pp | -15 | 13 | -1.15 |
| Consolidated Market Hours Daily | transformer | Transformer | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | lstm | LSTM | 191 | 87 | 104 | 45.55% | 45.55% | 45.55% | 4.45 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 191 | 87 | 104 | 45.55% | 45.55% | 45.55% | 4.45 pp | -17 | 13 | -1.31 |
| Consolidated Market Hours | lstm | LSTM | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | nn | NN | 191 | 86 | 105 | 45.03% | 45.03% | 45.03% | 4.97 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | nn | NN | 191 | 86 | 105 | 45.03% | 45.03% | 45.03% | 4.97 pp | -19 | 13 | -1.46 |
| Consolidated Market Hours Daily | lstm | LSTM | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| BTC Daily | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 10 | -1.70 |
| Consolidated Hourly | transformer | Transformer | 191 | 84 | 107 | 43.98% | 43.98% | 43.98% | 6.02 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 191 | 84 | 107 | 43.98% | 43.98% | 43.98% | 6.02 pp | -23 | 13 | -1.77 |
| BTC Market Hours | xgb | XGBoost | 221 | 94 | 127 | 42.53% | 42.53% | 42.53% | 7.47 pp | -33 | 17 | -1.94 |
| BTC Market Hours Daily | xgb | XGBoost | 221 | 93 | 128 | 42.08% | 42.08% | 42.08% | 7.92 pp | -35 | 18 | -1.94 |
| Consolidated Market Hours Daily | nn | NN | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours | nn | NN | 59 | 24 | 35 | 40.68% | 40.68% | 40.68% | 9.32 pp | -11 | 5 | -2.20 |
| BTC Daily | nn | NN | 223 | 100 | 123 | 44.84% | 44.84% | 44.84% | 5.16 pp | -23 | 10 | -2.30 |
| BTC Hourly | transformer | Transformer | 197 | 88 | 109 | 44.67% | 44.67% | 44.67% | 5.33 pp | -21 | 9 | -2.33 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 59 | 23 | 36 | 38.98% | 38.98% | 38.98% | 11.02 pp | -13 | 5 | -2.60 |
| BTC Hourly | nn | NN | 197 | 83 | 114 | 42.13% | 42.13% | 42.13% | 7.87 pp | -31 | 9 | -3.44 |
| BTC Hourly | rf | RandomForest | 197 | 83 | 114 | 42.13% | 42.13% | 42.13% | 7.87 pp | -31 | 9 | -3.44 |
| BTC Market Hours Daily | lstm | LSTM | 221 | 76 | 145 | 34.39% | 34.39% | 34.39% | 15.61 pp | -69 | 18 | -3.83 |
| BTC Market Hours | lstm | LSTM | 221 | 77 | 144 | 34.84% | 34.84% | 34.84% | 15.16 pp | -67 | 17 | -3.94 |
| BTC Daily | transformer | Transformer | 223 | 91 | 132 | 40.81% | 40.81% | 40.81% | 9.19 pp | -41 | 10 | -4.10 |
| BTC Daily | rf | RandomForest | 223 | 86 | 137 | 38.57% | 38.57% | 38.57% | 11.43 pp | -51 | 10 | -5.10 |
| BTC Hourly | lstm | LSTM | 197 | 74 | 123 | 37.56% | 37.56% | 37.56% | 12.44 pp | -49 | 9 | -5.44 |
| BTC Daily | xgb | XGBoost | 233 | 83 | 150 | 35.62% | 35.62% | 35.62% | 14.38 pp | -67 | 11 | -6.09 |
| BTC Hourly | xgb | XGBoost | 197 | 71 | 126 | 36.04% | 36.04% | 36.04% | 13.96 pp | -55 | 9 | -6.11 |
| BTC Daily | lstm | LSTM | 223 | 75 | 148 | 33.63% | 33.63% | 33.63% | 16.37 pp | -73 | 10 | -7.30 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 197 | 99 | 98 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 9 | 0.11 |
| BTC Hourly | transformer | Transformer | 197 | 88 | 109 | 44.67% | 44.67% | 44.67% | 5.33 pp | -21 | 9 | -2.33 |
| BTC Hourly | nn | NN | 197 | 83 | 114 | 42.13% | 42.13% | 42.13% | 7.87 pp | -31 | 9 | -3.44 |
| BTC Hourly | rf | RandomForest | 197 | 83 | 114 | 42.13% | 42.13% | 42.13% | 7.87 pp | -31 | 9 | -3.44 |
| BTC Hourly | lstm | LSTM | 197 | 74 | 123 | 37.56% | 37.56% | 37.56% | 12.44 pp | -49 | 9 | -5.44 |
| BTC Hourly | xgb | XGBoost | 197 | 71 | 126 | 36.04% | 36.04% | 36.04% | 13.96 pp | -55 | 9 | -6.11 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 223 | 103 | 120 | 46.19% | 46.19% | 46.19% | 3.81 pp | -17 | 10 | -1.70 |
| BTC Daily | nn | NN | 223 | 100 | 123 | 44.84% | 44.84% | 44.84% | 5.16 pp | -23 | 10 | -2.30 |
| BTC Daily | transformer | Transformer | 223 | 91 | 132 | 40.81% | 40.81% | 40.81% | 9.19 pp | -41 | 10 | -4.10 |
| BTC Daily | rf | RandomForest | 223 | 86 | 137 | 38.57% | 38.57% | 38.57% | 11.43 pp | -51 | 10 | -5.10 |
| BTC Daily | xgb | XGBoost | 233 | 83 | 150 | 35.62% | 35.62% | 35.62% | 14.38 pp | -67 | 11 | -6.09 |
| BTC Daily | lstm | LSTM | 223 | 75 | 148 | 33.63% | 33.63% | 33.63% | 16.37 pp | -73 | 10 | -7.30 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 221 | 115 | 106 | 52.04% | 52.04% | 52.04% | 2.04 pp | 9 | 17 | 0.53 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 221 | 105 | 116 | 47.51% | 47.51% | 47.51% | 2.49 pp | -11 | 17 | -0.65 |
| BTC Market Hours | rf | RandomForest | 221 | 105 | 116 | 47.51% | 47.51% | 47.51% | 2.49 pp | -11 | 17 | -0.65 |
| BTC Market Hours | transformer | Transformer | 221 | 101 | 120 | 45.70% | 45.70% | 45.70% | 4.30 pp | -19 | 17 | -1.12 |
| BTC Market Hours | xgb | XGBoost | 221 | 94 | 127 | 42.53% | 42.53% | 42.53% | 7.47 pp | -33 | 17 | -1.94 |
| BTC Market Hours | lstm | LSTM | 221 | 77 | 144 | 34.84% | 34.84% | 34.84% | 15.16 pp | -67 | 17 | -3.94 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 221 | 108 | 113 | 48.87% | 48.87% | 48.87% | 1.13 pp | -5 | 18 | -0.28 |
| BTC Market Hours Daily | transformer | Transformer | 221 | 107 | 114 | 48.42% | 48.42% | 48.42% | 1.58 pp | -7 | 18 | -0.39 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 221 | 105 | 116 | 47.51% | 47.51% | 47.51% | 2.49 pp | -11 | 18 | -0.61 |
| BTC Market Hours Daily | rf | RandomForest | 221 | 102 | 119 | 46.15% | 46.15% | 46.15% | 3.85 pp | -17 | 18 | -0.94 |
| BTC Market Hours Daily | xgb | XGBoost | 221 | 93 | 128 | 42.08% | 42.08% | 42.08% | 7.92 pp | -35 | 18 | -1.94 |
| BTC Market Hours Daily | lstm | LSTM | 221 | 76 | 145 | 34.39% | 34.39% | 34.39% | 15.61 pp | -69 | 18 | -3.83 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Hourly | rf | RandomForest | 191 | 93 | 98 | 48.69% | 48.69% | 48.69% | 1.31 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | xgb | XGBoost | 191 | 88 | 103 | 46.07% | 46.07% | 46.07% | 3.93 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | lstm | LSTM | 191 | 87 | 104 | 45.55% | 45.55% | 45.55% | 4.45 pp | -17 | 13 | -1.31 |
| Consolidated Hourly | nn | NN | 191 | 86 | 105 | 45.03% | 45.03% | 45.03% | 4.97 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | transformer | Transformer | 191 | 84 | 107 | 43.98% | 43.98% | 43.98% | 6.02 pp | -23 | 13 | -1.77 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 191 | 95 | 96 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 13 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 191 | 93 | 98 | 48.69% | 48.69% | 48.69% | 1.31 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 191 | 88 | 103 | 46.07% | 46.07% | 46.07% | 3.93 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 191 | 87 | 104 | 45.55% | 45.55% | 45.55% | 4.45 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | nn | NN | 191 | 86 | 105 | 45.03% | 45.03% | 45.03% | 4.97 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 191 | 84 | 107 | 43.98% | 43.98% | 43.98% | 6.02 pp | -23 | 13 | -1.77 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours | lstm | LSTM | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | rf | RandomForest | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | transformer | Transformer | 59 | 26 | 33 | 44.07% | 44.07% | 44.07% | 5.93 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | nn | NN | 59 | 24 | 35 | 40.68% | 40.68% | 40.68% | 9.32 pp | -11 | 5 | -2.20 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 59 | 23 | 36 | 38.98% | 38.98% | 38.98% | 11.02 pp | -13 | 5 | -2.60 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 60 | 29 | 31 | 48.33% | 48.33% | 48.33% | 1.67 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | transformer | Transformer | 60 | 27 | 33 | 45.00% | 45.00% | 45.00% | 5.00 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | rf | RandomForest | 60 | 26 | 34 | 43.33% | 43.33% | 43.33% | 6.67 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | nn | NN | 60 | 25 | 35 | 41.67% | 41.67% | 41.67% | 8.33 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 60 | 24 | 36 | 40.00% | 40.00% | 40.00% | 10.00 pp | -12 | 5 | -2.40 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
