# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T18:26:44.324262+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 263 | 203 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 299 | 239 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-07 17:00:00+00:00 | 428 | 227 | 201 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 17:00:00+00:00 | 428 | 227 | 201 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T20:00:00+00:00 | 197 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T20:00:00+00:00 | 197 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T20:00:00+00:00 | 197 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T20:00:00+00:00 | 198 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 227 | 118 | 109 | 51.98% | 51.98% | 51.98% | 1.98 pp | 9 | 18 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 227 | 113 | 114 | 49.78% | 49.78% | 49.78% | 0.22 pp | -1 | 19 | -0.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 203 | 101 | 102 | 49.75% | 49.75% | 49.75% | 0.25 pp | -1 | 9 | -0.11 |
| Consolidated Market Hours Daily | xgb | XGBoost | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | rf | RandomForest | 197 | 97 | 100 | 49.24% | 49.24% | 49.24% | 0.76 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 197 | 97 | 100 | 49.24% | 49.24% | 49.24% | 0.76 pp | -3 | 13 | -0.23 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 227 | 110 | 117 | 48.46% | 48.46% | 48.46% | 1.54 pp | -7 | 19 | -0.37 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 197 | 96 | 101 | 48.73% | 48.73% | 48.73% | 1.27 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 197 | 96 | 101 | 48.73% | 48.73% | 48.73% | 1.27 pp | -5 | 13 | -0.38 |
| Consolidated Market Hours | xgb | XGBoost | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | nn | NN | 227 | 108 | 119 | 47.58% | 47.58% | 47.58% | 2.42 pp | -11 | 19 | -0.58 |
| BTC Market Hours | transformer | Transformer | 227 | 108 | 119 | 47.58% | 47.58% | 47.58% | 2.42 pp | -11 | 18 | -0.61 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 227 | 107 | 120 | 47.14% | 47.14% | 47.14% | 2.86 pp | -13 | 18 | -0.72 |
| Consolidated Market Hours Daily | rf | RandomForest | 63 | 29 | 34 | 46.03% | 46.03% | 46.03% | 3.97 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 63 | 29 | 34 | 46.03% | 46.03% | 46.03% | 3.97 pp | -5 | 5 | -1.00 |
| BTC Market Hours | rf | RandomForest | 227 | 104 | 123 | 45.81% | 45.81% | 45.81% | 4.19 pp | -19 | 18 | -1.06 |
| Consolidated Hourly | xgb | XGBoost | 197 | 91 | 106 | 46.19% | 46.19% | 46.19% | 3.81 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 197 | 91 | 106 | 46.19% | 46.19% | 46.19% | 3.81 pp | -15 | 13 | -1.15 |
| Consolidated Market Hours | rf | RandomForest | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| BTC Market Hours Daily | rf | RandomForest | 227 | 101 | 126 | 44.49% | 44.49% | 44.49% | 5.51 pp | -25 | 19 | -1.32 |
| BTC Market Hours | xgb | XGBoost | 227 | 101 | 126 | 44.49% | 44.49% | 44.49% | 5.51 pp | -25 | 18 | -1.39 |
| Consolidated Market Hours Daily | lstm | LSTM | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| Consolidated Hourly | nn | NN | 197 | 89 | 108 | 45.18% | 45.18% | 45.18% | 4.82 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | nn | NN | 197 | 89 | 108 | 45.18% | 45.18% | 45.18% | 4.82 pp | -19 | 13 | -1.46 |
| Consolidated Market Hours | lstm | LSTM | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | lstm | LSTM | 197 | 88 | 109 | 44.67% | 44.67% | 44.67% | 5.33 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 197 | 88 | 109 | 44.67% | 44.67% | 44.67% | 5.33 pp | -21 | 13 | -1.62 |
| BTC Market Hours Daily | xgb | XGBoost | 227 | 96 | 131 | 42.29% | 42.29% | 42.29% | 7.71 pp | -35 | 19 | -1.84 |
| BTC Daily | mlp_sklearn | MLPClassifier | 229 | 105 | 124 | 45.85% | 45.85% | 45.85% | 4.15 pp | -19 | 10 | -1.90 |
| Consolidated Hourly | transformer | Transformer | 197 | 85 | 112 | 43.15% | 43.15% | 43.15% | 6.85 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 197 | 85 | 112 | 43.15% | 43.15% | 43.15% | 6.85 pp | -27 | 13 | -2.08 |
| BTC Market Hours | lstm | LSTM | 227 | 93 | 134 | 40.97% | 40.97% | 40.97% | 9.03 pp | -41 | 18 | -2.28 |
| BTC Hourly | transformer | Transformer | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 9 | -2.33 |
| BTC Market Hours Daily | lstm | LSTM | 227 | 91 | 136 | 40.09% | 40.09% | 40.09% | 9.91 pp | -45 | 19 | -2.37 |
| Consolidated Market Hours | nn | NN | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| BTC Daily | nn | NN | 229 | 102 | 127 | 44.54% | 44.54% | 44.54% | 5.46 pp | -25 | 10 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 62 | 24 | 38 | 38.71% | 38.71% | 38.71% | 11.29 pp | -14 | 5 | -2.80 |
| BTC Hourly | nn | NN | 203 | 86 | 117 | 42.36% | 42.36% | 42.36% | 7.64 pp | -31 | 9 | -3.44 |
| BTC Hourly | rf | RandomForest | 203 | 85 | 118 | 41.87% | 41.87% | 41.87% | 8.13 pp | -33 | 9 | -3.67 |
| BTC Daily | transformer | Transformer | 229 | 91 | 138 | 39.74% | 39.74% | 39.74% | 10.26 pp | -47 | 10 | -4.70 |
| BTC Daily | rf | RandomForest | 229 | 88 | 141 | 38.43% | 38.43% | 38.43% | 11.57 pp | -53 | 10 | -5.30 |
| BTC Hourly | lstm | LSTM | 203 | 76 | 127 | 37.44% | 37.44% | 37.44% | 12.56 pp | -51 | 9 | -5.67 |
| BTC Daily | xgb | XGBoost | 239 | 83 | 156 | 34.73% | 34.73% | 34.73% | 15.27 pp | -73 | 11 | -6.64 |
| BTC Hourly | xgb | XGBoost | 203 | 71 | 132 | 34.98% | 34.98% | 34.98% | 15.02 pp | -61 | 9 | -6.78 |
| BTC Daily | lstm | LSTM | 229 | 76 | 153 | 33.19% | 33.19% | 33.19% | 16.81 pp | -77 | 10 | -7.70 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 203 | 101 | 102 | 49.75% | 49.75% | 49.75% | 0.25 pp | -1 | 9 | -0.11 |
| BTC Hourly | transformer | Transformer | 203 | 91 | 112 | 44.83% | 44.83% | 44.83% | 5.17 pp | -21 | 9 | -2.33 |
| BTC Hourly | nn | NN | 203 | 86 | 117 | 42.36% | 42.36% | 42.36% | 7.64 pp | -31 | 9 | -3.44 |
| BTC Hourly | rf | RandomForest | 203 | 85 | 118 | 41.87% | 41.87% | 41.87% | 8.13 pp | -33 | 9 | -3.67 |
| BTC Hourly | lstm | LSTM | 203 | 76 | 127 | 37.44% | 37.44% | 37.44% | 12.56 pp | -51 | 9 | -5.67 |
| BTC Hourly | xgb | XGBoost | 203 | 71 | 132 | 34.98% | 34.98% | 34.98% | 15.02 pp | -61 | 9 | -6.78 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 229 | 105 | 124 | 45.85% | 45.85% | 45.85% | 4.15 pp | -19 | 10 | -1.90 |
| BTC Daily | nn | NN | 229 | 102 | 127 | 44.54% | 44.54% | 44.54% | 5.46 pp | -25 | 10 | -2.50 |
| BTC Daily | transformer | Transformer | 229 | 91 | 138 | 39.74% | 39.74% | 39.74% | 10.26 pp | -47 | 10 | -4.70 |
| BTC Daily | rf | RandomForest | 229 | 88 | 141 | 38.43% | 38.43% | 38.43% | 11.57 pp | -53 | 10 | -5.30 |
| BTC Daily | xgb | XGBoost | 239 | 83 | 156 | 34.73% | 34.73% | 34.73% | 15.27 pp | -73 | 11 | -6.64 |
| BTC Daily | lstm | LSTM | 229 | 76 | 153 | 33.19% | 33.19% | 33.19% | 16.81 pp | -77 | 10 | -7.70 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 227 | 118 | 109 | 51.98% | 51.98% | 51.98% | 1.98 pp | 9 | 18 | 0.50 |
| BTC Market Hours | transformer | Transformer | 227 | 108 | 119 | 47.58% | 47.58% | 47.58% | 2.42 pp | -11 | 18 | -0.61 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 227 | 107 | 120 | 47.14% | 47.14% | 47.14% | 2.86 pp | -13 | 18 | -0.72 |
| BTC Market Hours | rf | RandomForest | 227 | 104 | 123 | 45.81% | 45.81% | 45.81% | 4.19 pp | -19 | 18 | -1.06 |
| BTC Market Hours | xgb | XGBoost | 227 | 101 | 126 | 44.49% | 44.49% | 44.49% | 5.51 pp | -25 | 18 | -1.39 |
| BTC Market Hours | lstm | LSTM | 227 | 93 | 134 | 40.97% | 40.97% | 40.97% | 9.03 pp | -41 | 18 | -2.28 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 227 | 113 | 114 | 49.78% | 49.78% | 49.78% | 0.22 pp | -1 | 19 | -0.05 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 227 | 110 | 117 | 48.46% | 48.46% | 48.46% | 1.54 pp | -7 | 19 | -0.37 |
| BTC Market Hours Daily | nn | NN | 227 | 108 | 119 | 47.58% | 47.58% | 47.58% | 2.42 pp | -11 | 19 | -0.58 |
| BTC Market Hours Daily | rf | RandomForest | 227 | 101 | 126 | 44.49% | 44.49% | 44.49% | 5.51 pp | -25 | 19 | -1.32 |
| BTC Market Hours Daily | xgb | XGBoost | 227 | 96 | 131 | 42.29% | 42.29% | 42.29% | 7.71 pp | -35 | 19 | -1.84 |
| BTC Market Hours Daily | lstm | LSTM | 227 | 91 | 136 | 40.09% | 40.09% | 40.09% | 9.91 pp | -45 | 19 | -2.37 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 197 | 97 | 100 | 49.24% | 49.24% | 49.24% | 0.76 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 197 | 96 | 101 | 48.73% | 48.73% | 48.73% | 1.27 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | xgb | XGBoost | 197 | 91 | 106 | 46.19% | 46.19% | 46.19% | 3.81 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | nn | NN | 197 | 89 | 108 | 45.18% | 45.18% | 45.18% | 4.82 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | lstm | LSTM | 197 | 88 | 109 | 44.67% | 44.67% | 44.67% | 5.33 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | transformer | Transformer | 197 | 85 | 112 | 43.15% | 43.15% | 43.15% | 6.85 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 197 | 97 | 100 | 49.24% | 49.24% | 49.24% | 0.76 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 197 | 96 | 101 | 48.73% | 48.73% | 48.73% | 1.27 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 197 | 91 | 106 | 46.19% | 46.19% | 46.19% | 3.81 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | nn | NN | 197 | 89 | 108 | 45.18% | 45.18% | 45.18% | 4.82 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 197 | 88 | 109 | 44.67% | 44.67% | 44.67% | 5.33 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 197 | 85 | 112 | 43.15% | 43.15% | 43.15% | 6.85 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | rf | RandomForest | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | lstm | LSTM | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | nn | NN | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 62 | 24 | 38 | 38.71% | 38.71% | 38.71% | 11.29 pp | -14 | 5 | -2.80 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 5 | -0.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 63 | 29 | 34 | 46.03% | 46.03% | 46.03% | 3.97 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 63 | 29 | 34 | 46.03% | 46.03% | 46.03% | 3.97 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours Daily | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
