# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-07T19:02:03.723081+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 264 | 204 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 300 | 240 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-07 18:00:00+00:00 | 430 | 228 | 202 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-07 18:00:00+00:00 | 429 | 227 | 202 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 20:00:00+00:00 | 197 | 197 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 20:00:00+00:00 | 197 | 197 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 20:00:00+00:00 | 197 | 62 | 135 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-30 20:00:00+00:00 | 197 | 62 | 135 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 228 | 119 | 109 | 52.19% | 52.19% | 52.19% | 2.19 pp | 10 | 18 | 0.56 |
| Consolidated Hourly | rf | RandomForest | 197 | 99 | 98 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 197 | 99 | 98 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 204 | 102 | 102 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 227 | 113 | 114 | 49.78% | 49.78% | 49.78% | 0.22 pp | -1 | 19 | -0.05 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 227 | 110 | 117 | 48.46% | 48.46% | 48.46% | 1.54 pp | -7 | 19 | -0.37 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 197 | 96 | 101 | 48.73% | 48.73% | 48.73% | 1.27 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 197 | 96 | 101 | 48.73% | 48.73% | 48.73% | 1.27 pp | -5 | 13 | -0.38 |
| Consolidated Market Hours | xgb | XGBoost | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | xgb | XGBoost | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | nn | NN | 227 | 108 | 119 | 47.58% | 47.58% | 47.58% | 2.42 pp | -11 | 19 | -0.58 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 228 | 108 | 120 | 47.37% | 47.37% | 47.37% | 2.63 pp | -12 | 18 | -0.67 |
| BTC Market Hours | transformer | Transformer | 228 | 108 | 120 | 47.37% | 47.37% | 47.37% | 2.63 pp | -12 | 18 | -0.67 |
| BTC Market Hours | rf | RandomForest | 228 | 105 | 123 | 46.05% | 46.05% | 46.05% | 3.95 pp | -18 | 18 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | rf | RandomForest | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | lstm | LSTM | 197 | 90 | 107 | 45.69% | 45.69% | 45.69% | 4.31 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 197 | 90 | 107 | 45.69% | 45.69% | 45.69% | 4.31 pp | -17 | 13 | -1.31 |
| BTC Market Hours Daily | rf | RandomForest | 227 | 101 | 126 | 44.49% | 44.49% | 44.49% | 5.51 pp | -25 | 19 | -1.32 |
| BTC Market Hours | xgb | XGBoost | 228 | 102 | 126 | 44.74% | 44.74% | 44.74% | 5.26 pp | -24 | 18 | -1.33 |
| Consolidated Market Hours | lstm | LSTM | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | transformer | Transformer | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| BTC Daily | mlp_sklearn | MLPClassifier | 230 | 106 | 124 | 46.09% | 46.09% | 46.09% | 3.91 pp | -18 | 10 | -1.80 |
| BTC Market Hours Daily | xgb | XGBoost | 227 | 96 | 131 | 42.29% | 42.29% | 42.29% | 7.71 pp | -35 | 19 | -1.84 |
| Consolidated Hourly | nn | NN | 197 | 85 | 112 | 43.15% | 43.15% | 43.15% | 6.85 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 197 | 85 | 112 | 43.15% | 43.15% | 43.15% | 6.85 pp | -27 | 13 | -2.08 |
| BTC Hourly | transformer | Transformer | 204 | 92 | 112 | 45.10% | 45.10% | 45.10% | 4.90 pp | -20 | 9 | -2.22 |
| BTC Market Hours | lstm | LSTM | 228 | 94 | 134 | 41.23% | 41.23% | 41.23% | 8.77 pp | -40 | 18 | -2.22 |
| BTC Market Hours Daily | lstm | LSTM | 227 | 91 | 136 | 40.09% | 40.09% | 40.09% | 9.91 pp | -45 | 19 | -2.37 |
| BTC Daily | nn | NN | 230 | 103 | 127 | 44.78% | 44.78% | 44.78% | 5.22 pp | -24 | 10 | -2.40 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| BTC Hourly | nn | NN | 204 | 86 | 118 | 42.16% | 42.16% | 42.16% | 7.84 pp | -32 | 9 | -3.56 |
| BTC Hourly | rf | RandomForest | 204 | 85 | 119 | 41.67% | 41.67% | 41.67% | 8.33 pp | -34 | 9 | -3.78 |
| BTC Daily | transformer | Transformer | 230 | 92 | 138 | 40.00% | 40.00% | 40.00% | 10.00 pp | -46 | 10 | -4.60 |
| BTC Daily | rf | RandomForest | 230 | 88 | 142 | 38.26% | 38.26% | 38.26% | 11.74 pp | -54 | 10 | -5.40 |
| BTC Hourly | lstm | LSTM | 204 | 76 | 128 | 37.25% | 37.25% | 37.25% | 12.75 pp | -52 | 9 | -5.78 |
| BTC Daily | xgb | XGBoost | 240 | 84 | 156 | 35.00% | 35.00% | 35.00% | 15.00 pp | -72 | 11 | -6.55 |
| BTC Hourly | xgb | XGBoost | 204 | 71 | 133 | 34.80% | 34.80% | 34.80% | 15.20 pp | -62 | 9 | -6.89 |
| BTC Daily | lstm | LSTM | 230 | 76 | 154 | 33.04% | 33.04% | 33.04% | 16.96 pp | -78 | 10 | -7.80 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 204 | 102 | 102 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| BTC Hourly | transformer | Transformer | 204 | 92 | 112 | 45.10% | 45.10% | 45.10% | 4.90 pp | -20 | 9 | -2.22 |
| BTC Hourly | nn | NN | 204 | 86 | 118 | 42.16% | 42.16% | 42.16% | 7.84 pp | -32 | 9 | -3.56 |
| BTC Hourly | rf | RandomForest | 204 | 85 | 119 | 41.67% | 41.67% | 41.67% | 8.33 pp | -34 | 9 | -3.78 |
| BTC Hourly | lstm | LSTM | 204 | 76 | 128 | 37.25% | 37.25% | 37.25% | 12.75 pp | -52 | 9 | -5.78 |
| BTC Hourly | xgb | XGBoost | 204 | 71 | 133 | 34.80% | 34.80% | 34.80% | 15.20 pp | -62 | 9 | -6.89 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 230 | 106 | 124 | 46.09% | 46.09% | 46.09% | 3.91 pp | -18 | 10 | -1.80 |
| BTC Daily | nn | NN | 230 | 103 | 127 | 44.78% | 44.78% | 44.78% | 5.22 pp | -24 | 10 | -2.40 |
| BTC Daily | transformer | Transformer | 230 | 92 | 138 | 40.00% | 40.00% | 40.00% | 10.00 pp | -46 | 10 | -4.60 |
| BTC Daily | rf | RandomForest | 230 | 88 | 142 | 38.26% | 38.26% | 38.26% | 11.74 pp | -54 | 10 | -5.40 |
| BTC Daily | xgb | XGBoost | 240 | 84 | 156 | 35.00% | 35.00% | 35.00% | 15.00 pp | -72 | 11 | -6.55 |
| BTC Daily | lstm | LSTM | 230 | 76 | 154 | 33.04% | 33.04% | 33.04% | 16.96 pp | -78 | 10 | -7.80 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 228 | 119 | 109 | 52.19% | 52.19% | 52.19% | 2.19 pp | 10 | 18 | 0.56 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 228 | 108 | 120 | 47.37% | 47.37% | 47.37% | 2.63 pp | -12 | 18 | -0.67 |
| BTC Market Hours | transformer | Transformer | 228 | 108 | 120 | 47.37% | 47.37% | 47.37% | 2.63 pp | -12 | 18 | -0.67 |
| BTC Market Hours | rf | RandomForest | 228 | 105 | 123 | 46.05% | 46.05% | 46.05% | 3.95 pp | -18 | 18 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 228 | 102 | 126 | 44.74% | 44.74% | 44.74% | 5.26 pp | -24 | 18 | -1.33 |
| BTC Market Hours | lstm | LSTM | 228 | 94 | 134 | 41.23% | 41.23% | 41.23% | 8.77 pp | -40 | 18 | -2.22 |

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
| Consolidated Hourly | rf | RandomForest | 197 | 99 | 98 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 197 | 96 | 101 | 48.73% | 48.73% | 48.73% | 1.27 pp | -5 | 13 | -0.38 |
| Consolidated Hourly | lstm | LSTM | 197 | 90 | 107 | 45.69% | 45.69% | 45.69% | 4.31 pp | -17 | 13 | -1.31 |
| Consolidated Hourly | transformer | Transformer | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | xgb | XGBoost | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| Consolidated Hourly | nn | NN | 197 | 85 | 112 | 43.15% | 43.15% | 43.15% | 6.85 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 197 | 99 | 98 | 50.25% | 50.25% | 50.25% | 0.25 pp | 1 | 13 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 197 | 96 | 101 | 48.73% | 48.73% | 48.73% | 1.27 pp | -5 | 13 | -0.38 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 197 | 90 | 107 | 45.69% | 45.69% | 45.69% | 4.31 pp | -17 | 13 | -1.31 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 197 | 87 | 110 | 44.16% | 44.16% | 44.16% | 5.84 pp | -23 | 13 | -1.77 |
| Consolidated Daily/Hourly Refresh | nn | NN | 197 | 85 | 112 | 43.15% | 43.15% | 43.15% | 6.85 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours | rf | RandomForest | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | transformer | Transformer | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours | lstm | LSTM | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours | nn | NN | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 5 | -0.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | transformer | Transformer | 62 | 28 | 34 | 45.16% | 45.16% | 45.16% | 4.84 pp | -6 | 5 | -1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |
| Consolidated Market Hours Daily | nn | NN | 62 | 25 | 37 | 40.32% | 40.32% | 40.32% | 9.68 pp | -12 | 5 | -2.40 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
