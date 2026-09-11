# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-11T20:39:11.165844+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 328 | 268 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 364 | 304 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-11 19:00:00+00:00 | 547 | 292 | 255 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-11 19:00:00+00:00 | 547 | 292 | 255 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T13:00:00+00:00 | 256 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T13:00:00+00:00 | 256 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T13:00:00+00:00 | 256 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-06-02T13:00:00+00:00 | 257 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 292 | 150 | 142 | 51.37% | 50.42% | 51.37% | 1.37 pp | 8 | 23 | 0.35 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 292 | 144 | 148 | 49.32% | 49.17% | 49.32% | 0.68 pp | -4 | 24 | -0.17 |
| BTC Market Hours Daily | nn | NN | 292 | 143 | 149 | 48.97% | 50.83% | 48.97% | 1.03 pp | -6 | 24 | -0.25 |
| BTC Market Hours Daily | transformer | Transformer | 292 | 140 | 152 | 47.95% | 47.92% | 47.95% | 2.05 pp | -12 | 24 | -0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 292 | 135 | 157 | 46.23% | 46.25% | 46.23% | 3.77 pp | -22 | 23 | -0.96 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 268 | 128 | 140 | 47.76% | 46.25% | 47.76% | 2.24 pp | -12 | 12 | -1.00 |
| BTC Market Hours | transformer | Transformer | 292 | 134 | 158 | 45.89% | 46.25% | 45.89% | 4.11 pp | -24 | 23 | -1.04 |
| Consolidated Hourly | rf | RandomForest | 256 | 119 | 137 | 46.48% | 46.67% | 46.48% | 3.52 pp | -18 | 16 | -1.12 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 256 | 119 | 137 | 46.48% | 46.67% | 46.48% | 3.52 pp | -18 | 16 | -1.12 |
| BTC Market Hours | xgb | XGBoost | 292 | 132 | 160 | 45.21% | 46.67% | 45.21% | 4.79 pp | -28 | 23 | -1.22 |
| Consolidated Market Hours | transformer | Transformer | 94 | 42 | 52 | 44.68% | 44.68% | 44.68% | 5.32 pp | -10 | 8 | -1.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 256 | 117 | 139 | 45.70% | 45.42% | 45.70% | 4.30 pp | -22 | 16 | -1.38 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 256 | 117 | 139 | 45.70% | 45.42% | 45.70% | 4.30 pp | -22 | 16 | -1.38 |
| Consolidated Market Hours Daily | transformer | Transformer | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 8 | -1.38 |
| BTC Market Hours | rf | RandomForest | 292 | 130 | 162 | 44.52% | 43.75% | 44.52% | 5.48 pp | -32 | 23 | -1.39 |
| BTC Hourly | transformer | Transformer | 268 | 125 | 143 | 46.64% | 46.67% | 46.64% | 3.36 pp | -18 | 12 | -1.50 |
| Consolidated Hourly | lstm | LSTM | 256 | 116 | 140 | 45.31% | 44.58% | 45.31% | 4.69 pp | -24 | 16 | -1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 256 | 116 | 140 | 45.31% | 44.58% | 45.31% | 4.69 pp | -24 | 16 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 95 | 41 | 54 | 43.16% | 43.16% | 43.16% | 6.84 pp | -13 | 8 | -1.62 |
| BTC Market Hours Daily | rf | RandomForest | 292 | 126 | 166 | 43.15% | 43.33% | 43.15% | 6.85 pp | -40 | 24 | -1.67 |
| BTC Market Hours Daily | xgb | XGBoost | 292 | 126 | 166 | 43.15% | 43.75% | 43.15% | 6.85 pp | -40 | 24 | -1.67 |
| BTC Daily | nn | NN | 294 | 136 | 158 | 46.26% | 45.83% | 46.26% | 3.74 pp | -22 | 13 | -1.69 |
| Consolidated Hourly | transformer | Transformer | 256 | 114 | 142 | 44.53% | 43.75% | 44.53% | 5.47 pp | -28 | 16 | -1.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 256 | 114 | 142 | 44.53% | 43.75% | 44.53% | 5.47 pp | -28 | 16 | -1.75 |
| Consolidated Market Hours | rf | RandomForest | 94 | 40 | 54 | 42.55% | 42.55% | 42.55% | 7.45 pp | -14 | 8 | -1.75 |
| BTC Market Hours Daily | lstm | LSTM | 292 | 121 | 171 | 41.44% | 44.17% | 41.44% | 8.56 pp | -50 | 24 | -2.08 |
| BTC Market Hours | lstm | LSTM | 292 | 121 | 171 | 41.44% | 42.92% | 41.44% | 8.56 pp | -50 | 23 | -2.17 |
| Consolidated Hourly | nn | NN | 256 | 110 | 146 | 42.97% | 43.75% | 42.97% | 7.03 pp | -36 | 16 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 256 | 110 | 146 | 42.97% | 43.75% | 42.97% | 7.03 pp | -36 | 16 | -2.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 294 | 131 | 163 | 44.56% | 43.33% | 44.56% | 5.44 pp | -32 | 13 | -2.46 |
| Consolidated Hourly | xgb | XGBoost | 256 | 108 | 148 | 42.19% | 41.67% | 42.19% | 7.81 pp | -40 | 16 | -2.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 256 | 108 | 148 | 42.19% | 41.67% | 42.19% | 7.81 pp | -40 | 16 | -2.50 |
| Consolidated Market Hours Daily | nn | NN | 95 | 37 | 58 | 38.95% | 38.95% | 38.95% | 11.05 pp | -21 | 8 | -2.62 |
| Consolidated Market Hours | nn | NN | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours | xgb | XGBoost | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | xgb | XGBoost | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 94 | 35 | 59 | 37.23% | 37.23% | 37.23% | 12.77 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours | lstm | LSTM | 94 | 33 | 61 | 35.11% | 35.11% | 35.11% | 14.89 pp | -28 | 8 | -3.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 95 | 33 | 62 | 34.74% | 34.74% | 34.74% | 15.26 pp | -29 | 8 | -3.62 |
| BTC Hourly | nn | NN | 268 | 111 | 157 | 41.42% | 40.42% | 41.42% | 8.58 pp | -46 | 12 | -3.83 |
| BTC Daily | transformer | Transformer | 294 | 119 | 175 | 40.48% | 38.33% | 40.48% | 9.52 pp | -56 | 13 | -4.31 |
| BTC Hourly | rf | RandomForest | 268 | 107 | 161 | 39.93% | 40.42% | 39.93% | 10.07 pp | -54 | 12 | -4.50 |
| BTC Daily | rf | RandomForest | 294 | 113 | 181 | 38.44% | 37.50% | 38.44% | 11.56 pp | -68 | 13 | -5.23 |
| BTC Daily | xgb | XGBoost | 304 | 114 | 190 | 37.50% | 37.92% | 37.50% | 12.50 pp | -76 | 14 | -5.43 |
| BTC Hourly | lstm | LSTM | 268 | 96 | 172 | 35.82% | 34.58% | 35.82% | 14.18 pp | -76 | 12 | -6.33 |
| BTC Daily | lstm | LSTM | 294 | 105 | 189 | 35.71% | 36.25% | 35.71% | 14.29 pp | -84 | 13 | -6.46 |
| BTC Hourly | xgb | XGBoost | 268 | 94 | 174 | 35.07% | 35.00% | 35.07% | 14.93 pp | -80 | 12 | -6.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 268 | 128 | 140 | 47.76% | 46.25% | 47.76% | 2.24 pp | -12 | 12 | -1.00 |
| BTC Hourly | transformer | Transformer | 268 | 125 | 143 | 46.64% | 46.67% | 46.64% | 3.36 pp | -18 | 12 | -1.50 |
| BTC Hourly | nn | NN | 268 | 111 | 157 | 41.42% | 40.42% | 41.42% | 8.58 pp | -46 | 12 | -3.83 |
| BTC Hourly | rf | RandomForest | 268 | 107 | 161 | 39.93% | 40.42% | 39.93% | 10.07 pp | -54 | 12 | -4.50 |
| BTC Hourly | lstm | LSTM | 268 | 96 | 172 | 35.82% | 34.58% | 35.82% | 14.18 pp | -76 | 12 | -6.33 |
| BTC Hourly | xgb | XGBoost | 268 | 94 | 174 | 35.07% | 35.00% | 35.07% | 14.93 pp | -80 | 12 | -6.67 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | nn | NN | 294 | 136 | 158 | 46.26% | 45.83% | 46.26% | 3.74 pp | -22 | 13 | -1.69 |
| BTC Daily | mlp_sklearn | MLPClassifier | 294 | 131 | 163 | 44.56% | 43.33% | 44.56% | 5.44 pp | -32 | 13 | -2.46 |
| BTC Daily | transformer | Transformer | 294 | 119 | 175 | 40.48% | 38.33% | 40.48% | 9.52 pp | -56 | 13 | -4.31 |
| BTC Daily | rf | RandomForest | 294 | 113 | 181 | 38.44% | 37.50% | 38.44% | 11.56 pp | -68 | 13 | -5.23 |
| BTC Daily | xgb | XGBoost | 304 | 114 | 190 | 37.50% | 37.92% | 37.50% | 12.50 pp | -76 | 14 | -5.43 |
| BTC Daily | lstm | LSTM | 294 | 105 | 189 | 35.71% | 36.25% | 35.71% | 14.29 pp | -84 | 13 | -6.46 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 292 | 150 | 142 | 51.37% | 50.42% | 51.37% | 1.37 pp | 8 | 23 | 0.35 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 292 | 135 | 157 | 46.23% | 46.25% | 46.23% | 3.77 pp | -22 | 23 | -0.96 |
| BTC Market Hours | transformer | Transformer | 292 | 134 | 158 | 45.89% | 46.25% | 45.89% | 4.11 pp | -24 | 23 | -1.04 |
| BTC Market Hours | xgb | XGBoost | 292 | 132 | 160 | 45.21% | 46.67% | 45.21% | 4.79 pp | -28 | 23 | -1.22 |
| BTC Market Hours | rf | RandomForest | 292 | 130 | 162 | 44.52% | 43.75% | 44.52% | 5.48 pp | -32 | 23 | -1.39 |
| BTC Market Hours | lstm | LSTM | 292 | 121 | 171 | 41.44% | 42.92% | 41.44% | 8.56 pp | -50 | 23 | -2.17 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 292 | 144 | 148 | 49.32% | 49.17% | 49.32% | 0.68 pp | -4 | 24 | -0.17 |
| BTC Market Hours Daily | nn | NN | 292 | 143 | 149 | 48.97% | 50.83% | 48.97% | 1.03 pp | -6 | 24 | -0.25 |
| BTC Market Hours Daily | transformer | Transformer | 292 | 140 | 152 | 47.95% | 47.92% | 47.95% | 2.05 pp | -12 | 24 | -0.50 |
| BTC Market Hours Daily | rf | RandomForest | 292 | 126 | 166 | 43.15% | 43.33% | 43.15% | 6.85 pp | -40 | 24 | -1.67 |
| BTC Market Hours Daily | xgb | XGBoost | 292 | 126 | 166 | 43.15% | 43.75% | 43.15% | 6.85 pp | -40 | 24 | -1.67 |
| BTC Market Hours Daily | lstm | LSTM | 292 | 121 | 171 | 41.44% | 44.17% | 41.44% | 8.56 pp | -50 | 24 | -2.08 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 256 | 119 | 137 | 46.48% | 46.67% | 46.48% | 3.52 pp | -18 | 16 | -1.12 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 256 | 117 | 139 | 45.70% | 45.42% | 45.70% | 4.30 pp | -22 | 16 | -1.38 |
| Consolidated Hourly | lstm | LSTM | 256 | 116 | 140 | 45.31% | 44.58% | 45.31% | 4.69 pp | -24 | 16 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 256 | 114 | 142 | 44.53% | 43.75% | 44.53% | 5.47 pp | -28 | 16 | -1.75 |
| Consolidated Hourly | nn | NN | 256 | 110 | 146 | 42.97% | 43.75% | 42.97% | 7.03 pp | -36 | 16 | -2.25 |
| Consolidated Hourly | xgb | XGBoost | 256 | 108 | 148 | 42.19% | 41.67% | 42.19% | 7.81 pp | -40 | 16 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 256 | 119 | 137 | 46.48% | 46.67% | 46.48% | 3.52 pp | -18 | 16 | -1.12 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 256 | 117 | 139 | 45.70% | 45.42% | 45.70% | 4.30 pp | -22 | 16 | -1.38 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 256 | 116 | 140 | 45.31% | 44.58% | 45.31% | 4.69 pp | -24 | 16 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 256 | 114 | 142 | 44.53% | 43.75% | 44.53% | 5.47 pp | -28 | 16 | -1.75 |
| Consolidated Daily/Hourly Refresh | nn | NN | 256 | 110 | 146 | 42.97% | 43.75% | 42.97% | 7.03 pp | -36 | 16 | -2.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 256 | 108 | 148 | 42.19% | 41.67% | 42.19% | 7.81 pp | -40 | 16 | -2.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 94 | 42 | 52 | 44.68% | 44.68% | 44.68% | 5.32 pp | -10 | 8 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 94 | 40 | 54 | 42.55% | 42.55% | 42.55% | 7.45 pp | -14 | 8 | -1.75 |
| Consolidated Market Hours | nn | NN | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours | xgb | XGBoost | 94 | 36 | 58 | 38.30% | 38.30% | 38.30% | 11.70 pp | -22 | 8 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 94 | 35 | 59 | 37.23% | 37.23% | 37.23% | 12.77 pp | -24 | 8 | -3.00 |
| Consolidated Market Hours | lstm | LSTM | 94 | 33 | 61 | 35.11% | 35.11% | 35.11% | 14.89 pp | -28 | 8 | -3.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 95 | 42 | 53 | 44.21% | 44.21% | 44.21% | 5.79 pp | -11 | 8 | -1.38 |
| Consolidated Market Hours Daily | rf | RandomForest | 95 | 41 | 54 | 43.16% | 43.16% | 43.16% | 6.84 pp | -13 | 8 | -1.62 |
| Consolidated Market Hours Daily | nn | NN | 95 | 37 | 58 | 38.95% | 38.95% | 38.95% | 11.05 pp | -21 | 8 | -2.62 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | xgb | XGBoost | 95 | 36 | 59 | 37.89% | 37.89% | 37.89% | 12.11 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours Daily | lstm | LSTM | 95 | 33 | 62 | 34.74% | 34.74% | 34.74% | 15.26 pp | -29 | 8 | -3.62 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
