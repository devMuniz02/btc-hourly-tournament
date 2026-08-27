# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T07:34:33.010251+00:00
Scope: `all`

## Source Files

- `artifacts/btc/hourly/history.csv`
- `artifacts/btc/daily/history.csv`
- `artifacts/btc/market_hours/history.csv`
- `artifacts/btc/market_hours_daily/history.csv`
- `artifacts/consolidated/history.csv`

## Coverage Metadata

| Variation | Source File | Date Range | Rows | Validated | Missing | Failed |
| --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 798 | 321 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 961 | 596 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 525 | 358 | 166 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 527 | 412 | 113 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 06:00:00+00:00 | 22 | 22 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 06:00:00+00:00 | 22 | 22 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 06:00:00+00:00 | 22 | 0 | 22 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 06:00:00+00:00 | 22 | 0 | 22 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 22 | 15 | 7 | 68.18% | 68.18% | 68.18% | 18.18 pp | 8 | 3 | 2.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 22 | 15 | 7 | 68.18% | 68.18% | 68.18% | 18.18 pp | 8 | 3 | 2.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 22 | 13 | 9 | 59.09% | 59.09% | 59.09% | 9.09 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 22 | 13 | 9 | 59.09% | 59.09% | 59.09% | 9.09 pp | 4 | 3 | 1.33 |
| Consolidated Hourly | rf | RandomForest | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | transformer | Transformer | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | xgb | XGBoost | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 358 | 177 | 181 | 49.44% | 48.33% | 49.44% | 0.56 pp | -4 | 37 | -0.11 |
| BTC Daily | transformer | Transformer | 586 | 290 | 296 | 49.49% | 51.67% | 50.00% | 0.51 pp | -6 | 37 | -0.16 |
| BTC Daily | mlp_sklearn | MLPClassifier | 586 | 286 | 300 | 48.81% | 47.50% | 49.38% | 1.19 pp | -14 | 37 | -0.38 |
| BTC Market Hours | transformer | Transformer | 358 | 170 | 188 | 47.49% | 46.67% | 47.49% | 2.51 pp | -18 | 37 | -0.49 |
| BTC Market Hours | nn | NN | 358 | 163 | 195 | 45.53% | 47.92% | 45.53% | 4.47 pp | -32 | 37 | -0.86 |
| BTC Daily | nn | NN | 586 | 276 | 310 | 47.10% | 45.42% | 48.12% | 2.90 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 412 | 189 | 223 | 45.87% | 46.25% | 45.87% | 4.13 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | nn | NN | 412 | 189 | 223 | 45.87% | 46.67% | 45.87% | 4.13 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 412 | 189 | 223 | 45.87% | 47.92% | 45.87% | 4.13 pp | -34 | 37 | -0.92 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 764 | 360 | 404 | 47.12% | 43.75% | 47.71% | 2.88 pp | -44 | 42 | -1.05 |
| BTC Hourly | transformer | Transformer | 764 | 357 | 407 | 46.73% | 43.75% | 45.62% | 3.27 pp | -50 | 42 | -1.19 |
| BTC Market Hours | lstm | LSTM | 358 | 155 | 203 | 43.30% | 43.33% | 43.30% | 6.70 pp | -48 | 37 | -1.30 |
| BTC Market Hours | rf | RandomForest | 358 | 153 | 205 | 42.74% | 42.08% | 42.74% | 7.26 pp | -52 | 37 | -1.41 |
| BTC Daily | lstm | LSTM | 586 | 264 | 322 | 45.05% | 45.42% | 45.21% | 4.95 pp | -58 | 37 | -1.57 |
| BTC Market Hours | xgb | XGBoost | 358 | 147 | 211 | 41.06% | 42.08% | 41.06% | 8.94 pp | -64 | 37 | -1.73 |
| BTC Hourly | rf | RandomForest | 764 | 342 | 422 | 44.76% | 45.00% | 44.38% | 5.24 pp | -80 | 42 | -1.90 |
| BTC Hourly | nn | NN | 764 | 341 | 423 | 44.63% | 40.42% | 45.21% | 5.37 pp | -82 | 42 | -1.95 |
| BTC Market Hours Daily | rf | RandomForest | 412 | 169 | 243 | 41.02% | 40.00% | 41.02% | 8.98 pp | -74 | 37 | -2.00 |
| BTC Hourly | lstm | LSTM | 764 | 337 | 427 | 44.11% | 43.33% | 45.42% | 5.89 pp | -90 | 42 | -2.14 |
| BTC Daily | rf | RandomForest | 586 | 253 | 333 | 43.17% | 43.75% | 43.96% | 6.83 pp | -80 | 37 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 412 | 166 | 246 | 40.29% | 38.75% | 40.29% | 9.71 pp | -80 | 37 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 412 | 165 | 247 | 40.05% | 38.33% | 40.05% | 9.95 pp | -82 | 37 | -2.22 |
| BTC Hourly | xgb | XGBoost | 764 | 327 | 437 | 42.80% | 41.67% | 44.17% | 7.20 pp | -110 | 42 | -2.62 |
| Consolidated Hourly | nn | NN | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 3 | -2.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 3 | -2.67 |
| BTC Daily | xgb | XGBoost | 596 | 240 | 356 | 40.27% | 35.83% | 40.83% | 9.73 pp | -116 | 37 | -3.14 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 764 | 360 | 404 | 47.12% | 43.75% | 47.71% | 2.88 pp | -44 | 42 | -1.05 |
| BTC Hourly | transformer | Transformer | 764 | 357 | 407 | 46.73% | 43.75% | 45.62% | 3.27 pp | -50 | 42 | -1.19 |
| BTC Hourly | rf | RandomForest | 764 | 342 | 422 | 44.76% | 45.00% | 44.38% | 5.24 pp | -80 | 42 | -1.90 |
| BTC Hourly | nn | NN | 764 | 341 | 423 | 44.63% | 40.42% | 45.21% | 5.37 pp | -82 | 42 | -1.95 |
| BTC Hourly | lstm | LSTM | 764 | 337 | 427 | 44.11% | 43.33% | 45.42% | 5.89 pp | -90 | 42 | -2.14 |
| BTC Hourly | xgb | XGBoost | 764 | 327 | 437 | 42.80% | 41.67% | 44.17% | 7.20 pp | -110 | 42 | -2.62 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 586 | 290 | 296 | 49.49% | 51.67% | 50.00% | 0.51 pp | -6 | 37 | -0.16 |
| BTC Daily | mlp_sklearn | MLPClassifier | 586 | 286 | 300 | 48.81% | 47.50% | 49.38% | 1.19 pp | -14 | 37 | -0.38 |
| BTC Daily | nn | NN | 586 | 276 | 310 | 47.10% | 45.42% | 48.12% | 2.90 pp | -34 | 37 | -0.92 |
| BTC Daily | lstm | LSTM | 586 | 264 | 322 | 45.05% | 45.42% | 45.21% | 4.95 pp | -58 | 37 | -1.57 |
| BTC Daily | rf | RandomForest | 586 | 253 | 333 | 43.17% | 43.75% | 43.96% | 6.83 pp | -80 | 37 | -2.16 |
| BTC Daily | xgb | XGBoost | 596 | 240 | 356 | 40.27% | 35.83% | 40.83% | 9.73 pp | -116 | 37 | -3.14 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 358 | 177 | 181 | 49.44% | 48.33% | 49.44% | 0.56 pp | -4 | 37 | -0.11 |
| BTC Market Hours | transformer | Transformer | 358 | 170 | 188 | 47.49% | 46.67% | 47.49% | 2.51 pp | -18 | 37 | -0.49 |
| BTC Market Hours | nn | NN | 358 | 163 | 195 | 45.53% | 47.92% | 45.53% | 4.47 pp | -32 | 37 | -0.86 |
| BTC Market Hours | lstm | LSTM | 358 | 155 | 203 | 43.30% | 43.33% | 43.30% | 6.70 pp | -48 | 37 | -1.30 |
| BTC Market Hours | rf | RandomForest | 358 | 153 | 205 | 42.74% | 42.08% | 42.74% | 7.26 pp | -52 | 37 | -1.41 |
| BTC Market Hours | xgb | XGBoost | 358 | 147 | 211 | 41.06% | 42.08% | 41.06% | 8.94 pp | -64 | 37 | -1.73 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 412 | 189 | 223 | 45.87% | 46.25% | 45.87% | 4.13 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | nn | NN | 412 | 189 | 223 | 45.87% | 46.67% | 45.87% | 4.13 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 412 | 189 | 223 | 45.87% | 47.92% | 45.87% | 4.13 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | rf | RandomForest | 412 | 169 | 243 | 41.02% | 40.00% | 41.02% | 8.98 pp | -74 | 37 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 412 | 166 | 246 | 40.29% | 38.75% | 40.29% | 9.71 pp | -80 | 37 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 412 | 165 | 247 | 40.05% | 38.33% | 40.05% | 9.95 pp | -82 | 37 | -2.22 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 22 | 15 | 7 | 68.18% | 68.18% | 68.18% | 18.18 pp | 8 | 3 | 2.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 22 | 13 | 9 | 59.09% | 59.09% | 59.09% | 9.09 pp | 4 | 3 | 1.33 |
| Consolidated Hourly | rf | RandomForest | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | transformer | Transformer | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | xgb | XGBoost | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Hourly | nn | NN | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 3 | -2.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 22 | 15 | 7 | 68.18% | 68.18% | 68.18% | 18.18 pp | 8 | 3 | 2.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 22 | 13 | 9 | 59.09% | 59.09% | 59.09% | 9.09 pp | 4 | 3 | 1.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 22 | 12 | 10 | 54.55% | 54.55% | 54.55% | 4.55 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 22 | 11 | 11 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 3 | 0.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 22 | 7 | 15 | 31.82% | 31.82% | 31.82% | 18.18 pp | -8 | 3 | -2.67 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

_No model-level predictions available for this variation._

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
