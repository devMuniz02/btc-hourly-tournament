# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T11:48:41.427188+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 819 | 300 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 983 | 618 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 560 | 380 | 179 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 562 | 434 | 126 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-22T02:00:00+00:00 | 40 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-22T02:00:00+00:00 | 40 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-22T02:00:00+00:00 | 40 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-22T02:00:00+00:00 | 41 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 40 | 23 | 17 | 57.50% | 57.50% | 57.50% | 7.50 pp | 6 | 5 | 1.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 40 | 23 | 17 | 57.50% | 57.50% | 57.50% | 7.50 pp | 6 | 5 | 1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 40 | 21 | 19 | 52.50% | 52.50% | 52.50% | 2.50 pp | 2 | 5 | 0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 40 | 21 | 19 | 52.50% | 52.50% | 52.50% | 2.50 pp | 2 | 5 | 0.40 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 380 | 186 | 194 | 48.95% | 47.50% | 48.95% | 1.05 pp | -8 | 39 | -0.21 |
| BTC Daily | transformer | Transformer | 608 | 299 | 309 | 49.18% | 50.00% | 50.00% | 0.82 pp | -10 | 38 | -0.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 608 | 298 | 310 | 49.01% | 47.92% | 50.00% | 0.99 pp | -12 | 38 | -0.32 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 5 | -0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 5 | -0.40 |
| BTC Market Hours | transformer | Transformer | 380 | 178 | 202 | 46.84% | 44.17% | 46.84% | 3.16 pp | -24 | 39 | -0.62 |
| BTC Market Hours | nn | NN | 380 | 176 | 204 | 46.32% | 49.17% | 46.32% | 3.68 pp | -28 | 39 | -0.72 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 785 | 372 | 413 | 47.39% | 45.00% | 47.71% | 2.61 pp | -41 | 43 | -0.95 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 434 | 198 | 236 | 45.62% | 45.83% | 45.62% | 4.38 pp | -38 | 39 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 434 | 198 | 236 | 45.62% | 47.08% | 45.62% | 4.38 pp | -38 | 39 | -0.97 |
| BTC Daily | nn | NN | 608 | 285 | 323 | 46.88% | 44.17% | 48.12% | 3.12 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | nn | NN | 434 | 197 | 237 | 45.39% | 46.25% | 45.39% | 4.61 pp | -40 | 39 | -1.03 |
| BTC Hourly | transformer | Transformer | 785 | 368 | 417 | 46.88% | 42.92% | 45.83% | 3.12 pp | -49 | 43 | -1.14 |
| Consolidated Hourly | transformer | Transformer | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| BTC Market Hours | lstm | LSTM | 380 | 163 | 217 | 42.89% | 44.17% | 42.89% | 7.11 pp | -54 | 39 | -1.38 |
| BTC Market Hours | rf | RandomForest | 380 | 160 | 220 | 42.11% | 40.42% | 42.11% | 7.89 pp | -60 | 39 | -1.54 |
| BTC Daily | lstm | LSTM | 608 | 271 | 337 | 44.57% | 43.75% | 44.38% | 5.43 pp | -66 | 38 | -1.74 |
| BTC Hourly | nn | NN | 785 | 353 | 432 | 44.97% | 40.42% | 45.83% | 5.03 pp | -79 | 43 | -1.84 |
| BTC Hourly | rf | RandomForest | 785 | 350 | 435 | 44.59% | 43.33% | 44.17% | 5.41 pp | -85 | 43 | -1.98 |
| BTC Market Hours | xgb | XGBoost | 380 | 151 | 229 | 39.74% | 38.33% | 39.74% | 10.26 pp | -78 | 39 | -2.00 |
| BTC Hourly | lstm | LSTM | 785 | 347 | 438 | 44.20% | 43.75% | 45.62% | 5.80 pp | -91 | 43 | -2.12 |
| BTC Market Hours Daily | rf | RandomForest | 434 | 175 | 259 | 40.32% | 39.17% | 40.32% | 9.68 pp | -84 | 39 | -2.15 |
| BTC Daily | rf | RandomForest | 608 | 262 | 346 | 43.09% | 43.75% | 43.54% | 6.91 pp | -84 | 38 | -2.21 |
| BTC Market Hours Daily | lstm | LSTM | 434 | 171 | 263 | 39.40% | 37.92% | 39.40% | 10.60 pp | -92 | 39 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 434 | 170 | 264 | 39.17% | 38.33% | 39.17% | 10.83 pp | -94 | 39 | -2.41 |
| BTC Hourly | xgb | XGBoost | 785 | 336 | 449 | 42.80% | 40.00% | 44.17% | 7.20 pp | -113 | 43 | -2.63 |
| Consolidated Hourly | nn | NN | 40 | 13 | 27 | 32.50% | 32.50% | 32.50% | 17.50 pp | -14 | 5 | -2.80 |
| Consolidated Daily/Hourly Refresh | nn | NN | 40 | 13 | 27 | 32.50% | 32.50% | 32.50% | 17.50 pp | -14 | 5 | -2.80 |
| BTC Daily | xgb | XGBoost | 618 | 247 | 371 | 39.97% | 33.75% | 40.21% | 10.03 pp | -124 | 38 | -3.26 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 785 | 372 | 413 | 47.39% | 45.00% | 47.71% | 2.61 pp | -41 | 43 | -0.95 |
| BTC Hourly | transformer | Transformer | 785 | 368 | 417 | 46.88% | 42.92% | 45.83% | 3.12 pp | -49 | 43 | -1.14 |
| BTC Hourly | nn | NN | 785 | 353 | 432 | 44.97% | 40.42% | 45.83% | 5.03 pp | -79 | 43 | -1.84 |
| BTC Hourly | rf | RandomForest | 785 | 350 | 435 | 44.59% | 43.33% | 44.17% | 5.41 pp | -85 | 43 | -1.98 |
| BTC Hourly | lstm | LSTM | 785 | 347 | 438 | 44.20% | 43.75% | 45.62% | 5.80 pp | -91 | 43 | -2.12 |
| BTC Hourly | xgb | XGBoost | 785 | 336 | 449 | 42.80% | 40.00% | 44.17% | 7.20 pp | -113 | 43 | -2.63 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 608 | 299 | 309 | 49.18% | 50.00% | 50.00% | 0.82 pp | -10 | 38 | -0.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 608 | 298 | 310 | 49.01% | 47.92% | 50.00% | 0.99 pp | -12 | 38 | -0.32 |
| BTC Daily | nn | NN | 608 | 285 | 323 | 46.88% | 44.17% | 48.12% | 3.12 pp | -38 | 38 | -1.00 |
| BTC Daily | lstm | LSTM | 608 | 271 | 337 | 44.57% | 43.75% | 44.38% | 5.43 pp | -66 | 38 | -1.74 |
| BTC Daily | rf | RandomForest | 608 | 262 | 346 | 43.09% | 43.75% | 43.54% | 6.91 pp | -84 | 38 | -2.21 |
| BTC Daily | xgb | XGBoost | 618 | 247 | 371 | 39.97% | 33.75% | 40.21% | 10.03 pp | -124 | 38 | -3.26 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 380 | 186 | 194 | 48.95% | 47.50% | 48.95% | 1.05 pp | -8 | 39 | -0.21 |
| BTC Market Hours | transformer | Transformer | 380 | 178 | 202 | 46.84% | 44.17% | 46.84% | 3.16 pp | -24 | 39 | -0.62 |
| BTC Market Hours | nn | NN | 380 | 176 | 204 | 46.32% | 49.17% | 46.32% | 3.68 pp | -28 | 39 | -0.72 |
| BTC Market Hours | lstm | LSTM | 380 | 163 | 217 | 42.89% | 44.17% | 42.89% | 7.11 pp | -54 | 39 | -1.38 |
| BTC Market Hours | rf | RandomForest | 380 | 160 | 220 | 42.11% | 40.42% | 42.11% | 7.89 pp | -60 | 39 | -1.54 |
| BTC Market Hours | xgb | XGBoost | 380 | 151 | 229 | 39.74% | 38.33% | 39.74% | 10.26 pp | -78 | 39 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 434 | 198 | 236 | 45.62% | 45.83% | 45.62% | 4.38 pp | -38 | 39 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 434 | 198 | 236 | 45.62% | 47.08% | 45.62% | 4.38 pp | -38 | 39 | -0.97 |
| BTC Market Hours Daily | nn | NN | 434 | 197 | 237 | 45.39% | 46.25% | 45.39% | 4.61 pp | -40 | 39 | -1.03 |
| BTC Market Hours Daily | rf | RandomForest | 434 | 175 | 259 | 40.32% | 39.17% | 40.32% | 9.68 pp | -84 | 39 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 434 | 171 | 263 | 39.40% | 37.92% | 39.40% | 10.60 pp | -92 | 39 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 434 | 170 | 264 | 39.17% | 38.33% | 39.17% | 10.83 pp | -94 | 39 | -2.41 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 40 | 23 | 17 | 57.50% | 57.50% | 57.50% | 7.50 pp | 6 | 5 | 1.20 |
| Consolidated Hourly | lstm | LSTM | 40 | 21 | 19 | 52.50% | 52.50% | 52.50% | 2.50 pp | 2 | 5 | 0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 5 | -0.40 |
| Consolidated Hourly | transformer | Transformer | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | nn | NN | 40 | 13 | 27 | 32.50% | 32.50% | 32.50% | 17.50 pp | -14 | 5 | -2.80 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 40 | 23 | 17 | 57.50% | 57.50% | 57.50% | 7.50 pp | 6 | 5 | 1.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 40 | 21 | 19 | 52.50% | 52.50% | 52.50% | 2.50 pp | 2 | 5 | 0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 5 | -0.40 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 40 | 13 | 27 | 32.50% | 32.50% | 32.50% | 17.50 pp | -14 | 5 | -2.80 |

### Consolidated Market Hours

_No model-level predictions available for this variation._

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
