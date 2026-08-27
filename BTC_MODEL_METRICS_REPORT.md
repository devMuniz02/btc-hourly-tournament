# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T10:03:22.755171+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 799 | 320 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 963 | 598 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 527 | 360 | 166 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 529 | 414 | 113 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 08:00:00+00:00 | 24 | 24 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 08:00:00+00:00 | 24 | 24 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 08:00:00+00:00 | 24 | 0 | 24 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 08:00:00+00:00 | 24 | 0 | 24 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 24 | 16 | 8 | 66.67% | 66.67% | 66.67% | 16.67 pp | 8 | 3 | 2.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 24 | 16 | 8 | 66.67% | 66.67% | 66.67% | 16.67 pp | 8 | 3 | 2.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | transformer | Transformer | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 360 | 178 | 182 | 49.44% | 47.92% | 49.44% | 0.56 pp | -4 | 37 | -0.11 |
| BTC Daily | transformer | Transformer | 588 | 291 | 297 | 49.49% | 51.67% | 50.00% | 0.51 pp | -6 | 37 | -0.16 |
| BTC Daily | mlp_sklearn | MLPClassifier | 588 | 287 | 301 | 48.81% | 47.50% | 49.58% | 1.19 pp | -14 | 37 | -0.38 |
| BTC Market Hours | transformer | Transformer | 360 | 171 | 189 | 47.50% | 46.25% | 47.50% | 2.50 pp | -18 | 37 | -0.49 |
| Consolidated Hourly | xgb | XGBoost | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 3 | -0.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 3 | -0.67 |
| BTC Market Hours Daily | nn | NN | 414 | 191 | 223 | 46.14% | 47.08% | 46.14% | 3.86 pp | -32 | 37 | -0.86 |
| BTC Market Hours | nn | NN | 360 | 164 | 196 | 45.56% | 47.50% | 45.56% | 4.44 pp | -32 | 37 | -0.86 |
| BTC Daily | nn | NN | 588 | 277 | 311 | 47.11% | 45.42% | 48.12% | 2.89 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 414 | 190 | 224 | 45.89% | 46.25% | 45.89% | 4.11 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 414 | 189 | 225 | 45.65% | 47.92% | 45.65% | 4.35 pp | -36 | 37 | -0.97 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 765 | 361 | 404 | 47.19% | 43.75% | 47.71% | 2.81 pp | -43 | 42 | -1.02 |
| BTC Hourly | transformer | Transformer | 765 | 358 | 407 | 46.80% | 43.75% | 45.62% | 3.20 pp | -49 | 42 | -1.17 |
| BTC Market Hours | lstm | LSTM | 360 | 156 | 204 | 43.33% | 43.33% | 43.33% | 6.67 pp | -48 | 37 | -1.30 |
| BTC Market Hours | rf | RandomForest | 360 | 154 | 206 | 42.78% | 41.67% | 42.78% | 7.22 pp | -52 | 37 | -1.41 |
| BTC Daily | lstm | LSTM | 588 | 264 | 324 | 44.90% | 45.00% | 45.00% | 5.10 pp | -60 | 37 | -1.62 |
| BTC Market Hours | xgb | XGBoost | 360 | 147 | 213 | 40.83% | 41.67% | 40.83% | 9.17 pp | -66 | 37 | -1.78 |
| BTC Hourly | rf | RandomForest | 765 | 343 | 422 | 44.84% | 45.00% | 44.58% | 5.16 pp | -79 | 42 | -1.88 |
| BTC Hourly | nn | NN | 765 | 342 | 423 | 44.71% | 40.83% | 45.42% | 5.29 pp | -81 | 42 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 414 | 170 | 244 | 41.06% | 40.42% | 41.06% | 8.94 pp | -74 | 37 | -2.00 |
| BTC Hourly | lstm | LSTM | 765 | 338 | 427 | 44.18% | 43.75% | 45.62% | 5.82 pp | -89 | 42 | -2.12 |
| BTC Daily | rf | RandomForest | 588 | 254 | 334 | 43.20% | 43.75% | 44.17% | 6.80 pp | -80 | 37 | -2.16 |
| BTC Market Hours Daily | xgb | XGBoost | 414 | 166 | 248 | 40.10% | 38.75% | 40.10% | 9.90 pp | -82 | 37 | -2.22 |
| BTC Market Hours Daily | lstm | LSTM | 414 | 165 | 249 | 39.86% | 38.33% | 39.86% | 10.14 pp | -84 | 37 | -2.27 |
| BTC Hourly | xgb | XGBoost | 765 | 328 | 437 | 42.88% | 41.67% | 44.38% | 7.12 pp | -109 | 42 | -2.60 |
| BTC Daily | xgb | XGBoost | 598 | 241 | 357 | 40.30% | 36.25% | 40.62% | 9.70 pp | -116 | 37 | -3.14 |
| Consolidated Hourly | nn | NN | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 3 | -3.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 3 | -3.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 765 | 361 | 404 | 47.19% | 43.75% | 47.71% | 2.81 pp | -43 | 42 | -1.02 |
| BTC Hourly | transformer | Transformer | 765 | 358 | 407 | 46.80% | 43.75% | 45.62% | 3.20 pp | -49 | 42 | -1.17 |
| BTC Hourly | rf | RandomForest | 765 | 343 | 422 | 44.84% | 45.00% | 44.58% | 5.16 pp | -79 | 42 | -1.88 |
| BTC Hourly | nn | NN | 765 | 342 | 423 | 44.71% | 40.83% | 45.42% | 5.29 pp | -81 | 42 | -1.93 |
| BTC Hourly | lstm | LSTM | 765 | 338 | 427 | 44.18% | 43.75% | 45.62% | 5.82 pp | -89 | 42 | -2.12 |
| BTC Hourly | xgb | XGBoost | 765 | 328 | 437 | 42.88% | 41.67% | 44.38% | 7.12 pp | -109 | 42 | -2.60 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 588 | 291 | 297 | 49.49% | 51.67% | 50.00% | 0.51 pp | -6 | 37 | -0.16 |
| BTC Daily | mlp_sklearn | MLPClassifier | 588 | 287 | 301 | 48.81% | 47.50% | 49.58% | 1.19 pp | -14 | 37 | -0.38 |
| BTC Daily | nn | NN | 588 | 277 | 311 | 47.11% | 45.42% | 48.12% | 2.89 pp | -34 | 37 | -0.92 |
| BTC Daily | lstm | LSTM | 588 | 264 | 324 | 44.90% | 45.00% | 45.00% | 5.10 pp | -60 | 37 | -1.62 |
| BTC Daily | rf | RandomForest | 588 | 254 | 334 | 43.20% | 43.75% | 44.17% | 6.80 pp | -80 | 37 | -2.16 |
| BTC Daily | xgb | XGBoost | 598 | 241 | 357 | 40.30% | 36.25% | 40.62% | 9.70 pp | -116 | 37 | -3.14 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 360 | 178 | 182 | 49.44% | 47.92% | 49.44% | 0.56 pp | -4 | 37 | -0.11 |
| BTC Market Hours | transformer | Transformer | 360 | 171 | 189 | 47.50% | 46.25% | 47.50% | 2.50 pp | -18 | 37 | -0.49 |
| BTC Market Hours | nn | NN | 360 | 164 | 196 | 45.56% | 47.50% | 45.56% | 4.44 pp | -32 | 37 | -0.86 |
| BTC Market Hours | lstm | LSTM | 360 | 156 | 204 | 43.33% | 43.33% | 43.33% | 6.67 pp | -48 | 37 | -1.30 |
| BTC Market Hours | rf | RandomForest | 360 | 154 | 206 | 42.78% | 41.67% | 42.78% | 7.22 pp | -52 | 37 | -1.41 |
| BTC Market Hours | xgb | XGBoost | 360 | 147 | 213 | 40.83% | 41.67% | 40.83% | 9.17 pp | -66 | 37 | -1.78 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 414 | 191 | 223 | 46.14% | 47.08% | 46.14% | 3.86 pp | -32 | 37 | -0.86 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 414 | 190 | 224 | 45.89% | 46.25% | 45.89% | 4.11 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 414 | 189 | 225 | 45.65% | 47.92% | 45.65% | 4.35 pp | -36 | 37 | -0.97 |
| BTC Market Hours Daily | rf | RandomForest | 414 | 170 | 244 | 41.06% | 40.42% | 41.06% | 8.94 pp | -74 | 37 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 414 | 166 | 248 | 40.10% | 38.75% | 40.10% | 9.90 pp | -82 | 37 | -2.22 |
| BTC Market Hours Daily | lstm | LSTM | 414 | 165 | 249 | 39.86% | 38.33% | 39.86% | 10.14 pp | -84 | 37 | -2.27 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 24 | 16 | 8 | 66.67% | 66.67% | 66.67% | 16.67 pp | 8 | 3 | 2.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | transformer | Transformer | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | xgb | XGBoost | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 3 | -0.67 |
| Consolidated Hourly | nn | NN | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 3 | -3.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 24 | 16 | 8 | 66.67% | 66.67% | 66.67% | 16.67 pp | 8 | 3 | 2.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 24 | 13 | 11 | 54.17% | 54.17% | 54.17% | 4.17 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 24 | 11 | 13 | 45.83% | 45.83% | 45.83% | 4.17 pp | -2 | 3 | -0.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 24 | 7 | 17 | 29.17% | 29.17% | 29.17% | 20.83 pp | -10 | 3 | -3.33 |

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
