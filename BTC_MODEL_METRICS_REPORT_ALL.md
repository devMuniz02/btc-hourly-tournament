# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T15:34:41.201683+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1129 | 841 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1004 | 639 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 14:00:00+00:00 | 597 | 401 | 195 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 14:00:00+00:00 | 599 | 455 | 142 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 10:00:00+00:00 | 59 | 59 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 10:00:00+00:00 | 59 | 59 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 10:00:00+00:00 | 59 | 0 | 59 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 10:00:00+00:00 | 59 | 0 | 59 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 59 | 34 | 25 | 57.63% | 57.63% | 57.63% | 7.63 pp | 9 | 6 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 59 | 34 | 25 | 57.63% | 57.63% | 57.63% | 7.63 pp | 9 | 6 | 1.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 59 | 30 | 29 | 50.85% | 50.85% | 50.85% | 0.85 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 59 | 30 | 29 | 50.85% | 50.85% | 50.85% | 0.85 pp | 1 | 6 | 0.17 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 401 | 199 | 202 | 49.63% | 48.33% | 49.63% | 0.37 pp | -3 | 40 | -0.07 |
| Consolidated Hourly | lstm | LSTM | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 6 | -0.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 6 | -0.17 |
| BTC Daily | transformer | Transformer | 629 | 308 | 321 | 48.97% | 47.50% | 49.58% | 1.03 pp | -13 | 39 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 629 | 307 | 322 | 48.81% | 46.25% | 50.00% | 1.19 pp | -15 | 39 | -0.38 |
| BTC Market Hours | nn | NN | 401 | 190 | 211 | 47.38% | 50.83% | 47.38% | 2.62 pp | -21 | 40 | -0.53 |
| BTC Market Hours | transformer | Transformer | 401 | 185 | 216 | 46.13% | 42.08% | 46.13% | 3.87 pp | -31 | 40 | -0.78 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 455 | 210 | 245 | 46.15% | 45.83% | 46.15% | 3.85 pp | -35 | 40 | -0.88 |
| BTC Market Hours Daily | transformer | Transformer | 455 | 209 | 246 | 45.93% | 47.08% | 45.93% | 4.07 pp | -37 | 40 | -0.93 |
| BTC Daily | nn | NN | 629 | 295 | 334 | 46.90% | 42.92% | 48.75% | 3.10 pp | -39 | 39 | -1.00 |
| BTC Market Hours Daily | nn | NN | 455 | 207 | 248 | 45.49% | 45.83% | 45.49% | 4.51 pp | -41 | 40 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 807 | 380 | 427 | 47.09% | 44.17% | 46.67% | 2.91 pp | -47 | 44 | -1.07 |
| BTC Hourly | transformer | Transformer | 807 | 380 | 427 | 47.09% | 44.58% | 46.25% | 2.91 pp | -47 | 44 | -1.07 |
| BTC Market Hours | lstm | LSTM | 401 | 177 | 224 | 44.14% | 45.42% | 44.14% | 5.86 pp | -47 | 40 | -1.18 |
| BTC Market Hours | rf | RandomForest | 401 | 172 | 229 | 42.89% | 41.67% | 42.89% | 7.11 pp | -57 | 40 | -1.43 |
| Consolidated Hourly | transformer | Transformer | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| BTC Daily | lstm | LSTM | 629 | 279 | 350 | 44.36% | 42.50% | 43.96% | 5.64 pp | -71 | 39 | -1.82 |
| BTC Hourly | nn | NN | 807 | 362 | 445 | 44.86% | 40.00% | 44.38% | 5.14 pp | -83 | 44 | -1.89 |
| BTC Market Hours | xgb | XGBoost | 401 | 161 | 240 | 40.15% | 37.92% | 40.15% | 9.85 pp | -79 | 40 | -1.98 |
| BTC Hourly | rf | RandomForest | 807 | 360 | 447 | 44.61% | 43.75% | 44.38% | 5.39 pp | -87 | 44 | -1.98 |
| BTC Market Hours Daily | rf | RandomForest | 455 | 186 | 269 | 40.88% | 40.42% | 40.88% | 9.12 pp | -83 | 40 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 455 | 185 | 270 | 40.66% | 39.17% | 40.66% | 9.34 pp | -85 | 40 | -2.12 |
| BTC Hourly | lstm | LSTM | 807 | 353 | 454 | 43.74% | 42.50% | 44.79% | 6.26 pp | -101 | 44 | -2.30 |
| BTC Daily | rf | RandomForest | 629 | 268 | 361 | 42.61% | 42.08% | 43.33% | 7.39 pp | -93 | 39 | -2.38 |
| BTC Market Hours Daily | xgb | XGBoost | 455 | 178 | 277 | 39.12% | 37.08% | 39.12% | 10.88 pp | -99 | 40 | -2.48 |
| BTC Hourly | xgb | XGBoost | 807 | 342 | 465 | 42.38% | 39.58% | 43.12% | 7.62 pp | -123 | 44 | -2.80 |
| Consolidated Hourly | nn | NN | 59 | 21 | 38 | 35.59% | 35.59% | 35.59% | 14.41 pp | -17 | 6 | -2.83 |
| Consolidated Daily/Hourly Refresh | nn | NN | 59 | 21 | 38 | 35.59% | 35.59% | 35.59% | 14.41 pp | -17 | 6 | -2.83 |
| BTC Daily | xgb | XGBoost | 639 | 250 | 389 | 39.12% | 31.25% | 39.17% | 10.88 pp | -139 | 39 | -3.56 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 807 | 380 | 427 | 47.09% | 44.17% | 46.67% | 2.91 pp | -47 | 44 | -1.07 |
| BTC Hourly | transformer | Transformer | 807 | 380 | 427 | 47.09% | 44.58% | 46.25% | 2.91 pp | -47 | 44 | -1.07 |
| BTC Hourly | nn | NN | 807 | 362 | 445 | 44.86% | 40.00% | 44.38% | 5.14 pp | -83 | 44 | -1.89 |
| BTC Hourly | rf | RandomForest | 807 | 360 | 447 | 44.61% | 43.75% | 44.38% | 5.39 pp | -87 | 44 | -1.98 |
| BTC Hourly | lstm | LSTM | 807 | 353 | 454 | 43.74% | 42.50% | 44.79% | 6.26 pp | -101 | 44 | -2.30 |
| BTC Hourly | xgb | XGBoost | 807 | 342 | 465 | 42.38% | 39.58% | 43.12% | 7.62 pp | -123 | 44 | -2.80 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 629 | 308 | 321 | 48.97% | 47.50% | 49.58% | 1.03 pp | -13 | 39 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 629 | 307 | 322 | 48.81% | 46.25% | 50.00% | 1.19 pp | -15 | 39 | -0.38 |
| BTC Daily | nn | NN | 629 | 295 | 334 | 46.90% | 42.92% | 48.75% | 3.10 pp | -39 | 39 | -1.00 |
| BTC Daily | lstm | LSTM | 629 | 279 | 350 | 44.36% | 42.50% | 43.96% | 5.64 pp | -71 | 39 | -1.82 |
| BTC Daily | rf | RandomForest | 629 | 268 | 361 | 42.61% | 42.08% | 43.33% | 7.39 pp | -93 | 39 | -2.38 |
| BTC Daily | xgb | XGBoost | 639 | 250 | 389 | 39.12% | 31.25% | 39.17% | 10.88 pp | -139 | 39 | -3.56 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 401 | 199 | 202 | 49.63% | 48.33% | 49.63% | 0.37 pp | -3 | 40 | -0.07 |
| BTC Market Hours | nn | NN | 401 | 190 | 211 | 47.38% | 50.83% | 47.38% | 2.62 pp | -21 | 40 | -0.53 |
| BTC Market Hours | transformer | Transformer | 401 | 185 | 216 | 46.13% | 42.08% | 46.13% | 3.87 pp | -31 | 40 | -0.78 |
| BTC Market Hours | lstm | LSTM | 401 | 177 | 224 | 44.14% | 45.42% | 44.14% | 5.86 pp | -47 | 40 | -1.18 |
| BTC Market Hours | rf | RandomForest | 401 | 172 | 229 | 42.89% | 41.67% | 42.89% | 7.11 pp | -57 | 40 | -1.43 |
| BTC Market Hours | xgb | XGBoost | 401 | 161 | 240 | 40.15% | 37.92% | 40.15% | 9.85 pp | -79 | 40 | -1.98 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 455 | 210 | 245 | 46.15% | 45.83% | 46.15% | 3.85 pp | -35 | 40 | -0.88 |
| BTC Market Hours Daily | transformer | Transformer | 455 | 209 | 246 | 45.93% | 47.08% | 45.93% | 4.07 pp | -37 | 40 | -0.93 |
| BTC Market Hours Daily | nn | NN | 455 | 207 | 248 | 45.49% | 45.83% | 45.49% | 4.51 pp | -41 | 40 | -1.02 |
| BTC Market Hours Daily | rf | RandomForest | 455 | 186 | 269 | 40.88% | 40.42% | 40.88% | 9.12 pp | -83 | 40 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 455 | 185 | 270 | 40.66% | 39.17% | 40.66% | 9.34 pp | -85 | 40 | -2.12 |
| BTC Market Hours Daily | xgb | XGBoost | 455 | 178 | 277 | 39.12% | 37.08% | 39.12% | 10.88 pp | -99 | 40 | -2.48 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 59 | 34 | 25 | 57.63% | 57.63% | 57.63% | 7.63 pp | 9 | 6 | 1.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 59 | 30 | 29 | 50.85% | 50.85% | 50.85% | 0.85 pp | 1 | 6 | 0.17 |
| Consolidated Hourly | lstm | LSTM | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 6 | -0.17 |
| Consolidated Hourly | transformer | Transformer | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| Consolidated Hourly | nn | NN | 59 | 21 | 38 | 35.59% | 35.59% | 35.59% | 14.41 pp | -17 | 6 | -2.83 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 59 | 34 | 25 | 57.63% | 57.63% | 57.63% | 7.63 pp | 9 | 6 | 1.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 59 | 30 | 29 | 50.85% | 50.85% | 50.85% | 0.85 pp | 1 | 6 | 0.17 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 59 | 29 | 30 | 49.15% | 49.15% | 49.15% | 0.85 pp | -1 | 6 | -0.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 59 | 25 | 34 | 42.37% | 42.37% | 42.37% | 7.63 pp | -9 | 6 | -1.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 59 | 21 | 38 | 35.59% | 35.59% | 35.59% | 14.41 pp | -17 | 6 | -2.83 |

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
