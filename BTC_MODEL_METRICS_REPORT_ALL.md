# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T02:03:24.409373+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1137 | 849 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1012 | 647 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 615 | 409 | 205 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 617 | 463 | 152 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 05:00:00+00:00 | 65 | 65 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 05:00:00+00:00 | 65 | 65 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 05:00:00+00:00 | 65 | 0 | 65 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 05:00:00+00:00 | 65 | 0 | 65 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 65 | 38 | 27 | 58.46% | 58.46% | 58.46% | 8.46 pp | 11 | 7 | 1.57 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 65 | 38 | 27 | 58.46% | 58.46% | 58.46% | 8.46 pp | 11 | 7 | 1.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 65 | 35 | 30 | 53.85% | 53.85% | 53.85% | 3.85 pp | 5 | 7 | 0.71 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 65 | 35 | 30 | 53.85% | 53.85% | 53.85% | 3.85 pp | 5 | 7 | 0.71 |
| Consolidated Hourly | lstm | LSTM | 65 | 33 | 32 | 50.77% | 50.77% | 50.77% | 0.77 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 65 | 33 | 32 | 50.77% | 50.77% | 50.77% | 0.77 pp | 1 | 7 | 0.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 409 | 203 | 206 | 49.63% | 49.17% | 49.63% | 0.37 pp | -3 | 41 | -0.07 |
| BTC Daily | transformer | Transformer | 637 | 311 | 326 | 48.82% | 47.08% | 49.79% | 1.18 pp | -15 | 39 | -0.38 |
| BTC Daily | mlp_sklearn | MLPClassifier | 637 | 308 | 329 | 48.35% | 45.00% | 49.58% | 1.65 pp | -21 | 39 | -0.54 |
| BTC Market Hours | nn | NN | 409 | 193 | 216 | 47.19% | 50.42% | 47.19% | 2.81 pp | -23 | 41 | -0.56 |
| BTC Market Hours | transformer | Transformer | 409 | 189 | 220 | 46.21% | 42.50% | 46.21% | 3.79 pp | -31 | 41 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 463 | 214 | 249 | 46.22% | 45.83% | 46.22% | 3.78 pp | -35 | 41 | -0.85 |
| BTC Market Hours Daily | transformer | Transformer | 463 | 213 | 250 | 46.00% | 46.67% | 46.00% | 4.00 pp | -37 | 41 | -0.90 |
| BTC Daily | nn | NN | 637 | 299 | 338 | 46.94% | 42.92% | 48.96% | 3.06 pp | -39 | 39 | -1.00 |
| BTC Market Hours Daily | nn | NN | 463 | 211 | 252 | 45.57% | 45.00% | 45.57% | 4.43 pp | -41 | 41 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 7 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 7 | -1.00 |
| BTC Hourly | transformer | Transformer | 815 | 385 | 430 | 47.24% | 45.83% | 46.25% | 2.76 pp | -45 | 44 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 815 | 382 | 433 | 46.87% | 42.92% | 46.88% | 3.13 pp | -51 | 44 | -1.16 |
| BTC Market Hours | lstm | LSTM | 409 | 180 | 229 | 44.01% | 45.00% | 44.01% | 5.99 pp | -49 | 41 | -1.20 |
| BTC Market Hours | rf | RandomForest | 409 | 175 | 234 | 42.79% | 42.08% | 42.79% | 7.21 pp | -59 | 41 | -1.44 |
| Consolidated Hourly | transformer | Transformer | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 7 | -1.57 |
| BTC Daily | lstm | LSTM | 637 | 283 | 354 | 44.43% | 42.50% | 43.96% | 5.57 pp | -71 | 39 | -1.82 |
| BTC Hourly | nn | NN | 815 | 367 | 448 | 45.03% | 40.83% | 44.79% | 4.97 pp | -81 | 44 | -1.84 |
| BTC Hourly | rf | RandomForest | 815 | 365 | 450 | 44.79% | 44.58% | 44.58% | 5.21 pp | -85 | 44 | -1.93 |
| BTC Market Hours | xgb | XGBoost | 409 | 164 | 245 | 40.10% | 38.75% | 40.10% | 9.90 pp | -81 | 41 | -1.98 |
| BTC Market Hours Daily | rf | RandomForest | 463 | 190 | 273 | 41.04% | 41.67% | 41.04% | 8.96 pp | -83 | 41 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 463 | 189 | 274 | 40.82% | 40.00% | 40.82% | 9.18 pp | -85 | 41 | -2.07 |
| BTC Hourly | lstm | LSTM | 815 | 355 | 460 | 43.56% | 41.67% | 44.17% | 6.44 pp | -105 | 44 | -2.39 |
| Consolidated Hourly | nn | NN | 65 | 24 | 41 | 36.92% | 36.92% | 36.92% | 13.08 pp | -17 | 7 | -2.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 65 | 24 | 41 | 36.92% | 36.92% | 36.92% | 13.08 pp | -17 | 7 | -2.43 |
| BTC Daily | rf | RandomForest | 637 | 271 | 366 | 42.54% | 41.25% | 43.33% | 7.46 pp | -95 | 39 | -2.44 |
| BTC Market Hours Daily | xgb | XGBoost | 463 | 181 | 282 | 39.09% | 36.25% | 39.09% | 10.91 pp | -101 | 41 | -2.46 |
| BTC Hourly | xgb | XGBoost | 815 | 345 | 470 | 42.33% | 39.58% | 42.71% | 7.67 pp | -125 | 44 | -2.84 |
| BTC Daily | xgb | XGBoost | 647 | 252 | 395 | 38.95% | 30.42% | 38.75% | 11.05 pp | -143 | 39 | -3.67 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 815 | 385 | 430 | 47.24% | 45.83% | 46.25% | 2.76 pp | -45 | 44 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 815 | 382 | 433 | 46.87% | 42.92% | 46.88% | 3.13 pp | -51 | 44 | -1.16 |
| BTC Hourly | nn | NN | 815 | 367 | 448 | 45.03% | 40.83% | 44.79% | 4.97 pp | -81 | 44 | -1.84 |
| BTC Hourly | rf | RandomForest | 815 | 365 | 450 | 44.79% | 44.58% | 44.58% | 5.21 pp | -85 | 44 | -1.93 |
| BTC Hourly | lstm | LSTM | 815 | 355 | 460 | 43.56% | 41.67% | 44.17% | 6.44 pp | -105 | 44 | -2.39 |
| BTC Hourly | xgb | XGBoost | 815 | 345 | 470 | 42.33% | 39.58% | 42.71% | 7.67 pp | -125 | 44 | -2.84 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 637 | 311 | 326 | 48.82% | 47.08% | 49.79% | 1.18 pp | -15 | 39 | -0.38 |
| BTC Daily | mlp_sklearn | MLPClassifier | 637 | 308 | 329 | 48.35% | 45.00% | 49.58% | 1.65 pp | -21 | 39 | -0.54 |
| BTC Daily | nn | NN | 637 | 299 | 338 | 46.94% | 42.92% | 48.96% | 3.06 pp | -39 | 39 | -1.00 |
| BTC Daily | lstm | LSTM | 637 | 283 | 354 | 44.43% | 42.50% | 43.96% | 5.57 pp | -71 | 39 | -1.82 |
| BTC Daily | rf | RandomForest | 637 | 271 | 366 | 42.54% | 41.25% | 43.33% | 7.46 pp | -95 | 39 | -2.44 |
| BTC Daily | xgb | XGBoost | 647 | 252 | 395 | 38.95% | 30.42% | 38.75% | 11.05 pp | -143 | 39 | -3.67 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 409 | 203 | 206 | 49.63% | 49.17% | 49.63% | 0.37 pp | -3 | 41 | -0.07 |
| BTC Market Hours | nn | NN | 409 | 193 | 216 | 47.19% | 50.42% | 47.19% | 2.81 pp | -23 | 41 | -0.56 |
| BTC Market Hours | transformer | Transformer | 409 | 189 | 220 | 46.21% | 42.50% | 46.21% | 3.79 pp | -31 | 41 | -0.76 |
| BTC Market Hours | lstm | LSTM | 409 | 180 | 229 | 44.01% | 45.00% | 44.01% | 5.99 pp | -49 | 41 | -1.20 |
| BTC Market Hours | rf | RandomForest | 409 | 175 | 234 | 42.79% | 42.08% | 42.79% | 7.21 pp | -59 | 41 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 409 | 164 | 245 | 40.10% | 38.75% | 40.10% | 9.90 pp | -81 | 41 | -1.98 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 463 | 214 | 249 | 46.22% | 45.83% | 46.22% | 3.78 pp | -35 | 41 | -0.85 |
| BTC Market Hours Daily | transformer | Transformer | 463 | 213 | 250 | 46.00% | 46.67% | 46.00% | 4.00 pp | -37 | 41 | -0.90 |
| BTC Market Hours Daily | nn | NN | 463 | 211 | 252 | 45.57% | 45.00% | 45.57% | 4.43 pp | -41 | 41 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 463 | 190 | 273 | 41.04% | 41.67% | 41.04% | 8.96 pp | -83 | 41 | -2.02 |
| BTC Market Hours Daily | lstm | LSTM | 463 | 189 | 274 | 40.82% | 40.00% | 40.82% | 9.18 pp | -85 | 41 | -2.07 |
| BTC Market Hours Daily | xgb | XGBoost | 463 | 181 | 282 | 39.09% | 36.25% | 39.09% | 10.91 pp | -101 | 41 | -2.46 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 65 | 38 | 27 | 58.46% | 58.46% | 58.46% | 8.46 pp | 11 | 7 | 1.57 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 65 | 35 | 30 | 53.85% | 53.85% | 53.85% | 3.85 pp | 5 | 7 | 0.71 |
| Consolidated Hourly | lstm | LSTM | 65 | 33 | 32 | 50.77% | 50.77% | 50.77% | 0.77 pp | 1 | 7 | 0.14 |
| Consolidated Hourly | xgb | XGBoost | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 7 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 7 | -1.57 |
| Consolidated Hourly | nn | NN | 65 | 24 | 41 | 36.92% | 36.92% | 36.92% | 13.08 pp | -17 | 7 | -2.43 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 65 | 38 | 27 | 58.46% | 58.46% | 58.46% | 8.46 pp | 11 | 7 | 1.57 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 65 | 35 | 30 | 53.85% | 53.85% | 53.85% | 3.85 pp | 5 | 7 | 0.71 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 65 | 33 | 32 | 50.77% | 50.77% | 50.77% | 0.77 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 65 | 29 | 36 | 44.62% | 44.62% | 44.62% | 5.38 pp | -7 | 7 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 65 | 27 | 38 | 41.54% | 41.54% | 41.54% | 8.46 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 65 | 24 | 41 | 36.92% | 36.92% | 36.92% | 13.08 pp | -17 | 7 | -2.43 |

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
