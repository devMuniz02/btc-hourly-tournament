# BTC Model Metrics Report - All Rows

Generated at: 2026-09-05T18:54:07.379526+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1248 | 960 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1124 | 759 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 17:00:00+00:00 | 811 | 521 | 289 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-05 17:00:00+00:00 | 813 | 575 | 236 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 17:00:00+00:00 | 167 | 167 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 17:00:00+00:00 | 167 | 167 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 17:00:00+00:00 | 167 | 46 | 121 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 17:00:00+00:00 | 167 | 46 | 121 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 167 | 84 | 83 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 167 | 84 | 83 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 521 | 253 | 268 | 48.56% | 45.42% | 48.75% | 1.44 pp | -15 | 49 | -0.31 |
| BTC Market Hours | transformer | Transformer | 521 | 251 | 270 | 48.18% | 48.33% | 48.54% | 1.82 pp | -19 | 49 | -0.39 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| BTC Daily | mlp_sklearn | MLPClassifier | 749 | 365 | 384 | 48.73% | 47.92% | 49.17% | 1.27 pp | -19 | 44 | -0.43 |
| Consolidated Market Hours | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 575 | 274 | 301 | 47.65% | 51.25% | 48.96% | 2.35 pp | -27 | 49 | -0.55 |
| BTC Market Hours | nn | NN | 521 | 246 | 275 | 47.22% | 50.83% | 48.54% | 2.78 pp | -29 | 49 | -0.59 |
| BTC Market Hours Daily | nn | NN | 575 | 269 | 306 | 46.78% | 46.25% | 48.12% | 3.22 pp | -37 | 49 | -0.76 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 926 | 442 | 484 | 47.73% | 49.58% | 47.08% | 2.27 pp | -42 | 49 | -0.86 |
| BTC Daily | transformer | Transformer | 749 | 355 | 394 | 47.40% | 44.58% | 48.75% | 2.60 pp | -39 | 44 | -0.89 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 575 | 265 | 310 | 46.09% | 49.58% | 46.67% | 3.91 pp | -45 | 49 | -0.92 |
| Consolidated Market Hours | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| BTC Hourly | transformer | Transformer | 926 | 436 | 490 | 47.08% | 46.67% | 45.62% | 2.92 pp | -54 | 49 | -1.10 |
| BTC Daily | nn | NN | 749 | 348 | 401 | 46.46% | 44.17% | 46.88% | 3.54 pp | -53 | 44 | -1.20 |
| BTC Market Hours | lstm | LSTM | 521 | 228 | 293 | 43.76% | 43.75% | 44.17% | 6.24 pp | -65 | 49 | -1.33 |
| BTC Market Hours | rf | RandomForest | 521 | 225 | 296 | 43.19% | 45.83% | 43.75% | 6.81 pp | -71 | 49 | -1.45 |
| Consolidated Market Hours | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 167 | 74 | 93 | 44.31% | 44.31% | 44.31% | 5.69 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 167 | 74 | 93 | 44.31% | 44.31% | 44.31% | 5.69 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | xgb | XGBoost | 167 | 73 | 94 | 43.71% | 43.71% | 43.71% | 6.29 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 167 | 73 | 94 | 43.71% | 43.71% | 43.71% | 6.29 pp | -21 | 12 | -1.75 |
| BTC Market Hours | xgb | XGBoost | 521 | 216 | 305 | 41.46% | 43.33% | 42.29% | 8.54 pp | -89 | 49 | -1.82 |
| BTC Market Hours Daily | rf | RandomForest | 575 | 240 | 335 | 41.74% | 43.33% | 41.25% | 8.26 pp | -95 | 49 | -1.94 |
| BTC Hourly | rf | RandomForest | 926 | 414 | 512 | 44.71% | 44.58% | 44.38% | 5.29 pp | -98 | 49 | -2.00 |
| Consolidated Hourly | nn | NN | 167 | 71 | 96 | 42.51% | 42.51% | 42.51% | 7.49 pp | -25 | 12 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 167 | 71 | 96 | 42.51% | 42.51% | 42.51% | 7.49 pp | -25 | 12 | -2.08 |
| BTC Hourly | nn | NN | 926 | 411 | 515 | 44.38% | 42.92% | 42.29% | 5.62 pp | -104 | 49 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 575 | 235 | 340 | 40.87% | 40.83% | 40.83% | 9.13 pp | -105 | 49 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 575 | 229 | 346 | 39.83% | 41.25% | 39.17% | 10.17 pp | -117 | 49 | -2.39 |
| BTC Daily | lstm | LSTM | 749 | 319 | 430 | 42.59% | 35.83% | 40.83% | 7.41 pp | -111 | 44 | -2.52 |
| BTC Daily | rf | RandomForest | 749 | 315 | 434 | 42.06% | 38.33% | 42.08% | 7.94 pp | -119 | 44 | -2.70 |
| BTC Hourly | lstm | LSTM | 926 | 396 | 530 | 42.76% | 37.50% | 41.88% | 7.24 pp | -134 | 49 | -2.73 |
| Consolidated Market Hours | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | transformer | Transformer | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| BTC Hourly | xgb | XGBoost | 926 | 387 | 539 | 41.79% | 39.17% | 40.42% | 8.21 pp | -152 | 49 | -3.10 |
| BTC Daily | xgb | XGBoost | 759 | 298 | 461 | 39.26% | 36.25% | 37.08% | 10.74 pp | -163 | 44 | -3.70 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 926 | 442 | 484 | 47.73% | 49.58% | 47.08% | 2.27 pp | -42 | 49 | -0.86 |
| BTC Hourly | transformer | Transformer | 926 | 436 | 490 | 47.08% | 46.67% | 45.62% | 2.92 pp | -54 | 49 | -1.10 |
| BTC Hourly | rf | RandomForest | 926 | 414 | 512 | 44.71% | 44.58% | 44.38% | 5.29 pp | -98 | 49 | -2.00 |
| BTC Hourly | nn | NN | 926 | 411 | 515 | 44.38% | 42.92% | 42.29% | 5.62 pp | -104 | 49 | -2.12 |
| BTC Hourly | lstm | LSTM | 926 | 396 | 530 | 42.76% | 37.50% | 41.88% | 7.24 pp | -134 | 49 | -2.73 |
| BTC Hourly | xgb | XGBoost | 926 | 387 | 539 | 41.79% | 39.17% | 40.42% | 8.21 pp | -152 | 49 | -3.10 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 749 | 365 | 384 | 48.73% | 47.92% | 49.17% | 1.27 pp | -19 | 44 | -0.43 |
| BTC Daily | transformer | Transformer | 749 | 355 | 394 | 47.40% | 44.58% | 48.75% | 2.60 pp | -39 | 44 | -0.89 |
| BTC Daily | nn | NN | 749 | 348 | 401 | 46.46% | 44.17% | 46.88% | 3.54 pp | -53 | 44 | -1.20 |
| BTC Daily | lstm | LSTM | 749 | 319 | 430 | 42.59% | 35.83% | 40.83% | 7.41 pp | -111 | 44 | -2.52 |
| BTC Daily | rf | RandomForest | 749 | 315 | 434 | 42.06% | 38.33% | 42.08% | 7.94 pp | -119 | 44 | -2.70 |
| BTC Daily | xgb | XGBoost | 759 | 298 | 461 | 39.26% | 36.25% | 37.08% | 10.74 pp | -163 | 44 | -3.70 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 521 | 253 | 268 | 48.56% | 45.42% | 48.75% | 1.44 pp | -15 | 49 | -0.31 |
| BTC Market Hours | transformer | Transformer | 521 | 251 | 270 | 48.18% | 48.33% | 48.54% | 1.82 pp | -19 | 49 | -0.39 |
| BTC Market Hours | nn | NN | 521 | 246 | 275 | 47.22% | 50.83% | 48.54% | 2.78 pp | -29 | 49 | -0.59 |
| BTC Market Hours | lstm | LSTM | 521 | 228 | 293 | 43.76% | 43.75% | 44.17% | 6.24 pp | -65 | 49 | -1.33 |
| BTC Market Hours | rf | RandomForest | 521 | 225 | 296 | 43.19% | 45.83% | 43.75% | 6.81 pp | -71 | 49 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 521 | 216 | 305 | 41.46% | 43.33% | 42.29% | 8.54 pp | -89 | 49 | -1.82 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 575 | 274 | 301 | 47.65% | 51.25% | 48.96% | 2.35 pp | -27 | 49 | -0.55 |
| BTC Market Hours Daily | nn | NN | 575 | 269 | 306 | 46.78% | 46.25% | 48.12% | 3.22 pp | -37 | 49 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 575 | 265 | 310 | 46.09% | 49.58% | 46.67% | 3.91 pp | -45 | 49 | -0.92 |
| BTC Market Hours Daily | rf | RandomForest | 575 | 240 | 335 | 41.74% | 43.33% | 41.25% | 8.26 pp | -95 | 49 | -1.94 |
| BTC Market Hours Daily | lstm | LSTM | 575 | 235 | 340 | 40.87% | 40.83% | 40.83% | 9.13 pp | -105 | 49 | -2.14 |
| BTC Market Hours Daily | xgb | XGBoost | 575 | 229 | 346 | 39.83% | 41.25% | 39.17% | 10.17 pp | -117 | 49 | -2.39 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 167 | 84 | 83 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | lstm | LSTM | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | transformer | Transformer | 167 | 74 | 93 | 44.31% | 44.31% | 44.31% | 5.69 pp | -19 | 12 | -1.58 |
| Consolidated Hourly | xgb | XGBoost | 167 | 73 | 94 | 43.71% | 43.71% | 43.71% | 6.29 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | nn | NN | 167 | 71 | 96 | 42.51% | 42.51% | 42.51% | 7.49 pp | -25 | 12 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 167 | 84 | 83 | 50.30% | 50.30% | 50.30% | 0.30 pp | 1 | 12 | 0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 167 | 74 | 93 | 44.31% | 44.31% | 44.31% | 5.69 pp | -19 | 12 | -1.58 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 167 | 73 | 94 | 43.71% | 43.71% | 43.71% | 6.29 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | nn | NN | 167 | 71 | 96 | 42.51% | 42.51% | 42.51% | 7.49 pp | -25 | 12 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | transformer | Transformer | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours Daily | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
