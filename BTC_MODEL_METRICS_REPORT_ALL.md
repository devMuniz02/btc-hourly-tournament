# BTC Model Metrics Report - All Rows

Generated at: 2026-09-03T13:22:51.864068+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1212 | 924 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1087 | 722 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 12:00:00+00:00 | 743 | 484 | 258 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-03 12:00:00+00:00 | 745 | 538 | 205 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 12:00:00+00:00 | 133 | 133 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 12:00:00+00:00 | 133 | 133 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 12:00:00+00:00 | 133 | 28 | 105 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-28 12:00:00+00:00 | 133 | 28 | 105 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | xgb | XGBoost | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 133 | 70 | 63 | 52.63% | 52.63% | 52.63% | 2.63 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 133 | 70 | 63 | 52.63% | 52.63% | 52.63% | 2.63 pp | 7 | 11 | 0.64 |
| BTC Daily | mlp_sklearn | MLPClassifier | 712 | 365 | 347 | 51.26% | 48.33% | 51.46% | 1.26 pp | 18 | 43 | 0.42 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 484 | 232 | 252 | 47.93% | 43.75% | 47.92% | 2.07 pp | -20 | 47 | -0.43 |
| Consolidated Hourly | lstm | LSTM | 133 | 64 | 69 | 48.12% | 48.12% | 48.12% | 1.88 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 133 | 64 | 69 | 48.12% | 48.12% | 48.12% | 1.88 pp | -5 | 11 | -0.45 |
| BTC Market Hours | nn | NN | 484 | 229 | 255 | 47.31% | 49.17% | 47.71% | 2.69 pp | -26 | 47 | -0.55 |
| Consolidated Market Hours | rf | RandomForest | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| BTC Market Hours | transformer | Transformer | 484 | 225 | 259 | 46.49% | 41.67% | 46.67% | 3.51 pp | -34 | 47 | -0.72 |
| BTC Daily | nn | NN | 712 | 339 | 373 | 47.61% | 46.67% | 48.54% | 2.39 pp | -34 | 43 | -0.79 |
| Consolidated Hourly | xgb | XGBoost | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| BTC Market Hours Daily | transformer | Transformer | 538 | 249 | 289 | 46.28% | 49.17% | 47.50% | 3.72 pp | -40 | 47 | -0.85 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 890 | 424 | 466 | 47.64% | 49.58% | 48.12% | 2.36 pp | -42 | 47 | -0.89 |
| BTC Daily | transformer | Transformer | 712 | 335 | 377 | 47.05% | 45.42% | 49.17% | 2.95 pp | -42 | 43 | -0.98 |
| BTC Hourly | transformer | Transformer | 890 | 422 | 468 | 47.42% | 48.75% | 47.50% | 2.58 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 538 | 246 | 292 | 45.72% | 47.92% | 46.88% | 4.28 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | nn | NN | 538 | 246 | 292 | 45.72% | 44.17% | 46.88% | 4.28 pp | -46 | 47 | -0.98 |
| Consolidated Hourly | transformer | Transformer | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| Consolidated Market Hours | lstm | LSTM | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | lstm | LSTM | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 484 | 209 | 275 | 43.18% | 42.08% | 43.12% | 6.82 pp | -66 | 47 | -1.40 |
| BTC Market Hours | rf | RandomForest | 484 | 206 | 278 | 42.56% | 42.08% | 42.71% | 7.44 pp | -72 | 47 | -1.53 |
| BTC Daily | lstm | LSTM | 712 | 321 | 391 | 45.08% | 38.75% | 44.38% | 4.92 pp | -70 | 43 | -1.63 |
| BTC Daily | rf | RandomForest | 712 | 318 | 394 | 44.66% | 41.67% | 44.58% | 5.34 pp | -76 | 43 | -1.77 |
| Consolidated Hourly | nn | NN | 133 | 56 | 77 | 42.11% | 42.11% | 42.11% | 7.89 pp | -21 | 11 | -1.91 |
| Consolidated Daily/Hourly Refresh | nn | NN | 133 | 56 | 77 | 42.11% | 42.11% | 42.11% | 7.89 pp | -21 | 11 | -1.91 |
| BTC Market Hours | xgb | XGBoost | 484 | 196 | 288 | 40.50% | 39.58% | 40.62% | 9.50 pp | -92 | 47 | -1.96 |
| BTC Hourly | nn | NN | 890 | 398 | 492 | 44.72% | 45.83% | 42.92% | 5.28 pp | -94 | 47 | -2.00 |
| BTC Hourly | rf | RandomForest | 890 | 397 | 493 | 44.61% | 45.00% | 44.17% | 5.39 pp | -96 | 47 | -2.04 |
| BTC Market Hours Daily | rf | RandomForest | 538 | 221 | 317 | 41.08% | 41.25% | 41.46% | 8.92 pp | -96 | 47 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 538 | 215 | 323 | 39.96% | 37.50% | 40.62% | 10.04 pp | -108 | 47 | -2.30 |
| BTC Market Hours Daily | xgb | XGBoost | 538 | 214 | 324 | 39.78% | 39.58% | 39.58% | 10.22 pp | -110 | 47 | -2.34 |
| Consolidated Market Hours | nn | NN | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | nn | NN | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 3 | -2.67 |
| BTC Hourly | lstm | LSTM | 890 | 381 | 509 | 42.81% | 38.75% | 42.08% | 7.19 pp | -128 | 47 | -2.72 |
| BTC Hourly | xgb | XGBoost | 890 | 376 | 514 | 42.25% | 42.50% | 42.29% | 7.75 pp | -138 | 47 | -2.94 |
| BTC Daily | xgb | XGBoost | 722 | 290 | 432 | 40.17% | 36.25% | 39.58% | 9.83 pp | -142 | 43 | -3.30 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 3 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 3 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 890 | 424 | 466 | 47.64% | 49.58% | 48.12% | 2.36 pp | -42 | 47 | -0.89 |
| BTC Hourly | transformer | Transformer | 890 | 422 | 468 | 47.42% | 48.75% | 47.50% | 2.58 pp | -46 | 47 | -0.98 |
| BTC Hourly | nn | NN | 890 | 398 | 492 | 44.72% | 45.83% | 42.92% | 5.28 pp | -94 | 47 | -2.00 |
| BTC Hourly | rf | RandomForest | 890 | 397 | 493 | 44.61% | 45.00% | 44.17% | 5.39 pp | -96 | 47 | -2.04 |
| BTC Hourly | lstm | LSTM | 890 | 381 | 509 | 42.81% | 38.75% | 42.08% | 7.19 pp | -128 | 47 | -2.72 |
| BTC Hourly | xgb | XGBoost | 890 | 376 | 514 | 42.25% | 42.50% | 42.29% | 7.75 pp | -138 | 47 | -2.94 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 712 | 365 | 347 | 51.26% | 48.33% | 51.46% | 1.26 pp | 18 | 43 | 0.42 |
| BTC Daily | nn | NN | 712 | 339 | 373 | 47.61% | 46.67% | 48.54% | 2.39 pp | -34 | 43 | -0.79 |
| BTC Daily | transformer | Transformer | 712 | 335 | 377 | 47.05% | 45.42% | 49.17% | 2.95 pp | -42 | 43 | -0.98 |
| BTC Daily | lstm | LSTM | 712 | 321 | 391 | 45.08% | 38.75% | 44.38% | 4.92 pp | -70 | 43 | -1.63 |
| BTC Daily | rf | RandomForest | 712 | 318 | 394 | 44.66% | 41.67% | 44.58% | 5.34 pp | -76 | 43 | -1.77 |
| BTC Daily | xgb | XGBoost | 722 | 290 | 432 | 40.17% | 36.25% | 39.58% | 9.83 pp | -142 | 43 | -3.30 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 484 | 232 | 252 | 47.93% | 43.75% | 47.92% | 2.07 pp | -20 | 47 | -0.43 |
| BTC Market Hours | nn | NN | 484 | 229 | 255 | 47.31% | 49.17% | 47.71% | 2.69 pp | -26 | 47 | -0.55 |
| BTC Market Hours | transformer | Transformer | 484 | 225 | 259 | 46.49% | 41.67% | 46.67% | 3.51 pp | -34 | 47 | -0.72 |
| BTC Market Hours | lstm | LSTM | 484 | 209 | 275 | 43.18% | 42.08% | 43.12% | 6.82 pp | -66 | 47 | -1.40 |
| BTC Market Hours | rf | RandomForest | 484 | 206 | 278 | 42.56% | 42.08% | 42.71% | 7.44 pp | -72 | 47 | -1.53 |
| BTC Market Hours | xgb | XGBoost | 484 | 196 | 288 | 40.50% | 39.58% | 40.62% | 9.50 pp | -92 | 47 | -1.96 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 538 | 249 | 289 | 46.28% | 49.17% | 47.50% | 3.72 pp | -40 | 47 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 538 | 246 | 292 | 45.72% | 47.92% | 46.88% | 4.28 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | nn | NN | 538 | 246 | 292 | 45.72% | 44.17% | 46.88% | 4.28 pp | -46 | 47 | -0.98 |
| BTC Market Hours Daily | rf | RandomForest | 538 | 221 | 317 | 41.08% | 41.25% | 41.46% | 8.92 pp | -96 | 47 | -2.04 |
| BTC Market Hours Daily | lstm | LSTM | 538 | 215 | 323 | 39.96% | 37.50% | 40.62% | 10.04 pp | -108 | 47 | -2.30 |
| BTC Market Hours Daily | xgb | XGBoost | 538 | 214 | 324 | 39.78% | 39.58% | 39.58% | 10.22 pp | -110 | 47 | -2.34 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 133 | 70 | 63 | 52.63% | 52.63% | 52.63% | 2.63 pp | 7 | 11 | 0.64 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| Consolidated Hourly | lstm | LSTM | 133 | 64 | 69 | 48.12% | 48.12% | 48.12% | 1.88 pp | -5 | 11 | -0.45 |
| Consolidated Hourly | xgb | XGBoost | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| Consolidated Hourly | transformer | Transformer | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| Consolidated Hourly | nn | NN | 133 | 56 | 77 | 42.11% | 42.11% | 42.11% | 7.89 pp | -21 | 11 | -1.91 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 133 | 70 | 63 | 52.63% | 52.63% | 52.63% | 2.63 pp | 7 | 11 | 0.64 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 133 | 65 | 68 | 48.87% | 48.87% | 48.87% | 1.13 pp | -3 | 11 | -0.27 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 133 | 64 | 69 | 48.12% | 48.12% | 48.12% | 1.88 pp | -5 | 11 | -0.45 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 133 | 62 | 71 | 46.62% | 46.62% | 46.62% | 3.38 pp | -9 | 11 | -0.82 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 133 | 60 | 73 | 45.11% | 45.11% | 45.11% | 4.89 pp | -13 | 11 | -1.18 |
| Consolidated Daily/Hourly Refresh | nn | NN | 133 | 56 | 77 | 42.11% | 42.11% | 42.11% | 7.89 pp | -21 | 11 | -1.91 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours | rf | RandomForest | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours | lstm | LSTM | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours | nn | NN | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 3 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 28 | 15 | 13 | 53.57% | 53.57% | 53.57% | 3.57 pp | 2 | 3 | 0.67 |
| Consolidated Market Hours Daily | rf | RandomForest | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | transformer | Transformer | 28 | 13 | 15 | 46.43% | 46.43% | 46.43% | 3.57 pp | -2 | 3 | -0.67 |
| Consolidated Market Hours Daily | lstm | LSTM | 28 | 12 | 16 | 42.86% | 42.86% | 42.86% | 7.14 pp | -4 | 3 | -1.33 |
| Consolidated Market Hours Daily | nn | NN | 28 | 10 | 18 | 35.71% | 35.71% | 35.71% | 14.29 pp | -8 | 3 | -2.67 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 28 | 8 | 20 | 28.57% | 28.57% | 28.57% | 21.43 pp | -12 | 3 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
