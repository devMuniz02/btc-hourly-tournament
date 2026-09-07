# BTC Model Metrics Report - All Rows

Generated at: 2026-09-07T21:04:52.639100+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1282 | 994 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1158 | 793 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 20:00:00+00:00 | 874 | 555 | 318 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-07 20:00:00+00:00 | 876 | 609 | 265 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T21:00:00+00:00 | 199 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T21:00:00+00:00 | 199 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T21:00:00+00:00 | 199 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-30T21:00:00+00:00 | 200 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 199 | 98 | 101 | 49.25% | 49.25% | 49.25% | 0.75 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 199 | 98 | 101 | 49.25% | 49.25% | 49.25% | 0.75 pp | -3 | 13 | -0.23 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 555 | 271 | 284 | 48.83% | 47.50% | 47.92% | 1.17 pp | -13 | 52 | -0.25 |
| BTC Market Hours | nn | NN | 555 | 265 | 290 | 47.75% | 51.67% | 49.58% | 2.25 pp | -25 | 52 | -0.48 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 199 | 96 | 103 | 48.24% | 48.24% | 48.24% | 1.76 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 199 | 96 | 103 | 48.24% | 48.24% | 48.24% | 1.76 pp | -7 | 13 | -0.54 |
| BTC Market Hours | transformer | Transformer | 555 | 262 | 293 | 47.21% | 47.08% | 47.71% | 2.79 pp | -31 | 52 | -0.60 |
| Consolidated Market Hours | xgb | XGBoost | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 5 | -0.60 |
| BTC Daily | mlp_sklearn | MLPClassifier | 783 | 377 | 406 | 48.15% | 45.83% | 47.71% | 1.85 pp | -29 | 45 | -0.64 |
| BTC Market Hours Daily | transformer | Transformer | 609 | 286 | 323 | 46.96% | 50.00% | 47.71% | 3.04 pp | -37 | 52 | -0.71 |
| BTC Market Hours Daily | nn | NN | 609 | 284 | 325 | 46.63% | 47.92% | 48.12% | 3.37 pp | -41 | 52 | -0.79 |
| Consolidated Market Hours Daily | transformer | Transformer | 64 | 30 | 34 | 46.88% | 46.88% | 46.88% | 3.12 pp | -4 | 5 | -0.80 |
| Consolidated Market Hours Daily | xgb | XGBoost | 64 | 30 | 34 | 46.88% | 46.88% | 46.88% | 3.12 pp | -4 | 5 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 609 | 283 | 326 | 46.47% | 49.17% | 47.08% | 3.53 pp | -43 | 52 | -0.83 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 960 | 458 | 502 | 47.71% | 49.58% | 47.29% | 2.29 pp | -44 | 50 | -0.88 |
| Consolidated Market Hours | transformer | Transformer | 63 | 29 | 34 | 46.03% | 46.03% | 46.03% | 3.97 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 199 | 92 | 107 | 46.23% | 46.23% | 46.23% | 3.77 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 199 | 92 | 107 | 46.23% | 46.23% | 46.23% | 3.77 pp | -15 | 13 | -1.15 |
| BTC Daily | transformer | Transformer | 783 | 363 | 420 | 46.36% | 40.42% | 46.25% | 3.64 pp | -57 | 45 | -1.27 |
| BTC Daily | nn | NN | 783 | 362 | 421 | 46.23% | 44.17% | 45.00% | 3.77 pp | -59 | 45 | -1.31 |
| BTC Hourly | transformer | Transformer | 960 | 447 | 513 | 46.56% | 45.00% | 43.75% | 3.44 pp | -66 | 50 | -1.32 |
| Consolidated Market Hours | rf | RandomForest | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| BTC Market Hours | rf | RandomForest | 555 | 241 | 314 | 43.42% | 45.83% | 43.33% | 6.58 pp | -73 | 52 | -1.40 |
| Consolidated Hourly | nn | NN | 199 | 90 | 109 | 45.23% | 45.23% | 45.23% | 4.77 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | nn | NN | 199 | 90 | 109 | 45.23% | 45.23% | 45.23% | 4.77 pp | -19 | 13 | -1.46 |
| BTC Market Hours | lstm | LSTM | 555 | 239 | 316 | 43.06% | 41.67% | 43.33% | 6.94 pp | -77 | 52 | -1.48 |
| Consolidated Market Hours Daily | rf | RandomForest | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | lstm | LSTM | 199 | 89 | 110 | 44.72% | 44.72% | 44.72% | 5.28 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 199 | 89 | 110 | 44.72% | 44.72% | 44.72% | 5.28 pp | -21 | 13 | -1.62 |
| BTC Market Hours | xgb | XGBoost | 555 | 234 | 321 | 42.16% | 45.00% | 42.29% | 7.84 pp | -87 | 52 | -1.67 |
| Consolidated Market Hours | lstm | LSTM | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 5 | -1.80 |
| BTC Market Hours Daily | rf | RandomForest | 609 | 255 | 354 | 41.87% | 44.17% | 41.25% | 8.13 pp | -99 | 52 | -1.90 |
| Consolidated Market Hours Daily | lstm | LSTM | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 5 | -2.00 |
| Consolidated Hourly | transformer | Transformer | 199 | 86 | 113 | 43.22% | 43.22% | 43.22% | 6.78 pp | -27 | 13 | -2.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 199 | 86 | 113 | 43.22% | 43.22% | 43.22% | 6.78 pp | -27 | 13 | -2.08 |
| BTC Hourly | nn | NN | 960 | 425 | 535 | 44.27% | 42.08% | 42.50% | 5.73 pp | -110 | 50 | -2.20 |
| BTC Hourly | rf | RandomForest | 960 | 425 | 535 | 44.27% | 42.92% | 43.12% | 5.73 pp | -110 | 50 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 609 | 247 | 362 | 40.56% | 40.42% | 40.62% | 9.44 pp | -115 | 52 | -2.21 |
| BTC Market Hours Daily | xgb | XGBoost | 609 | 246 | 363 | 40.39% | 41.67% | 39.79% | 9.61 pp | -117 | 52 | -2.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| BTC Daily | lstm | LSTM | 783 | 329 | 454 | 42.02% | 34.17% | 39.79% | 7.98 pp | -125 | 45 | -2.78 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 64 | 25 | 39 | 39.06% | 39.06% | 39.06% | 10.94 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | nn | NN | 64 | 25 | 39 | 39.06% | 39.06% | 39.06% | 10.94 pp | -14 | 5 | -2.80 |
| BTC Hourly | lstm | LSTM | 960 | 409 | 551 | 42.60% | 37.08% | 41.88% | 7.40 pp | -142 | 50 | -2.84 |
| BTC Daily | rf | RandomForest | 783 | 327 | 456 | 41.76% | 37.50% | 41.67% | 8.24 pp | -129 | 45 | -2.87 |
| BTC Hourly | xgb | XGBoost | 960 | 397 | 563 | 41.35% | 37.50% | 39.38% | 8.65 pp | -166 | 50 | -3.32 |
| BTC Daily | xgb | XGBoost | 793 | 309 | 484 | 38.97% | 35.00% | 36.46% | 11.03 pp | -175 | 45 | -3.89 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 960 | 458 | 502 | 47.71% | 49.58% | 47.29% | 2.29 pp | -44 | 50 | -0.88 |
| BTC Hourly | transformer | Transformer | 960 | 447 | 513 | 46.56% | 45.00% | 43.75% | 3.44 pp | -66 | 50 | -1.32 |
| BTC Hourly | nn | NN | 960 | 425 | 535 | 44.27% | 42.08% | 42.50% | 5.73 pp | -110 | 50 | -2.20 |
| BTC Hourly | rf | RandomForest | 960 | 425 | 535 | 44.27% | 42.92% | 43.12% | 5.73 pp | -110 | 50 | -2.20 |
| BTC Hourly | lstm | LSTM | 960 | 409 | 551 | 42.60% | 37.08% | 41.88% | 7.40 pp | -142 | 50 | -2.84 |
| BTC Hourly | xgb | XGBoost | 960 | 397 | 563 | 41.35% | 37.50% | 39.38% | 8.65 pp | -166 | 50 | -3.32 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 783 | 377 | 406 | 48.15% | 45.83% | 47.71% | 1.85 pp | -29 | 45 | -0.64 |
| BTC Daily | transformer | Transformer | 783 | 363 | 420 | 46.36% | 40.42% | 46.25% | 3.64 pp | -57 | 45 | -1.27 |
| BTC Daily | nn | NN | 783 | 362 | 421 | 46.23% | 44.17% | 45.00% | 3.77 pp | -59 | 45 | -1.31 |
| BTC Daily | lstm | LSTM | 783 | 329 | 454 | 42.02% | 34.17% | 39.79% | 7.98 pp | -125 | 45 | -2.78 |
| BTC Daily | rf | RandomForest | 783 | 327 | 456 | 41.76% | 37.50% | 41.67% | 8.24 pp | -129 | 45 | -2.87 |
| BTC Daily | xgb | XGBoost | 793 | 309 | 484 | 38.97% | 35.00% | 36.46% | 11.03 pp | -175 | 45 | -3.89 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 555 | 271 | 284 | 48.83% | 47.50% | 47.92% | 1.17 pp | -13 | 52 | -0.25 |
| BTC Market Hours | nn | NN | 555 | 265 | 290 | 47.75% | 51.67% | 49.58% | 2.25 pp | -25 | 52 | -0.48 |
| BTC Market Hours | transformer | Transformer | 555 | 262 | 293 | 47.21% | 47.08% | 47.71% | 2.79 pp | -31 | 52 | -0.60 |
| BTC Market Hours | rf | RandomForest | 555 | 241 | 314 | 43.42% | 45.83% | 43.33% | 6.58 pp | -73 | 52 | -1.40 |
| BTC Market Hours | lstm | LSTM | 555 | 239 | 316 | 43.06% | 41.67% | 43.33% | 6.94 pp | -77 | 52 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 555 | 234 | 321 | 42.16% | 45.00% | 42.29% | 7.84 pp | -87 | 52 | -1.67 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 609 | 286 | 323 | 46.96% | 50.00% | 47.71% | 3.04 pp | -37 | 52 | -0.71 |
| BTC Market Hours Daily | nn | NN | 609 | 284 | 325 | 46.63% | 47.92% | 48.12% | 3.37 pp | -41 | 52 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 609 | 283 | 326 | 46.47% | 49.17% | 47.08% | 3.53 pp | -43 | 52 | -0.83 |
| BTC Market Hours Daily | rf | RandomForest | 609 | 255 | 354 | 41.87% | 44.17% | 41.25% | 8.13 pp | -99 | 52 | -1.90 |
| BTC Market Hours Daily | lstm | LSTM | 609 | 247 | 362 | 40.56% | 40.42% | 40.62% | 9.44 pp | -115 | 52 | -2.21 |
| BTC Market Hours Daily | xgb | XGBoost | 609 | 246 | 363 | 40.39% | 41.67% | 39.79% | 9.61 pp | -117 | 52 | -2.25 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 199 | 98 | 101 | 49.25% | 49.25% | 49.25% | 0.75 pp | -3 | 13 | -0.23 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 199 | 96 | 103 | 48.24% | 48.24% | 48.24% | 1.76 pp | -7 | 13 | -0.54 |
| Consolidated Hourly | xgb | XGBoost | 199 | 92 | 107 | 46.23% | 46.23% | 46.23% | 3.77 pp | -15 | 13 | -1.15 |
| Consolidated Hourly | nn | NN | 199 | 90 | 109 | 45.23% | 45.23% | 45.23% | 4.77 pp | -19 | 13 | -1.46 |
| Consolidated Hourly | lstm | LSTM | 199 | 89 | 110 | 44.72% | 44.72% | 44.72% | 5.28 pp | -21 | 13 | -1.62 |
| Consolidated Hourly | transformer | Transformer | 199 | 86 | 113 | 43.22% | 43.22% | 43.22% | 6.78 pp | -27 | 13 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 199 | 98 | 101 | 49.25% | 49.25% | 49.25% | 0.75 pp | -3 | 13 | -0.23 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 199 | 96 | 103 | 48.24% | 48.24% | 48.24% | 1.76 pp | -7 | 13 | -0.54 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 199 | 92 | 107 | 46.23% | 46.23% | 46.23% | 3.77 pp | -15 | 13 | -1.15 |
| Consolidated Daily/Hourly Refresh | nn | NN | 199 | 90 | 109 | 45.23% | 45.23% | 45.23% | 4.77 pp | -19 | 13 | -1.46 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 199 | 89 | 110 | 44.72% | 44.72% | 44.72% | 5.28 pp | -21 | 13 | -1.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 199 | 86 | 113 | 43.22% | 43.22% | 43.22% | 6.78 pp | -27 | 13 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 63 | 30 | 33 | 47.62% | 47.62% | 47.62% | 2.38 pp | -3 | 5 | -0.60 |
| Consolidated Market Hours | transformer | Transformer | 63 | 29 | 34 | 46.03% | 46.03% | 46.03% | 3.97 pp | -5 | 5 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 63 | 28 | 35 | 44.44% | 44.44% | 44.44% | 5.56 pp | -7 | 5 | -1.40 |
| Consolidated Market Hours | lstm | LSTM | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 5 | -1.80 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |
| Consolidated Market Hours | nn | NN | 63 | 25 | 38 | 39.68% | 39.68% | 39.68% | 10.32 pp | -13 | 5 | -2.60 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 64 | 30 | 34 | 46.88% | 46.88% | 46.88% | 3.12 pp | -4 | 5 | -0.80 |
| Consolidated Market Hours Daily | xgb | XGBoost | 64 | 30 | 34 | 46.88% | 46.88% | 46.88% | 3.12 pp | -4 | 5 | -0.80 |
| Consolidated Market Hours Daily | rf | RandomForest | 64 | 28 | 36 | 43.75% | 43.75% | 43.75% | 6.25 pp | -8 | 5 | -1.60 |
| Consolidated Market Hours Daily | lstm | LSTM | 64 | 27 | 37 | 42.19% | 42.19% | 42.19% | 7.81 pp | -10 | 5 | -2.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 64 | 25 | 39 | 39.06% | 39.06% | 39.06% | 10.94 pp | -14 | 5 | -2.80 |
| Consolidated Market Hours Daily | nn | NN | 64 | 25 | 39 | 39.06% | 39.06% | 39.06% | 10.94 pp | -14 | 5 | -2.80 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
