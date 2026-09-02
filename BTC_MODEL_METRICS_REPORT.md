# BTC Model Metrics Report - All Rows

Generated at: 2026-09-02T23:58:09.429873+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1202 | 914 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1078 | 713 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 22:00:00+00:00 | 731 | 475 | 255 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-02 22:00:00+00:00 | 733 | 529 | 202 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 20:00:00+00:00 | 125 | 125 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 20:00:00+00:00 | 125 | 125 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 20:00:00+00:00 | 125 | 23 | 102 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-27 20:00:00+00:00 | 125 | 23 | 102 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 125 | 66 | 59 | 52.80% | 52.80% | 52.80% | 2.80 pp | 7 | 10 | 0.70 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 125 | 66 | 59 | 52.80% | 52.80% | 52.80% | 2.80 pp | 7 | 10 | 0.70 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 125 | 61 | 64 | 48.80% | 48.80% | 48.80% | 1.20 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 125 | 61 | 64 | 48.80% | 48.80% | 48.80% | 1.20 pp | -3 | 10 | -0.30 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 475 | 230 | 245 | 48.42% | 43.75% | 48.42% | 1.58 pp | -15 | 46 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 703 | 344 | 359 | 48.93% | 47.08% | 48.96% | 1.07 pp | -15 | 42 | -0.36 |
| Consolidated Hourly | lstm | LSTM | 125 | 60 | 65 | 48.00% | 48.00% | 48.00% | 2.00 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 125 | 60 | 65 | 48.00% | 48.00% | 48.00% | 2.00 pp | -5 | 10 | -0.50 |
| Consolidated Market Hours | rf | RandomForest | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | transformer | Transformer | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | rf | RandomForest | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| BTC Market Hours | nn | NN | 475 | 224 | 251 | 47.16% | 47.92% | 47.16% | 2.84 pp | -27 | 46 | -0.59 |
| BTC Daily | transformer | Transformer | 703 | 338 | 365 | 48.08% | 47.08% | 50.00% | 1.92 pp | -27 | 42 | -0.64 |
| BTC Market Hours | transformer | Transformer | 475 | 220 | 255 | 46.32% | 40.42% | 46.32% | 3.68 pp | -35 | 46 | -0.76 |
| BTC Market Hours Daily | transformer | Transformer | 529 | 243 | 286 | 45.94% | 47.92% | 46.67% | 4.06 pp | -43 | 46 | -0.93 |
| BTC Hourly | transformer | Transformer | 880 | 418 | 462 | 47.50% | 49.58% | 48.12% | 2.50 pp | -44 | 47 | -0.94 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 529 | 242 | 287 | 45.75% | 47.08% | 46.46% | 4.25 pp | -45 | 46 | -0.98 |
| BTC Market Hours Daily | nn | NN | 529 | 242 | 287 | 45.75% | 42.92% | 46.46% | 4.25 pp | -45 | 46 | -0.98 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 880 | 417 | 463 | 47.39% | 48.75% | 47.92% | 2.61 pp | -46 | 47 | -0.98 |
| Consolidated Hourly | transformer | Transformer | 125 | 57 | 68 | 45.60% | 45.60% | 45.60% | 4.40 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 125 | 57 | 68 | 45.60% | 45.60% | 45.60% | 4.40 pp | -11 | 10 | -1.10 |
| BTC Daily | nn | NN | 703 | 326 | 377 | 46.37% | 42.92% | 48.54% | 3.63 pp | -51 | 42 | -1.21 |
| Consolidated Hourly | xgb | XGBoost | 125 | 56 | 69 | 44.80% | 44.80% | 44.80% | 5.20 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 125 | 56 | 69 | 44.80% | 44.80% | 44.80% | 5.20 pp | -13 | 10 | -1.30 |
| BTC Market Hours | lstm | LSTM | 475 | 204 | 271 | 42.95% | 40.83% | 42.95% | 7.05 pp | -67 | 46 | -1.46 |
| BTC Market Hours | rf | RandomForest | 475 | 204 | 271 | 42.95% | 42.08% | 42.95% | 7.05 pp | -67 | 46 | -1.46 |
| Consolidated Market Hours | lstm | LSTM | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 2 | -1.50 |
| BTC Hourly | nn | NN | 880 | 396 | 484 | 45.00% | 46.25% | 43.96% | 5.00 pp | -88 | 47 | -1.87 |
| BTC Market Hours | xgb | XGBoost | 475 | 193 | 282 | 40.63% | 39.17% | 40.63% | 9.37 pp | -89 | 46 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 529 | 219 | 310 | 41.40% | 41.67% | 41.46% | 8.60 pp | -91 | 46 | -1.98 |
| BTC Hourly | rf | RandomForest | 880 | 392 | 488 | 44.55% | 45.00% | 44.38% | 5.45 pp | -96 | 47 | -2.04 |
| BTC Daily | lstm | LSTM | 703 | 305 | 398 | 43.39% | 38.75% | 42.29% | 6.61 pp | -93 | 42 | -2.21 |
| BTC Market Hours Daily | lstm | LSTM | 529 | 211 | 318 | 39.89% | 37.08% | 40.83% | 10.11 pp | -107 | 46 | -2.33 |
| BTC Daily | rf | RandomForest | 703 | 302 | 401 | 42.96% | 41.25% | 43.33% | 7.04 pp | -99 | 42 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 529 | 210 | 319 | 39.70% | 37.92% | 38.96% | 10.30 pp | -109 | 46 | -2.37 |
| Consolidated Hourly | nn | NN | 125 | 50 | 75 | 40.00% | 40.00% | 40.00% | 10.00 pp | -25 | 10 | -2.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 125 | 50 | 75 | 40.00% | 40.00% | 40.00% | 10.00 pp | -25 | 10 | -2.50 |
| BTC Hourly | lstm | LSTM | 880 | 376 | 504 | 42.73% | 38.75% | 42.08% | 7.27 pp | -128 | 47 | -2.72 |
| BTC Hourly | xgb | XGBoost | 880 | 373 | 507 | 42.39% | 42.08% | 42.92% | 7.61 pp | -134 | 47 | -2.85 |
| BTC Daily | xgb | XGBoost | 713 | 282 | 431 | 39.55% | 35.00% | 39.38% | 10.45 pp | -149 | 42 | -3.55 |
| Consolidated Market Hours | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 23 | 6 | 17 | 26.09% | 26.09% | 26.09% | 23.91 pp | -11 | 2 | -5.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 23 | 6 | 17 | 26.09% | 26.09% | 26.09% | 23.91 pp | -11 | 2 | -5.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 880 | 418 | 462 | 47.50% | 49.58% | 48.12% | 2.50 pp | -44 | 47 | -0.94 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 880 | 417 | 463 | 47.39% | 48.75% | 47.92% | 2.61 pp | -46 | 47 | -0.98 |
| BTC Hourly | nn | NN | 880 | 396 | 484 | 45.00% | 46.25% | 43.96% | 5.00 pp | -88 | 47 | -1.87 |
| BTC Hourly | rf | RandomForest | 880 | 392 | 488 | 44.55% | 45.00% | 44.38% | 5.45 pp | -96 | 47 | -2.04 |
| BTC Hourly | lstm | LSTM | 880 | 376 | 504 | 42.73% | 38.75% | 42.08% | 7.27 pp | -128 | 47 | -2.72 |
| BTC Hourly | xgb | XGBoost | 880 | 373 | 507 | 42.39% | 42.08% | 42.92% | 7.61 pp | -134 | 47 | -2.85 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 703 | 344 | 359 | 48.93% | 47.08% | 48.96% | 1.07 pp | -15 | 42 | -0.36 |
| BTC Daily | transformer | Transformer | 703 | 338 | 365 | 48.08% | 47.08% | 50.00% | 1.92 pp | -27 | 42 | -0.64 |
| BTC Daily | nn | NN | 703 | 326 | 377 | 46.37% | 42.92% | 48.54% | 3.63 pp | -51 | 42 | -1.21 |
| BTC Daily | lstm | LSTM | 703 | 305 | 398 | 43.39% | 38.75% | 42.29% | 6.61 pp | -93 | 42 | -2.21 |
| BTC Daily | rf | RandomForest | 703 | 302 | 401 | 42.96% | 41.25% | 43.33% | 7.04 pp | -99 | 42 | -2.36 |
| BTC Daily | xgb | XGBoost | 713 | 282 | 431 | 39.55% | 35.00% | 39.38% | 10.45 pp | -149 | 42 | -3.55 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 475 | 230 | 245 | 48.42% | 43.75% | 48.42% | 1.58 pp | -15 | 46 | -0.33 |
| BTC Market Hours | nn | NN | 475 | 224 | 251 | 47.16% | 47.92% | 47.16% | 2.84 pp | -27 | 46 | -0.59 |
| BTC Market Hours | transformer | Transformer | 475 | 220 | 255 | 46.32% | 40.42% | 46.32% | 3.68 pp | -35 | 46 | -0.76 |
| BTC Market Hours | lstm | LSTM | 475 | 204 | 271 | 42.95% | 40.83% | 42.95% | 7.05 pp | -67 | 46 | -1.46 |
| BTC Market Hours | rf | RandomForest | 475 | 204 | 271 | 42.95% | 42.08% | 42.95% | 7.05 pp | -67 | 46 | -1.46 |
| BTC Market Hours | xgb | XGBoost | 475 | 193 | 282 | 40.63% | 39.17% | 40.63% | 9.37 pp | -89 | 46 | -1.93 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 529 | 243 | 286 | 45.94% | 47.92% | 46.67% | 4.06 pp | -43 | 46 | -0.93 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 529 | 242 | 287 | 45.75% | 47.08% | 46.46% | 4.25 pp | -45 | 46 | -0.98 |
| BTC Market Hours Daily | nn | NN | 529 | 242 | 287 | 45.75% | 42.92% | 46.46% | 4.25 pp | -45 | 46 | -0.98 |
| BTC Market Hours Daily | rf | RandomForest | 529 | 219 | 310 | 41.40% | 41.67% | 41.46% | 8.60 pp | -91 | 46 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 529 | 211 | 318 | 39.89% | 37.08% | 40.83% | 10.11 pp | -107 | 46 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 529 | 210 | 319 | 39.70% | 37.92% | 38.96% | 10.30 pp | -109 | 46 | -2.37 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 125 | 66 | 59 | 52.80% | 52.80% | 52.80% | 2.80 pp | 7 | 10 | 0.70 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 125 | 61 | 64 | 48.80% | 48.80% | 48.80% | 1.20 pp | -3 | 10 | -0.30 |
| Consolidated Hourly | lstm | LSTM | 125 | 60 | 65 | 48.00% | 48.00% | 48.00% | 2.00 pp | -5 | 10 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 125 | 57 | 68 | 45.60% | 45.60% | 45.60% | 4.40 pp | -11 | 10 | -1.10 |
| Consolidated Hourly | xgb | XGBoost | 125 | 56 | 69 | 44.80% | 44.80% | 44.80% | 5.20 pp | -13 | 10 | -1.30 |
| Consolidated Hourly | nn | NN | 125 | 50 | 75 | 40.00% | 40.00% | 40.00% | 10.00 pp | -25 | 10 | -2.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 125 | 66 | 59 | 52.80% | 52.80% | 52.80% | 2.80 pp | 7 | 10 | 0.70 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 125 | 61 | 64 | 48.80% | 48.80% | 48.80% | 1.20 pp | -3 | 10 | -0.30 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 125 | 60 | 65 | 48.00% | 48.00% | 48.00% | 2.00 pp | -5 | 10 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 125 | 57 | 68 | 45.60% | 45.60% | 45.60% | 4.40 pp | -11 | 10 | -1.10 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 125 | 56 | 69 | 44.80% | 44.80% | 44.80% | 5.20 pp | -13 | 10 | -1.30 |
| Consolidated Daily/Hourly Refresh | nn | NN | 125 | 50 | 75 | 40.00% | 40.00% | 40.00% | 10.00 pp | -25 | 10 | -2.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | rf | RandomForest | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | transformer | Transformer | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 23 | 6 | 17 | 26.09% | 26.09% | 26.09% | 23.91 pp | -11 | 2 | -5.50 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | rf | RandomForest | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | xgb | XGBoost | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 2 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 2 | -1.50 |
| Consolidated Market Hours Daily | nn | NN | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 2 | -4.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 23 | 6 | 17 | 26.09% | 26.09% | 26.09% | 23.91 pp | -11 | 2 | -5.50 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
