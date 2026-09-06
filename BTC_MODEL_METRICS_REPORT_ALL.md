# BTC Model Metrics Report - All Rows

Generated at: 2026-09-06T05:13:49.649315+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1255 | 967 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1131 | 766 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 825 | 528 | 296 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-06 00:00:00+00:00 | 827 | 582 | 243 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T20:00:00+00:00 | 174 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T20:00:00+00:00 | 174 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T20:00:00+00:00 | 174 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T20:00:00+00:00 | 175 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 528 | 257 | 271 | 48.67% | 45.83% | 48.54% | 1.33 pp | -14 | 50 | -0.28 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 174 | 85 | 89 | 48.85% | 48.85% | 48.85% | 1.15 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 174 | 85 | 89 | 48.85% | 48.85% | 48.85% | 1.15 pp | -4 | 12 | -0.33 |
| BTC Market Hours | transformer | Transformer | 528 | 253 | 275 | 47.92% | 47.92% | 48.33% | 2.08 pp | -22 | 50 | -0.44 |
| BTC Daily | mlp_sklearn | MLPClassifier | 756 | 367 | 389 | 48.54% | 47.92% | 48.75% | 1.46 pp | -22 | 44 | -0.50 |
| Consolidated Hourly | rf | RandomForest | 174 | 84 | 90 | 48.28% | 48.28% | 48.28% | 1.72 pp | -6 | 12 | -0.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 174 | 84 | 90 | 48.28% | 48.28% | 48.28% | 1.72 pp | -6 | 12 | -0.50 |
| BTC Market Hours Daily | transformer | Transformer | 582 | 276 | 306 | 47.42% | 50.83% | 48.54% | 2.58 pp | -30 | 50 | -0.60 |
| BTC Market Hours | nn | NN | 528 | 249 | 279 | 47.16% | 50.83% | 48.54% | 2.84 pp | -30 | 50 | -0.60 |
| Consolidated Market Hours | xgb | XGBoost | 49 | 23 | 26 | 46.94% | 46.94% | 46.94% | 3.06 pp | -3 | 4 | -0.75 |
| BTC Market Hours Daily | nn | NN | 582 | 271 | 311 | 46.56% | 46.25% | 47.71% | 3.44 pp | -40 | 50 | -0.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 933 | 446 | 487 | 47.80% | 50.00% | 47.29% | 2.20 pp | -41 | 49 | -0.84 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 582 | 269 | 313 | 46.22% | 51.25% | 46.88% | 3.78 pp | -44 | 50 | -0.88 |
| BTC Daily | transformer | Transformer | 756 | 357 | 399 | 47.22% | 43.75% | 48.54% | 2.78 pp | -42 | 44 | -0.95 |
| Consolidated Market Hours Daily | lstm | LSTM | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| BTC Hourly | transformer | Transformer | 933 | 437 | 496 | 46.84% | 45.83% | 45.21% | 3.16 pp | -59 | 49 | -1.20 |
| BTC Daily | nn | NN | 756 | 351 | 405 | 46.43% | 45.00% | 46.46% | 3.57 pp | -54 | 44 | -1.23 |
| Consolidated Market Hours | lstm | LSTM | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | lstm | LSTM | 174 | 79 | 95 | 45.40% | 45.40% | 45.40% | 4.60 pp | -16 | 12 | -1.33 |
| Consolidated Hourly | xgb | XGBoost | 174 | 79 | 95 | 45.40% | 45.40% | 45.40% | 4.60 pp | -16 | 12 | -1.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 174 | 79 | 95 | 45.40% | 45.40% | 45.40% | 4.60 pp | -16 | 12 | -1.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 174 | 79 | 95 | 45.40% | 45.40% | 45.40% | 4.60 pp | -16 | 12 | -1.33 |
| BTC Market Hours | lstm | LSTM | 528 | 228 | 300 | 43.18% | 42.08% | 43.96% | 6.82 pp | -72 | 50 | -1.44 |
| BTC Market Hours | rf | RandomForest | 528 | 227 | 301 | 42.99% | 44.58% | 43.54% | 7.01 pp | -74 | 50 | -1.48 |
| Consolidated Hourly | nn | NN | 174 | 77 | 97 | 44.25% | 44.25% | 44.25% | 5.75 pp | -20 | 12 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 174 | 77 | 97 | 44.25% | 44.25% | 44.25% | 5.75 pp | -20 | 12 | -1.67 |
| BTC Market Hours | xgb | XGBoost | 528 | 218 | 310 | 41.29% | 42.92% | 41.67% | 8.71 pp | -92 | 50 | -1.84 |
| BTC Market Hours Daily | rf | RandomForest | 582 | 241 | 341 | 41.41% | 43.33% | 41.04% | 8.59 pp | -100 | 50 | -2.00 |
| BTC Hourly | rf | RandomForest | 933 | 415 | 518 | 44.48% | 44.17% | 43.96% | 5.52 pp | -103 | 49 | -2.10 |
| Consolidated Hourly | transformer | Transformer | 174 | 74 | 100 | 42.53% | 42.53% | 42.53% | 7.47 pp | -26 | 12 | -2.17 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 174 | 74 | 100 | 42.53% | 42.53% | 42.53% | 7.47 pp | -26 | 12 | -2.17 |
| BTC Hourly | nn | NN | 933 | 413 | 520 | 44.27% | 42.50% | 42.08% | 5.73 pp | -107 | 49 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 582 | 236 | 346 | 40.55% | 40.00% | 40.42% | 9.45 pp | -110 | 50 | -2.20 |
| Consolidated Market Hours | rf | RandomForest | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 4 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 582 | 231 | 351 | 39.69% | 40.83% | 38.75% | 10.31 pp | -120 | 50 | -2.40 |
| Consolidated Market Hours Daily | rf | RandomForest | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| BTC Daily | lstm | LSTM | 756 | 320 | 436 | 42.33% | 35.42% | 40.42% | 7.67 pp | -116 | 44 | -2.64 |
| Consolidated Market Hours | transformer | Transformer | 49 | 19 | 30 | 38.78% | 38.78% | 38.78% | 11.22 pp | -11 | 4 | -2.75 |
| BTC Hourly | lstm | LSTM | 933 | 398 | 535 | 42.66% | 37.50% | 41.67% | 7.34 pp | -137 | 49 | -2.80 |
| BTC Daily | rf | RandomForest | 756 | 316 | 440 | 41.80% | 38.33% | 41.67% | 8.20 pp | -124 | 44 | -2.82 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| BTC Hourly | xgb | XGBoost | 933 | 392 | 541 | 42.02% | 40.83% | 40.83% | 7.98 pp | -149 | 49 | -3.04 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | nn | NN | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| BTC Daily | xgb | XGBoost | 766 | 300 | 466 | 39.16% | 35.42% | 36.88% | 10.84 pp | -166 | 44 | -3.77 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 933 | 446 | 487 | 47.80% | 50.00% | 47.29% | 2.20 pp | -41 | 49 | -0.84 |
| BTC Hourly | transformer | Transformer | 933 | 437 | 496 | 46.84% | 45.83% | 45.21% | 3.16 pp | -59 | 49 | -1.20 |
| BTC Hourly | rf | RandomForest | 933 | 415 | 518 | 44.48% | 44.17% | 43.96% | 5.52 pp | -103 | 49 | -2.10 |
| BTC Hourly | nn | NN | 933 | 413 | 520 | 44.27% | 42.50% | 42.08% | 5.73 pp | -107 | 49 | -2.18 |
| BTC Hourly | lstm | LSTM | 933 | 398 | 535 | 42.66% | 37.50% | 41.67% | 7.34 pp | -137 | 49 | -2.80 |
| BTC Hourly | xgb | XGBoost | 933 | 392 | 541 | 42.02% | 40.83% | 40.83% | 7.98 pp | -149 | 49 | -3.04 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 756 | 367 | 389 | 48.54% | 47.92% | 48.75% | 1.46 pp | -22 | 44 | -0.50 |
| BTC Daily | transformer | Transformer | 756 | 357 | 399 | 47.22% | 43.75% | 48.54% | 2.78 pp | -42 | 44 | -0.95 |
| BTC Daily | nn | NN | 756 | 351 | 405 | 46.43% | 45.00% | 46.46% | 3.57 pp | -54 | 44 | -1.23 |
| BTC Daily | lstm | LSTM | 756 | 320 | 436 | 42.33% | 35.42% | 40.42% | 7.67 pp | -116 | 44 | -2.64 |
| BTC Daily | rf | RandomForest | 756 | 316 | 440 | 41.80% | 38.33% | 41.67% | 8.20 pp | -124 | 44 | -2.82 |
| BTC Daily | xgb | XGBoost | 766 | 300 | 466 | 39.16% | 35.42% | 36.88% | 10.84 pp | -166 | 44 | -3.77 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 528 | 257 | 271 | 48.67% | 45.83% | 48.54% | 1.33 pp | -14 | 50 | -0.28 |
| BTC Market Hours | transformer | Transformer | 528 | 253 | 275 | 47.92% | 47.92% | 48.33% | 2.08 pp | -22 | 50 | -0.44 |
| BTC Market Hours | nn | NN | 528 | 249 | 279 | 47.16% | 50.83% | 48.54% | 2.84 pp | -30 | 50 | -0.60 |
| BTC Market Hours | lstm | LSTM | 528 | 228 | 300 | 43.18% | 42.08% | 43.96% | 6.82 pp | -72 | 50 | -1.44 |
| BTC Market Hours | rf | RandomForest | 528 | 227 | 301 | 42.99% | 44.58% | 43.54% | 7.01 pp | -74 | 50 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 528 | 218 | 310 | 41.29% | 42.92% | 41.67% | 8.71 pp | -92 | 50 | -1.84 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 582 | 276 | 306 | 47.42% | 50.83% | 48.54% | 2.58 pp | -30 | 50 | -0.60 |
| BTC Market Hours Daily | nn | NN | 582 | 271 | 311 | 46.56% | 46.25% | 47.71% | 3.44 pp | -40 | 50 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 582 | 269 | 313 | 46.22% | 51.25% | 46.88% | 3.78 pp | -44 | 50 | -0.88 |
| BTC Market Hours Daily | rf | RandomForest | 582 | 241 | 341 | 41.41% | 43.33% | 41.04% | 8.59 pp | -100 | 50 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 582 | 236 | 346 | 40.55% | 40.00% | 40.42% | 9.45 pp | -110 | 50 | -2.20 |
| BTC Market Hours Daily | xgb | XGBoost | 582 | 231 | 351 | 39.69% | 40.83% | 38.75% | 10.31 pp | -120 | 50 | -2.40 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 174 | 85 | 89 | 48.85% | 48.85% | 48.85% | 1.15 pp | -4 | 12 | -0.33 |
| Consolidated Hourly | rf | RandomForest | 174 | 84 | 90 | 48.28% | 48.28% | 48.28% | 1.72 pp | -6 | 12 | -0.50 |
| Consolidated Hourly | lstm | LSTM | 174 | 79 | 95 | 45.40% | 45.40% | 45.40% | 4.60 pp | -16 | 12 | -1.33 |
| Consolidated Hourly | xgb | XGBoost | 174 | 79 | 95 | 45.40% | 45.40% | 45.40% | 4.60 pp | -16 | 12 | -1.33 |
| Consolidated Hourly | nn | NN | 174 | 77 | 97 | 44.25% | 44.25% | 44.25% | 5.75 pp | -20 | 12 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 174 | 74 | 100 | 42.53% | 42.53% | 42.53% | 7.47 pp | -26 | 12 | -2.17 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 174 | 85 | 89 | 48.85% | 48.85% | 48.85% | 1.15 pp | -4 | 12 | -0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 174 | 84 | 90 | 48.28% | 48.28% | 48.28% | 1.72 pp | -6 | 12 | -0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 174 | 79 | 95 | 45.40% | 45.40% | 45.40% | 4.60 pp | -16 | 12 | -1.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 174 | 79 | 95 | 45.40% | 45.40% | 45.40% | 4.60 pp | -16 | 12 | -1.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 174 | 77 | 97 | 44.25% | 44.25% | 44.25% | 5.75 pp | -20 | 12 | -1.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 174 | 74 | 100 | 42.53% | 42.53% | 42.53% | 7.47 pp | -26 | 12 | -2.17 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 49 | 23 | 26 | 46.94% | 46.94% | 46.94% | 3.06 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours | lstm | LSTM | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 4 | -2.25 |
| Consolidated Market Hours | transformer | Transformer | 49 | 19 | 30 | 38.78% | 38.78% | 38.78% | 11.22 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | nn | NN | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 4 | -3.25 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 50 | 23 | 27 | 46.00% | 46.00% | 46.00% | 4.00 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | transformer | Transformer | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 4 | -2.50 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours Daily | nn | NN | 50 | 19 | 31 | 38.00% | 38.00% | 38.00% | 12.00 pp | -12 | 4 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
