# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T21:27:53.846508+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1133 | 845 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1009 | 644 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 20:00:00+00:00 | 608 | 406 | 201 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 20:00:00+00:00 | 610 | 460 | 148 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 13:00:00+00:00 | 63 | 63 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 13:00:00+00:00 | 63 | 63 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 13:00:00+00:00 | 63 | 1 | 62 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 13:00:00+00:00 | 63 | 1 | 62 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 63 | 36 | 27 | 57.14% | 57.14% | 57.14% | 7.14 pp | 9 | 7 | 1.29 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 63 | 36 | 27 | 57.14% | 57.14% | 57.14% | 7.14 pp | 9 | 7 | 1.29 |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 63 | 32 | 31 | 50.79% | 50.79% | 50.79% | 0.79 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 63 | 32 | 31 | 50.79% | 50.79% | 50.79% | 0.79 pp | 1 | 7 | 0.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 406 | 201 | 205 | 49.51% | 48.75% | 49.51% | 0.49 pp | -4 | 41 | -0.10 |
| Consolidated Hourly | lstm | LSTM | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 7 | -0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 7 | -0.14 |
| BTC Daily | transformer | Transformer | 634 | 311 | 323 | 49.05% | 48.33% | 49.79% | 0.95 pp | -12 | 39 | -0.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 634 | 309 | 325 | 48.74% | 46.25% | 50.21% | 1.26 pp | -16 | 39 | -0.41 |
| BTC Market Hours | nn | NN | 406 | 192 | 214 | 47.29% | 51.25% | 47.29% | 2.71 pp | -22 | 41 | -0.54 |
| BTC Market Hours | transformer | Transformer | 406 | 188 | 218 | 46.31% | 42.92% | 46.31% | 3.69 pp | -30 | 41 | -0.73 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 460 | 213 | 247 | 46.30% | 46.25% | 46.30% | 3.70 pp | -34 | 41 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 460 | 212 | 248 | 46.09% | 47.08% | 46.09% | 3.91 pp | -36 | 41 | -0.88 |
| BTC Daily | nn | NN | 634 | 298 | 336 | 47.00% | 42.92% | 49.17% | 3.00 pp | -38 | 39 | -0.97 |
| BTC Market Hours Daily | nn | NN | 460 | 210 | 250 | 45.65% | 45.42% | 45.65% | 4.35 pp | -40 | 41 | -0.98 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | transformer | Transformer | 811 | 383 | 428 | 47.23% | 45.42% | 46.25% | 2.77 pp | -45 | 44 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 811 | 382 | 429 | 47.10% | 44.17% | 47.08% | 2.90 pp | -47 | 44 | -1.07 |
| BTC Market Hours | lstm | LSTM | 406 | 179 | 227 | 44.09% | 45.42% | 44.09% | 5.91 pp | -48 | 41 | -1.17 |
| Consolidated Hourly | xgb | XGBoost | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 7 | -1.29 |
| BTC Market Hours | rf | RandomForest | 406 | 174 | 232 | 42.86% | 42.08% | 42.86% | 7.14 pp | -58 | 41 | -1.41 |
| Consolidated Hourly | transformer | Transformer | 63 | 26 | 37 | 41.27% | 41.27% | 41.27% | 8.73 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 63 | 26 | 37 | 41.27% | 41.27% | 41.27% | 8.73 pp | -11 | 7 | -1.57 |
| BTC Hourly | nn | NN | 811 | 365 | 446 | 45.01% | 40.83% | 45.00% | 4.99 pp | -81 | 44 | -1.84 |
| BTC Daily | lstm | LSTM | 634 | 281 | 353 | 44.32% | 42.50% | 43.54% | 5.68 pp | -72 | 39 | -1.85 |
| BTC Hourly | rf | RandomForest | 811 | 363 | 448 | 44.76% | 44.58% | 44.58% | 5.24 pp | -85 | 44 | -1.93 |
| BTC Market Hours | xgb | XGBoost | 406 | 163 | 243 | 40.15% | 38.33% | 40.15% | 9.85 pp | -80 | 41 | -1.95 |
| BTC Market Hours Daily | rf | RandomForest | 460 | 189 | 271 | 41.09% | 41.25% | 41.09% | 8.91 pp | -82 | 41 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 460 | 188 | 272 | 40.87% | 40.42% | 40.87% | 9.13 pp | -84 | 41 | -2.05 |
| BTC Hourly | lstm | LSTM | 811 | 355 | 456 | 43.77% | 42.50% | 44.58% | 6.23 pp | -101 | 44 | -2.30 |
| BTC Daily | rf | RandomForest | 634 | 271 | 363 | 42.74% | 42.50% | 43.75% | 7.26 pp | -92 | 39 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 460 | 181 | 279 | 39.35% | 37.08% | 39.35% | 10.65 pp | -98 | 41 | -2.39 |
| Consolidated Hourly | nn | NN | 63 | 22 | 41 | 34.92% | 34.92% | 34.92% | 15.08 pp | -19 | 7 | -2.71 |
| Consolidated Daily/Hourly Refresh | nn | NN | 63 | 22 | 41 | 34.92% | 34.92% | 34.92% | 15.08 pp | -19 | 7 | -2.71 |
| BTC Hourly | xgb | XGBoost | 811 | 344 | 467 | 42.42% | 40.00% | 42.71% | 7.58 pp | -123 | 44 | -2.80 |
| BTC Daily | xgb | XGBoost | 644 | 252 | 392 | 39.13% | 30.83% | 39.17% | 10.87 pp | -140 | 39 | -3.59 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 811 | 383 | 428 | 47.23% | 45.42% | 46.25% | 2.77 pp | -45 | 44 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 811 | 382 | 429 | 47.10% | 44.17% | 47.08% | 2.90 pp | -47 | 44 | -1.07 |
| BTC Hourly | nn | NN | 811 | 365 | 446 | 45.01% | 40.83% | 45.00% | 4.99 pp | -81 | 44 | -1.84 |
| BTC Hourly | rf | RandomForest | 811 | 363 | 448 | 44.76% | 44.58% | 44.58% | 5.24 pp | -85 | 44 | -1.93 |
| BTC Hourly | lstm | LSTM | 811 | 355 | 456 | 43.77% | 42.50% | 44.58% | 6.23 pp | -101 | 44 | -2.30 |
| BTC Hourly | xgb | XGBoost | 811 | 344 | 467 | 42.42% | 40.00% | 42.71% | 7.58 pp | -123 | 44 | -2.80 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 634 | 311 | 323 | 49.05% | 48.33% | 49.79% | 0.95 pp | -12 | 39 | -0.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 634 | 309 | 325 | 48.74% | 46.25% | 50.21% | 1.26 pp | -16 | 39 | -0.41 |
| BTC Daily | nn | NN | 634 | 298 | 336 | 47.00% | 42.92% | 49.17% | 3.00 pp | -38 | 39 | -0.97 |
| BTC Daily | lstm | LSTM | 634 | 281 | 353 | 44.32% | 42.50% | 43.54% | 5.68 pp | -72 | 39 | -1.85 |
| BTC Daily | rf | RandomForest | 634 | 271 | 363 | 42.74% | 42.50% | 43.75% | 7.26 pp | -92 | 39 | -2.36 |
| BTC Daily | xgb | XGBoost | 644 | 252 | 392 | 39.13% | 30.83% | 39.17% | 10.87 pp | -140 | 39 | -3.59 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 406 | 201 | 205 | 49.51% | 48.75% | 49.51% | 0.49 pp | -4 | 41 | -0.10 |
| BTC Market Hours | nn | NN | 406 | 192 | 214 | 47.29% | 51.25% | 47.29% | 2.71 pp | -22 | 41 | -0.54 |
| BTC Market Hours | transformer | Transformer | 406 | 188 | 218 | 46.31% | 42.92% | 46.31% | 3.69 pp | -30 | 41 | -0.73 |
| BTC Market Hours | lstm | LSTM | 406 | 179 | 227 | 44.09% | 45.42% | 44.09% | 5.91 pp | -48 | 41 | -1.17 |
| BTC Market Hours | rf | RandomForest | 406 | 174 | 232 | 42.86% | 42.08% | 42.86% | 7.14 pp | -58 | 41 | -1.41 |
| BTC Market Hours | xgb | XGBoost | 406 | 163 | 243 | 40.15% | 38.33% | 40.15% | 9.85 pp | -80 | 41 | -1.95 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 460 | 213 | 247 | 46.30% | 46.25% | 46.30% | 3.70 pp | -34 | 41 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 460 | 212 | 248 | 46.09% | 47.08% | 46.09% | 3.91 pp | -36 | 41 | -0.88 |
| BTC Market Hours Daily | nn | NN | 460 | 210 | 250 | 45.65% | 45.42% | 45.65% | 4.35 pp | -40 | 41 | -0.98 |
| BTC Market Hours Daily | rf | RandomForest | 460 | 189 | 271 | 41.09% | 41.25% | 41.09% | 8.91 pp | -82 | 41 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 460 | 188 | 272 | 40.87% | 40.42% | 40.87% | 9.13 pp | -84 | 41 | -2.05 |
| BTC Market Hours Daily | xgb | XGBoost | 460 | 181 | 279 | 39.35% | 37.08% | 39.35% | 10.65 pp | -98 | 41 | -2.39 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 63 | 36 | 27 | 57.14% | 57.14% | 57.14% | 7.14 pp | 9 | 7 | 1.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 63 | 32 | 31 | 50.79% | 50.79% | 50.79% | 0.79 pp | 1 | 7 | 0.14 |
| Consolidated Hourly | lstm | LSTM | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 7 | -0.14 |
| Consolidated Hourly | xgb | XGBoost | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | transformer | Transformer | 63 | 26 | 37 | 41.27% | 41.27% | 41.27% | 8.73 pp | -11 | 7 | -1.57 |
| Consolidated Hourly | nn | NN | 63 | 22 | 41 | 34.92% | 34.92% | 34.92% | 15.08 pp | -19 | 7 | -2.71 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 63 | 36 | 27 | 57.14% | 57.14% | 57.14% | 7.14 pp | 9 | 7 | 1.29 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 63 | 32 | 31 | 50.79% | 50.79% | 50.79% | 0.79 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 63 | 31 | 32 | 49.21% | 49.21% | 49.21% | 0.79 pp | -1 | 7 | -0.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 63 | 27 | 36 | 42.86% | 42.86% | 42.86% | 7.14 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 63 | 26 | 37 | 41.27% | 41.27% | 41.27% | 8.73 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 63 | 22 | 41 | 34.92% | 34.92% | 34.92% | 15.08 pp | -19 | 7 | -2.71 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
