# BTC Model Metrics Report - All Rows

Generated at: 2026-09-01T05:08:01.511334+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1174 | 886 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1050 | 685 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 679 | 447 | 231 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-09-01 00:00:00+00:00 | 681 | 501 | 178 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T19:00:00+00:00 | 99 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T19:00:00+00:00 | 99 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T19:00:00+00:00 | 99 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T19:00:00+00:00 | 100 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | nn | NN | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 99 | 50 | 49 | 50.51% | 50.51% | 50.51% | 0.51 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 99 | 50 | 49 | 50.51% | 50.51% | 50.51% | 0.51 pp | 1 | 9 | 0.11 |
| Consolidated Market Hours Daily | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 447 | 218 | 229 | 48.77% | 45.42% | 48.77% | 1.23 pp | -11 | 44 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 675 | 332 | 343 | 49.19% | 48.75% | 50.00% | 0.81 pp | -11 | 41 | -0.27 |
| BTC Daily | transformer | Transformer | 675 | 328 | 347 | 48.59% | 46.67% | 49.58% | 1.41 pp | -19 | 41 | -0.46 |
| Consolidated Hourly | nn | NN | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| BTC Market Hours | nn | NN | 447 | 210 | 237 | 46.98% | 48.33% | 46.98% | 3.02 pp | -27 | 44 | -0.61 |
| Consolidated Hourly | lstm | LSTM | 99 | 46 | 53 | 46.46% | 46.46% | 46.46% | 3.54 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 99 | 46 | 53 | 46.46% | 46.46% | 46.46% | 3.54 pp | -7 | 9 | -0.78 |
| BTC Market Hours | transformer | Transformer | 447 | 206 | 241 | 46.09% | 40.83% | 46.09% | 3.91 pp | -35 | 44 | -0.80 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 501 | 231 | 270 | 46.11% | 47.08% | 46.67% | 3.89 pp | -39 | 44 | -0.89 |
| BTC Market Hours Daily | nn | NN | 501 | 229 | 272 | 45.71% | 43.33% | 46.46% | 4.29 pp | -43 | 44 | -0.98 |
| Consolidated Hourly | transformer | Transformer | 99 | 45 | 54 | 45.45% | 45.45% | 45.45% | 4.55 pp | -9 | 9 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 99 | 45 | 54 | 45.45% | 45.45% | 45.45% | 4.55 pp | -9 | 9 | -1.00 |
| BTC Hourly | transformer | Transformer | 852 | 402 | 450 | 47.18% | 47.08% | 46.88% | 2.82 pp | -48 | 45 | -1.07 |
| BTC Market Hours Daily | transformer | Transformer | 501 | 227 | 274 | 45.31% | 45.42% | 45.42% | 4.69 pp | -47 | 44 | -1.07 |
| BTC Daily | nn | NN | 675 | 315 | 360 | 46.67% | 42.92% | 48.75% | 3.33 pp | -45 | 41 | -1.10 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 852 | 400 | 452 | 46.95% | 44.17% | 46.67% | 3.05 pp | -52 | 45 | -1.16 |
| BTC Market Hours | rf | RandomForest | 447 | 194 | 253 | 43.40% | 43.33% | 43.40% | 6.60 pp | -59 | 44 | -1.34 |
| BTC Market Hours | lstm | LSTM | 447 | 191 | 256 | 42.73% | 40.42% | 42.73% | 7.27 pp | -65 | 44 | -1.48 |
| BTC Hourly | nn | NN | 852 | 385 | 467 | 45.19% | 45.00% | 44.58% | 4.81 pp | -82 | 45 | -1.82 |
| BTC Market Hours Daily | rf | RandomForest | 501 | 209 | 292 | 41.72% | 42.50% | 41.88% | 8.28 pp | -83 | 44 | -1.89 |
| BTC Market Hours | xgb | XGBoost | 447 | 180 | 267 | 40.27% | 38.33% | 40.27% | 9.73 pp | -87 | 44 | -1.98 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 10 | 4 | 6 | 40.00% | 40.00% | 40.00% | 10.00 pp | -2 | 1 | -2.00 |
| BTC Hourly | rf | RandomForest | 852 | 379 | 473 | 44.48% | 42.92% | 43.75% | 5.52 pp | -94 | 45 | -2.09 |
| BTC Daily | lstm | LSTM | 675 | 294 | 381 | 43.56% | 38.33% | 42.71% | 6.44 pp | -87 | 41 | -2.12 |
| BTC Market Hours Daily | lstm | LSTM | 501 | 201 | 300 | 40.12% | 37.92% | 40.83% | 9.88 pp | -99 | 44 | -2.25 |
| BTC Daily | rf | RandomForest | 675 | 290 | 385 | 42.96% | 41.25% | 43.75% | 7.04 pp | -95 | 41 | -2.32 |
| BTC Market Hours Daily | xgb | XGBoost | 501 | 197 | 304 | 39.32% | 36.25% | 39.17% | 10.68 pp | -107 | 44 | -2.43 |
| BTC Hourly | lstm | LSTM | 852 | 364 | 488 | 42.72% | 38.75% | 42.29% | 7.28 pp | -124 | 45 | -2.76 |
| Consolidated Market Hours | lstm | LSTM | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| BTC Hourly | xgb | XGBoost | 852 | 358 | 494 | 42.02% | 40.00% | 42.29% | 7.98 pp | -136 | 45 | -3.02 |
| BTC Daily | xgb | XGBoost | 685 | 272 | 413 | 39.71% | 35.00% | 39.58% | 10.29 pp | -141 | 41 | -3.44 |
| Consolidated Market Hours Daily | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 852 | 402 | 450 | 47.18% | 47.08% | 46.88% | 2.82 pp | -48 | 45 | -1.07 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 852 | 400 | 452 | 46.95% | 44.17% | 46.67% | 3.05 pp | -52 | 45 | -1.16 |
| BTC Hourly | nn | NN | 852 | 385 | 467 | 45.19% | 45.00% | 44.58% | 4.81 pp | -82 | 45 | -1.82 |
| BTC Hourly | rf | RandomForest | 852 | 379 | 473 | 44.48% | 42.92% | 43.75% | 5.52 pp | -94 | 45 | -2.09 |
| BTC Hourly | lstm | LSTM | 852 | 364 | 488 | 42.72% | 38.75% | 42.29% | 7.28 pp | -124 | 45 | -2.76 |
| BTC Hourly | xgb | XGBoost | 852 | 358 | 494 | 42.02% | 40.00% | 42.29% | 7.98 pp | -136 | 45 | -3.02 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 675 | 332 | 343 | 49.19% | 48.75% | 50.00% | 0.81 pp | -11 | 41 | -0.27 |
| BTC Daily | transformer | Transformer | 675 | 328 | 347 | 48.59% | 46.67% | 49.58% | 1.41 pp | -19 | 41 | -0.46 |
| BTC Daily | nn | NN | 675 | 315 | 360 | 46.67% | 42.92% | 48.75% | 3.33 pp | -45 | 41 | -1.10 |
| BTC Daily | lstm | LSTM | 675 | 294 | 381 | 43.56% | 38.33% | 42.71% | 6.44 pp | -87 | 41 | -2.12 |
| BTC Daily | rf | RandomForest | 675 | 290 | 385 | 42.96% | 41.25% | 43.75% | 7.04 pp | -95 | 41 | -2.32 |
| BTC Daily | xgb | XGBoost | 685 | 272 | 413 | 39.71% | 35.00% | 39.58% | 10.29 pp | -141 | 41 | -3.44 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 447 | 218 | 229 | 48.77% | 45.42% | 48.77% | 1.23 pp | -11 | 44 | -0.25 |
| BTC Market Hours | nn | NN | 447 | 210 | 237 | 46.98% | 48.33% | 46.98% | 3.02 pp | -27 | 44 | -0.61 |
| BTC Market Hours | transformer | Transformer | 447 | 206 | 241 | 46.09% | 40.83% | 46.09% | 3.91 pp | -35 | 44 | -0.80 |
| BTC Market Hours | rf | RandomForest | 447 | 194 | 253 | 43.40% | 43.33% | 43.40% | 6.60 pp | -59 | 44 | -1.34 |
| BTC Market Hours | lstm | LSTM | 447 | 191 | 256 | 42.73% | 40.42% | 42.73% | 7.27 pp | -65 | 44 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 447 | 180 | 267 | 40.27% | 38.33% | 40.27% | 9.73 pp | -87 | 44 | -1.98 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 501 | 231 | 270 | 46.11% | 47.08% | 46.67% | 3.89 pp | -39 | 44 | -0.89 |
| BTC Market Hours Daily | nn | NN | 501 | 229 | 272 | 45.71% | 43.33% | 46.46% | 4.29 pp | -43 | 44 | -0.98 |
| BTC Market Hours Daily | transformer | Transformer | 501 | 227 | 274 | 45.31% | 45.42% | 45.42% | 4.69 pp | -47 | 44 | -1.07 |
| BTC Market Hours Daily | rf | RandomForest | 501 | 209 | 292 | 41.72% | 42.50% | 41.88% | 8.28 pp | -83 | 44 | -1.89 |
| BTC Market Hours Daily | lstm | LSTM | 501 | 201 | 300 | 40.12% | 37.92% | 40.83% | 9.88 pp | -99 | 44 | -2.25 |
| BTC Market Hours Daily | xgb | XGBoost | 501 | 197 | 304 | 39.32% | 36.25% | 39.17% | 10.68 pp | -107 | 44 | -2.43 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | rf | RandomForest | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 99 | 50 | 49 | 50.51% | 50.51% | 50.51% | 0.51 pp | 1 | 9 | 0.11 |
| Consolidated Hourly | nn | NN | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | lstm | LSTM | 99 | 46 | 53 | 46.46% | 46.46% | 46.46% | 3.54 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | transformer | Transformer | 99 | 45 | 54 | 45.45% | 45.45% | 45.45% | 4.55 pp | -9 | 9 | -1.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 99 | 51 | 48 | 51.52% | 51.52% | 51.52% | 1.52 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 99 | 50 | 49 | 50.51% | 50.51% | 50.51% | 0.51 pp | 1 | 9 | 0.11 |
| Consolidated Daily/Hourly Refresh | nn | NN | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 99 | 46 | 53 | 46.46% | 46.46% | 46.46% | 3.54 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 99 | 45 | 54 | 45.45% | 45.45% | 45.45% | 4.55 pp | -9 | 9 | -1.00 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | transformer | Transformer | 9 | 6 | 3 | 66.67% | 66.67% | 66.67% | 16.67 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | nn | NN | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 9 | 5 | 4 | 55.56% | 55.56% | 55.56% | 5.56 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 9 | 3 | 6 | 33.33% | 33.33% | 33.33% | 16.67 pp | -3 | 1 | -3.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | transformer | Transformer | 10 | 6 | 4 | 60.00% | 60.00% | 60.00% | 10.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours Daily | nn | NN | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 10 | 5 | 5 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 10 | 4 | 6 | 40.00% | 40.00% | 40.00% | 10.00 pp | -2 | 1 | -2.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 10 | 3 | 7 | 30.00% | 30.00% | 30.00% | 20.00 pp | -4 | 1 | -4.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
