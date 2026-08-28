# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T08:40:19.370693+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 817 | 302 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 980 | 615 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 557 | 377 | 179 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 559 | 431 | 126 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 11:00:00+00:00 | 38 | 38 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 11:00:00+00:00 | 38 | 38 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 11:00:00+00:00 | 38 | 0 | 38 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 11:00:00+00:00 | 38 | 0 | 38 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 38 | 22 | 16 | 57.89% | 57.89% | 57.89% | 7.89 pp | 6 | 4 | 1.50 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 38 | 22 | 16 | 57.89% | 57.89% | 57.89% | 7.89 pp | 6 | 4 | 1.50 |
| Consolidated Hourly | lstm | LSTM | 38 | 20 | 18 | 52.63% | 52.63% | 52.63% | 2.63 pp | 2 | 4 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 38 | 20 | 18 | 52.63% | 52.63% | 52.63% | 2.63 pp | 2 | 4 | 0.50 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 377 | 185 | 192 | 49.07% | 47.92% | 49.07% | 0.93 pp | -7 | 38 | -0.18 |
| BTC Daily | transformer | Transformer | 605 | 297 | 308 | 49.09% | 50.42% | 49.79% | 0.91 pp | -11 | 38 | -0.29 |
| BTC Daily | mlp_sklearn | MLPClassifier | 605 | 296 | 309 | 48.93% | 48.33% | 50.21% | 1.07 pp | -13 | 38 | -0.34 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 38 | 18 | 20 | 47.37% | 47.37% | 47.37% | 2.63 pp | -2 | 4 | -0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 38 | 18 | 20 | 47.37% | 47.37% | 47.37% | 2.63 pp | -2 | 4 | -0.50 |
| BTC Market Hours | transformer | Transformer | 377 | 177 | 200 | 46.95% | 45.00% | 46.95% | 3.05 pp | -23 | 38 | -0.61 |
| BTC Market Hours | nn | NN | 377 | 174 | 203 | 46.15% | 48.75% | 46.15% | 3.85 pp | -29 | 38 | -0.76 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 783 | 372 | 411 | 47.51% | 45.42% | 47.92% | 2.49 pp | -39 | 43 | -0.91 |
| BTC Daily | nn | NN | 605 | 284 | 321 | 46.94% | 44.58% | 48.12% | 3.06 pp | -37 | 38 | -0.97 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 431 | 196 | 235 | 45.48% | 45.42% | 45.48% | 4.52 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | nn | NN | 431 | 196 | 235 | 45.48% | 47.08% | 45.48% | 4.52 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | transformer | Transformer | 431 | 196 | 235 | 45.48% | 47.50% | 45.48% | 4.52 pp | -39 | 38 | -1.03 |
| BTC Hourly | transformer | Transformer | 783 | 368 | 415 | 47.00% | 43.33% | 46.04% | 3.00 pp | -47 | 43 | -1.09 |
| BTC Market Hours | lstm | LSTM | 377 | 161 | 216 | 42.71% | 43.33% | 42.71% | 7.29 pp | -55 | 38 | -1.45 |
| BTC Market Hours | rf | RandomForest | 377 | 160 | 217 | 42.44% | 40.42% | 42.44% | 7.56 pp | -57 | 38 | -1.50 |
| Consolidated Hourly | transformer | Transformer | 38 | 16 | 22 | 42.11% | 42.11% | 42.11% | 7.89 pp | -6 | 4 | -1.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 38 | 16 | 22 | 42.11% | 42.11% | 42.11% | 7.89 pp | -6 | 4 | -1.50 |
| BTC Daily | lstm | LSTM | 605 | 270 | 335 | 44.63% | 43.75% | 44.38% | 5.37 pp | -65 | 38 | -1.71 |
| BTC Hourly | nn | NN | 783 | 352 | 431 | 44.96% | 40.42% | 45.83% | 5.04 pp | -79 | 43 | -1.84 |
| BTC Hourly | rf | RandomForest | 783 | 350 | 433 | 44.70% | 43.75% | 44.38% | 5.30 pp | -83 | 43 | -1.93 |
| BTC Market Hours | xgb | XGBoost | 377 | 151 | 226 | 40.05% | 39.17% | 40.05% | 9.95 pp | -75 | 38 | -1.97 |
| Consolidated Hourly | xgb | XGBoost | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 4 | -2.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 4 | -2.00 |
| BTC Hourly | lstm | LSTM | 783 | 346 | 437 | 44.19% | 43.33% | 45.42% | 5.81 pp | -91 | 43 | -2.12 |
| BTC Daily | rf | RandomForest | 605 | 261 | 344 | 43.14% | 44.17% | 43.54% | 6.86 pp | -83 | 38 | -2.18 |
| BTC Market Hours Daily | rf | RandomForest | 431 | 174 | 257 | 40.37% | 39.58% | 40.37% | 9.63 pp | -83 | 38 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 431 | 170 | 261 | 39.44% | 38.33% | 39.44% | 10.56 pp | -91 | 38 | -2.39 |
| BTC Market Hours Daily | xgb | XGBoost | 431 | 170 | 261 | 39.44% | 38.75% | 39.44% | 10.56 pp | -91 | 38 | -2.39 |
| BTC Hourly | xgb | XGBoost | 783 | 336 | 447 | 42.91% | 40.42% | 44.38% | 7.09 pp | -111 | 43 | -2.58 |
| BTC Daily | xgb | XGBoost | 615 | 246 | 369 | 40.00% | 34.58% | 40.42% | 10.00 pp | -123 | 38 | -3.24 |
| Consolidated Hourly | nn | NN | 38 | 12 | 26 | 31.58% | 31.58% | 31.58% | 18.42 pp | -14 | 4 | -3.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 38 | 12 | 26 | 31.58% | 31.58% | 31.58% | 18.42 pp | -14 | 4 | -3.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 783 | 372 | 411 | 47.51% | 45.42% | 47.92% | 2.49 pp | -39 | 43 | -0.91 |
| BTC Hourly | transformer | Transformer | 783 | 368 | 415 | 47.00% | 43.33% | 46.04% | 3.00 pp | -47 | 43 | -1.09 |
| BTC Hourly | nn | NN | 783 | 352 | 431 | 44.96% | 40.42% | 45.83% | 5.04 pp | -79 | 43 | -1.84 |
| BTC Hourly | rf | RandomForest | 783 | 350 | 433 | 44.70% | 43.75% | 44.38% | 5.30 pp | -83 | 43 | -1.93 |
| BTC Hourly | lstm | LSTM | 783 | 346 | 437 | 44.19% | 43.33% | 45.42% | 5.81 pp | -91 | 43 | -2.12 |
| BTC Hourly | xgb | XGBoost | 783 | 336 | 447 | 42.91% | 40.42% | 44.38% | 7.09 pp | -111 | 43 | -2.58 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 605 | 297 | 308 | 49.09% | 50.42% | 49.79% | 0.91 pp | -11 | 38 | -0.29 |
| BTC Daily | mlp_sklearn | MLPClassifier | 605 | 296 | 309 | 48.93% | 48.33% | 50.21% | 1.07 pp | -13 | 38 | -0.34 |
| BTC Daily | nn | NN | 605 | 284 | 321 | 46.94% | 44.58% | 48.12% | 3.06 pp | -37 | 38 | -0.97 |
| BTC Daily | lstm | LSTM | 605 | 270 | 335 | 44.63% | 43.75% | 44.38% | 5.37 pp | -65 | 38 | -1.71 |
| BTC Daily | rf | RandomForest | 605 | 261 | 344 | 43.14% | 44.17% | 43.54% | 6.86 pp | -83 | 38 | -2.18 |
| BTC Daily | xgb | XGBoost | 615 | 246 | 369 | 40.00% | 34.58% | 40.42% | 10.00 pp | -123 | 38 | -3.24 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 377 | 185 | 192 | 49.07% | 47.92% | 49.07% | 0.93 pp | -7 | 38 | -0.18 |
| BTC Market Hours | transformer | Transformer | 377 | 177 | 200 | 46.95% | 45.00% | 46.95% | 3.05 pp | -23 | 38 | -0.61 |
| BTC Market Hours | nn | NN | 377 | 174 | 203 | 46.15% | 48.75% | 46.15% | 3.85 pp | -29 | 38 | -0.76 |
| BTC Market Hours | lstm | LSTM | 377 | 161 | 216 | 42.71% | 43.33% | 42.71% | 7.29 pp | -55 | 38 | -1.45 |
| BTC Market Hours | rf | RandomForest | 377 | 160 | 217 | 42.44% | 40.42% | 42.44% | 7.56 pp | -57 | 38 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 377 | 151 | 226 | 40.05% | 39.17% | 40.05% | 9.95 pp | -75 | 38 | -1.97 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 431 | 196 | 235 | 45.48% | 45.42% | 45.48% | 4.52 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | nn | NN | 431 | 196 | 235 | 45.48% | 47.08% | 45.48% | 4.52 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | transformer | Transformer | 431 | 196 | 235 | 45.48% | 47.50% | 45.48% | 4.52 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | rf | RandomForest | 431 | 174 | 257 | 40.37% | 39.58% | 40.37% | 9.63 pp | -83 | 38 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 431 | 170 | 261 | 39.44% | 38.33% | 39.44% | 10.56 pp | -91 | 38 | -2.39 |
| BTC Market Hours Daily | xgb | XGBoost | 431 | 170 | 261 | 39.44% | 38.75% | 39.44% | 10.56 pp | -91 | 38 | -2.39 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 38 | 22 | 16 | 57.89% | 57.89% | 57.89% | 7.89 pp | 6 | 4 | 1.50 |
| Consolidated Hourly | lstm | LSTM | 38 | 20 | 18 | 52.63% | 52.63% | 52.63% | 2.63 pp | 2 | 4 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 38 | 18 | 20 | 47.37% | 47.37% | 47.37% | 2.63 pp | -2 | 4 | -0.50 |
| Consolidated Hourly | transformer | Transformer | 38 | 16 | 22 | 42.11% | 42.11% | 42.11% | 7.89 pp | -6 | 4 | -1.50 |
| Consolidated Hourly | xgb | XGBoost | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 4 | -2.00 |
| Consolidated Hourly | nn | NN | 38 | 12 | 26 | 31.58% | 31.58% | 31.58% | 18.42 pp | -14 | 4 | -3.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 38 | 22 | 16 | 57.89% | 57.89% | 57.89% | 7.89 pp | 6 | 4 | 1.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 38 | 20 | 18 | 52.63% | 52.63% | 52.63% | 2.63 pp | 2 | 4 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 38 | 18 | 20 | 47.37% | 47.37% | 47.37% | 2.63 pp | -2 | 4 | -0.50 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 38 | 16 | 22 | 42.11% | 42.11% | 42.11% | 7.89 pp | -6 | 4 | -1.50 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 38 | 15 | 23 | 39.47% | 39.47% | 39.47% | 10.53 pp | -8 | 4 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 38 | 12 | 26 | 31.58% | 31.58% | 31.58% | 18.42 pp | -14 | 4 | -3.50 |

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
