# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T03:09:23.460916+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 831 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 995 | 630 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 585 | 392 | 192 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 586 | 445 | 139 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 11:00:00+00:00 | 49 | 49 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 11:00:00+00:00 | 49 | 49 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 11:00:00+00:00 | 49 | 0 | 49 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 11:00:00+00:00 | 49 | 0 | 49 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 49 | 27 | 22 | 55.10% | 55.10% | 55.10% | 5.10 pp | 5 | 5 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 49 | 27 | 22 | 55.10% | 55.10% | 55.10% | 5.10 pp | 5 | 5 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 49 | 25 | 24 | 51.02% | 51.02% | 51.02% | 1.02 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 49 | 25 | 24 | 51.02% | 51.02% | 51.02% | 1.02 pp | 1 | 5 | 0.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 392 | 193 | 199 | 49.23% | 48.33% | 49.23% | 0.77 pp | -6 | 40 | -0.15 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 5 | -0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 5 | -0.20 |
| BTC Daily | transformer | Transformer | 620 | 305 | 315 | 49.19% | 48.75% | 50.21% | 0.81 pp | -10 | 39 | -0.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 620 | 303 | 317 | 48.87% | 47.50% | 50.21% | 1.13 pp | -14 | 39 | -0.36 |
| BTC Market Hours | nn | NN | 392 | 183 | 209 | 46.68% | 48.75% | 46.68% | 3.32 pp | -26 | 40 | -0.65 |
| BTC Market Hours | transformer | Transformer | 392 | 183 | 209 | 46.68% | 44.17% | 46.68% | 3.32 pp | -26 | 40 | -0.65 |
| BTC Market Hours Daily | transformer | Transformer | 445 | 207 | 238 | 46.52% | 48.75% | 46.52% | 3.48 pp | -31 | 39 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 445 | 205 | 240 | 46.07% | 46.25% | 46.07% | 3.93 pp | -35 | 39 | -0.90 |
| BTC Daily | nn | NN | 620 | 292 | 328 | 47.10% | 43.75% | 49.17% | 2.90 pp | -36 | 39 | -0.92 |
| Consolidated Hourly | transformer | Transformer | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 5 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 797 | 376 | 421 | 47.18% | 45.00% | 47.08% | 2.82 pp | -45 | 43 | -1.05 |
| BTC Hourly | transformer | Transformer | 797 | 376 | 421 | 47.18% | 45.00% | 46.67% | 2.82 pp | -45 | 43 | -1.05 |
| BTC Market Hours Daily | nn | NN | 445 | 202 | 243 | 45.39% | 46.25% | 45.39% | 4.61 pp | -41 | 39 | -1.05 |
| BTC Market Hours | lstm | LSTM | 392 | 170 | 222 | 43.37% | 43.33% | 43.37% | 6.63 pp | -52 | 40 | -1.30 |
| BTC Market Hours | rf | RandomForest | 392 | 167 | 225 | 42.60% | 40.83% | 42.60% | 7.40 pp | -58 | 40 | -1.45 |
| BTC Daily | lstm | LSTM | 620 | 276 | 344 | 44.52% | 43.33% | 44.38% | 5.48 pp | -68 | 39 | -1.74 |
| BTC Market Hours | xgb | XGBoost | 392 | 160 | 232 | 40.82% | 39.17% | 40.82% | 9.18 pp | -72 | 40 | -1.80 |
| Consolidated Hourly | xgb | XGBoost | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 5 | -1.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 5 | -1.80 |
| BTC Hourly | nn | NN | 797 | 359 | 438 | 45.04% | 41.25% | 45.42% | 4.96 pp | -79 | 43 | -1.84 |
| BTC Hourly | rf | RandomForest | 797 | 355 | 442 | 44.54% | 43.33% | 44.17% | 5.46 pp | -87 | 43 | -2.02 |
| BTC Market Hours Daily | rf | RandomForest | 445 | 182 | 263 | 40.90% | 40.00% | 40.90% | 9.10 pp | -81 | 39 | -2.08 |
| BTC Hourly | lstm | LSTM | 797 | 352 | 445 | 44.17% | 45.00% | 45.83% | 5.83 pp | -93 | 43 | -2.16 |
| BTC Daily | rf | RandomForest | 620 | 266 | 354 | 42.90% | 42.92% | 43.96% | 7.10 pp | -88 | 39 | -2.26 |
| BTC Market Hours Daily | lstm | LSTM | 445 | 178 | 267 | 40.00% | 38.33% | 40.00% | 10.00 pp | -89 | 39 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 445 | 176 | 269 | 39.55% | 37.92% | 39.55% | 10.45 pp | -93 | 39 | -2.38 |
| BTC Hourly | xgb | XGBoost | 797 | 340 | 457 | 42.66% | 40.00% | 44.38% | 7.34 pp | -117 | 43 | -2.72 |
| Consolidated Hourly | nn | NN | 49 | 17 | 32 | 34.69% | 34.69% | 34.69% | 15.31 pp | -15 | 5 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 49 | 17 | 32 | 34.69% | 34.69% | 34.69% | 15.31 pp | -15 | 5 | -3.00 |
| BTC Daily | xgb | XGBoost | 630 | 250 | 380 | 39.68% | 32.92% | 40.21% | 10.32 pp | -130 | 39 | -3.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 797 | 376 | 421 | 47.18% | 45.00% | 47.08% | 2.82 pp | -45 | 43 | -1.05 |
| BTC Hourly | transformer | Transformer | 797 | 376 | 421 | 47.18% | 45.00% | 46.67% | 2.82 pp | -45 | 43 | -1.05 |
| BTC Hourly | nn | NN | 797 | 359 | 438 | 45.04% | 41.25% | 45.42% | 4.96 pp | -79 | 43 | -1.84 |
| BTC Hourly | rf | RandomForest | 797 | 355 | 442 | 44.54% | 43.33% | 44.17% | 5.46 pp | -87 | 43 | -2.02 |
| BTC Hourly | lstm | LSTM | 797 | 352 | 445 | 44.17% | 45.00% | 45.83% | 5.83 pp | -93 | 43 | -2.16 |
| BTC Hourly | xgb | XGBoost | 797 | 340 | 457 | 42.66% | 40.00% | 44.38% | 7.34 pp | -117 | 43 | -2.72 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 620 | 305 | 315 | 49.19% | 48.75% | 50.21% | 0.81 pp | -10 | 39 | -0.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 620 | 303 | 317 | 48.87% | 47.50% | 50.21% | 1.13 pp | -14 | 39 | -0.36 |
| BTC Daily | nn | NN | 620 | 292 | 328 | 47.10% | 43.75% | 49.17% | 2.90 pp | -36 | 39 | -0.92 |
| BTC Daily | lstm | LSTM | 620 | 276 | 344 | 44.52% | 43.33% | 44.38% | 5.48 pp | -68 | 39 | -1.74 |
| BTC Daily | rf | RandomForest | 620 | 266 | 354 | 42.90% | 42.92% | 43.96% | 7.10 pp | -88 | 39 | -2.26 |
| BTC Daily | xgb | XGBoost | 630 | 250 | 380 | 39.68% | 32.92% | 40.21% | 10.32 pp | -130 | 39 | -3.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 392 | 193 | 199 | 49.23% | 48.33% | 49.23% | 0.77 pp | -6 | 40 | -0.15 |
| BTC Market Hours | nn | NN | 392 | 183 | 209 | 46.68% | 48.75% | 46.68% | 3.32 pp | -26 | 40 | -0.65 |
| BTC Market Hours | transformer | Transformer | 392 | 183 | 209 | 46.68% | 44.17% | 46.68% | 3.32 pp | -26 | 40 | -0.65 |
| BTC Market Hours | lstm | LSTM | 392 | 170 | 222 | 43.37% | 43.33% | 43.37% | 6.63 pp | -52 | 40 | -1.30 |
| BTC Market Hours | rf | RandomForest | 392 | 167 | 225 | 42.60% | 40.83% | 42.60% | 7.40 pp | -58 | 40 | -1.45 |
| BTC Market Hours | xgb | XGBoost | 392 | 160 | 232 | 40.82% | 39.17% | 40.82% | 9.18 pp | -72 | 40 | -1.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 445 | 207 | 238 | 46.52% | 48.75% | 46.52% | 3.48 pp | -31 | 39 | -0.79 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 445 | 205 | 240 | 46.07% | 46.25% | 46.07% | 3.93 pp | -35 | 39 | -0.90 |
| BTC Market Hours Daily | nn | NN | 445 | 202 | 243 | 45.39% | 46.25% | 45.39% | 4.61 pp | -41 | 39 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 445 | 182 | 263 | 40.90% | 40.00% | 40.90% | 9.10 pp | -81 | 39 | -2.08 |
| BTC Market Hours Daily | lstm | LSTM | 445 | 178 | 267 | 40.00% | 38.33% | 40.00% | 10.00 pp | -89 | 39 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 445 | 176 | 269 | 39.55% | 37.92% | 39.55% | 10.45 pp | -93 | 39 | -2.38 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 49 | 27 | 22 | 55.10% | 55.10% | 55.10% | 5.10 pp | 5 | 5 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 49 | 25 | 24 | 51.02% | 51.02% | 51.02% | 1.02 pp | 1 | 5 | 0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | transformer | Transformer | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 5 | -1.80 |
| Consolidated Hourly | nn | NN | 49 | 17 | 32 | 34.69% | 34.69% | 34.69% | 15.31 pp | -15 | 5 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 49 | 27 | 22 | 55.10% | 55.10% | 55.10% | 5.10 pp | 5 | 5 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 49 | 25 | 24 | 51.02% | 51.02% | 51.02% | 1.02 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 49 | 24 | 25 | 48.98% | 48.98% | 48.98% | 1.02 pp | -1 | 5 | -0.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 49 | 22 | 27 | 44.90% | 44.90% | 44.90% | 5.10 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 49 | 20 | 29 | 40.82% | 40.82% | 40.82% | 9.18 pp | -9 | 5 | -1.80 |
| Consolidated Daily/Hourly Refresh | nn | NN | 49 | 17 | 32 | 34.69% | 34.69% | 34.69% | 15.31 pp | -15 | 5 | -3.00 |

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
