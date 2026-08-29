# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T09:14:27.990156+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1124 | 836 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1000 | 635 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 590 | 397 | 192 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 591 | 450 | 139 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 05:00:00+00:00 | 54 | 54 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 05:00:00+00:00 | 54 | 54 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 05:00:00+00:00 | 54 | 0 | 54 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 05:00:00+00:00 | 54 | 0 | 54 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 54 | 31 | 23 | 57.41% | 57.41% | 57.41% | 7.41 pp | 8 | 6 | 1.33 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 54 | 31 | 23 | 57.41% | 57.41% | 57.41% | 7.41 pp | 8 | 6 | 1.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 54 | 28 | 26 | 51.85% | 51.85% | 51.85% | 1.85 pp | 2 | 6 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 54 | 28 | 26 | 51.85% | 51.85% | 51.85% | 1.85 pp | 2 | 6 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 397 | 195 | 202 | 49.12% | 47.50% | 49.12% | 0.88 pp | -7 | 40 | -0.17 |
| BTC Daily | transformer | Transformer | 625 | 307 | 318 | 49.12% | 47.50% | 50.00% | 0.88 pp | -11 | 39 | -0.28 |
| BTC Daily | mlp_sklearn | MLPClassifier | 625 | 305 | 320 | 48.80% | 46.67% | 50.21% | 1.20 pp | -15 | 39 | -0.38 |
| BTC Market Hours | nn | NN | 397 | 186 | 211 | 46.85% | 49.17% | 46.85% | 3.15 pp | -25 | 40 | -0.62 |
| BTC Market Hours | transformer | Transformer | 397 | 185 | 212 | 46.60% | 42.92% | 46.60% | 3.40 pp | -27 | 40 | -0.68 |
| BTC Market Hours Daily | transformer | Transformer | 450 | 208 | 242 | 46.22% | 47.92% | 46.22% | 3.78 pp | -34 | 40 | -0.85 |
| BTC Daily | nn | NN | 625 | 294 | 331 | 47.04% | 43.75% | 48.96% | 2.96 pp | -37 | 39 | -0.95 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 450 | 206 | 244 | 45.78% | 45.00% | 45.78% | 4.22 pp | -38 | 40 | -0.95 |
| BTC Market Hours Daily | nn | NN | 450 | 204 | 246 | 45.33% | 45.42% | 45.33% | 4.67 pp | -42 | 40 | -1.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 802 | 378 | 424 | 47.13% | 44.58% | 46.88% | 2.87 pp | -46 | 43 | -1.07 |
| BTC Hourly | transformer | Transformer | 802 | 378 | 424 | 47.13% | 45.00% | 46.46% | 2.87 pp | -46 | 43 | -1.07 |
| BTC Market Hours | lstm | LSTM | 397 | 173 | 224 | 43.58% | 43.75% | 43.58% | 6.42 pp | -51 | 40 | -1.27 |
| Consolidated Hourly | transformer | Transformer | 54 | 23 | 31 | 42.59% | 42.59% | 42.59% | 7.41 pp | -8 | 6 | -1.33 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 54 | 23 | 31 | 42.59% | 42.59% | 42.59% | 7.41 pp | -8 | 6 | -1.33 |
| BTC Market Hours | rf | RandomForest | 397 | 169 | 228 | 42.57% | 41.25% | 42.57% | 7.43 pp | -59 | 40 | -1.48 |
| Consolidated Hourly | xgb | XGBoost | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 6 | -1.67 |
| BTC Daily | lstm | LSTM | 625 | 278 | 347 | 44.48% | 42.50% | 44.17% | 5.52 pp | -69 | 39 | -1.77 |
| BTC Market Hours | xgb | XGBoost | 397 | 161 | 236 | 40.55% | 38.75% | 40.55% | 9.45 pp | -75 | 40 | -1.88 |
| BTC Hourly | nn | NN | 802 | 360 | 442 | 44.89% | 40.42% | 45.00% | 5.11 pp | -82 | 43 | -1.91 |
| BTC Hourly | rf | RandomForest | 802 | 357 | 445 | 44.51% | 43.75% | 44.17% | 5.49 pp | -88 | 43 | -2.05 |
| BTC Market Hours Daily | rf | RandomForest | 450 | 183 | 267 | 40.67% | 39.58% | 40.67% | 9.33 pp | -84 | 40 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 450 | 181 | 269 | 40.22% | 38.33% | 40.22% | 9.78 pp | -88 | 40 | -2.20 |
| BTC Hourly | lstm | LSTM | 802 | 352 | 450 | 43.89% | 43.33% | 45.00% | 6.11 pp | -98 | 43 | -2.28 |
| BTC Daily | rf | RandomForest | 625 | 268 | 357 | 42.88% | 42.50% | 43.75% | 7.12 pp | -89 | 39 | -2.28 |
| BTC Market Hours Daily | xgb | XGBoost | 450 | 177 | 273 | 39.33% | 37.50% | 39.33% | 10.67 pp | -96 | 40 | -2.40 |
| Consolidated Hourly | nn | NN | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 6 | -2.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 6 | -2.67 |
| BTC Hourly | xgb | XGBoost | 802 | 341 | 461 | 42.52% | 40.00% | 43.75% | 7.48 pp | -120 | 43 | -2.79 |
| BTC Daily | xgb | XGBoost | 635 | 250 | 385 | 39.37% | 32.08% | 39.79% | 10.63 pp | -135 | 39 | -3.46 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 802 | 378 | 424 | 47.13% | 44.58% | 46.88% | 2.87 pp | -46 | 43 | -1.07 |
| BTC Hourly | transformer | Transformer | 802 | 378 | 424 | 47.13% | 45.00% | 46.46% | 2.87 pp | -46 | 43 | -1.07 |
| BTC Hourly | nn | NN | 802 | 360 | 442 | 44.89% | 40.42% | 45.00% | 5.11 pp | -82 | 43 | -1.91 |
| BTC Hourly | rf | RandomForest | 802 | 357 | 445 | 44.51% | 43.75% | 44.17% | 5.49 pp | -88 | 43 | -2.05 |
| BTC Hourly | lstm | LSTM | 802 | 352 | 450 | 43.89% | 43.33% | 45.00% | 6.11 pp | -98 | 43 | -2.28 |
| BTC Hourly | xgb | XGBoost | 802 | 341 | 461 | 42.52% | 40.00% | 43.75% | 7.48 pp | -120 | 43 | -2.79 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 625 | 307 | 318 | 49.12% | 47.50% | 50.00% | 0.88 pp | -11 | 39 | -0.28 |
| BTC Daily | mlp_sklearn | MLPClassifier | 625 | 305 | 320 | 48.80% | 46.67% | 50.21% | 1.20 pp | -15 | 39 | -0.38 |
| BTC Daily | nn | NN | 625 | 294 | 331 | 47.04% | 43.75% | 48.96% | 2.96 pp | -37 | 39 | -0.95 |
| BTC Daily | lstm | LSTM | 625 | 278 | 347 | 44.48% | 42.50% | 44.17% | 5.52 pp | -69 | 39 | -1.77 |
| BTC Daily | rf | RandomForest | 625 | 268 | 357 | 42.88% | 42.50% | 43.75% | 7.12 pp | -89 | 39 | -2.28 |
| BTC Daily | xgb | XGBoost | 635 | 250 | 385 | 39.37% | 32.08% | 39.79% | 10.63 pp | -135 | 39 | -3.46 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 397 | 195 | 202 | 49.12% | 47.50% | 49.12% | 0.88 pp | -7 | 40 | -0.17 |
| BTC Market Hours | nn | NN | 397 | 186 | 211 | 46.85% | 49.17% | 46.85% | 3.15 pp | -25 | 40 | -0.62 |
| BTC Market Hours | transformer | Transformer | 397 | 185 | 212 | 46.60% | 42.92% | 46.60% | 3.40 pp | -27 | 40 | -0.68 |
| BTC Market Hours | lstm | LSTM | 397 | 173 | 224 | 43.58% | 43.75% | 43.58% | 6.42 pp | -51 | 40 | -1.27 |
| BTC Market Hours | rf | RandomForest | 397 | 169 | 228 | 42.57% | 41.25% | 42.57% | 7.43 pp | -59 | 40 | -1.48 |
| BTC Market Hours | xgb | XGBoost | 397 | 161 | 236 | 40.55% | 38.75% | 40.55% | 9.45 pp | -75 | 40 | -1.88 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 450 | 208 | 242 | 46.22% | 47.92% | 46.22% | 3.78 pp | -34 | 40 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 450 | 206 | 244 | 45.78% | 45.00% | 45.78% | 4.22 pp | -38 | 40 | -0.95 |
| BTC Market Hours Daily | nn | NN | 450 | 204 | 246 | 45.33% | 45.42% | 45.33% | 4.67 pp | -42 | 40 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 450 | 183 | 267 | 40.67% | 39.58% | 40.67% | 9.33 pp | -84 | 40 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 450 | 181 | 269 | 40.22% | 38.33% | 40.22% | 9.78 pp | -88 | 40 | -2.20 |
| BTC Market Hours Daily | xgb | XGBoost | 450 | 177 | 273 | 39.33% | 37.50% | 39.33% | 10.67 pp | -96 | 40 | -2.40 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 54 | 31 | 23 | 57.41% | 57.41% | 57.41% | 7.41 pp | 8 | 6 | 1.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 54 | 28 | 26 | 51.85% | 51.85% | 51.85% | 1.85 pp | 2 | 6 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | transformer | Transformer | 54 | 23 | 31 | 42.59% | 42.59% | 42.59% | 7.41 pp | -8 | 6 | -1.33 |
| Consolidated Hourly | xgb | XGBoost | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 6 | -1.67 |
| Consolidated Hourly | nn | NN | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 6 | -2.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 54 | 31 | 23 | 57.41% | 57.41% | 57.41% | 7.41 pp | 8 | 6 | 1.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 54 | 28 | 26 | 51.85% | 51.85% | 51.85% | 1.85 pp | 2 | 6 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 54 | 27 | 27 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 54 | 23 | 31 | 42.59% | 42.59% | 42.59% | 7.41 pp | -8 | 6 | -1.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 54 | 22 | 32 | 40.74% | 40.74% | 40.74% | 9.26 pp | -10 | 6 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 54 | 19 | 35 | 35.19% | 35.19% | 35.19% | 14.81 pp | -16 | 6 | -2.67 |

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
