# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T22:57:48.487087+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 828 | 291 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 991 | 626 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 21:00:00+00:00 | 578 | 388 | 189 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 21:00:00+00:00 | 580 | 442 | 136 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 10:00:00+00:00 | 48 | 48 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 10:00:00+00:00 | 48 | 48 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 10:00:00+00:00 | 48 | 0 | 48 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 10:00:00+00:00 | 48 | 0 | 48 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 48 | 27 | 21 | 56.25% | 56.25% | 56.25% | 6.25 pp | 6 | 5 | 1.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 48 | 27 | 21 | 56.25% | 56.25% | 56.25% | 6.25 pp | 6 | 5 | 1.20 |
| Consolidated Hourly | lstm | LSTM | 48 | 25 | 23 | 52.08% | 52.08% | 52.08% | 2.08 pp | 2 | 5 | 0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 48 | 25 | 23 | 52.08% | 52.08% | 52.08% | 2.08 pp | 2 | 5 | 0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 48 | 24 | 24 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 48 | 24 | 24 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 388 | 192 | 196 | 49.48% | 47.92% | 49.48% | 0.52 pp | -4 | 39 | -0.10 |
| BTC Daily | transformer | Transformer | 616 | 303 | 313 | 49.19% | 49.17% | 50.21% | 0.81 pp | -10 | 39 | -0.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 616 | 301 | 315 | 48.86% | 47.50% | 50.21% | 1.14 pp | -14 | 39 | -0.36 |
| BTC Market Hours | nn | NN | 388 | 181 | 207 | 46.65% | 49.17% | 46.65% | 3.35 pp | -26 | 39 | -0.67 |
| BTC Market Hours | transformer | Transformer | 388 | 181 | 207 | 46.65% | 44.58% | 46.65% | 3.35 pp | -26 | 39 | -0.67 |
| Consolidated Hourly | transformer | Transformer | 48 | 22 | 26 | 45.83% | 45.83% | 45.83% | 4.17 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 48 | 22 | 26 | 45.83% | 45.83% | 45.83% | 4.17 pp | -4 | 5 | -0.80 |
| BTC Market Hours Daily | transformer | Transformer | 442 | 204 | 238 | 46.15% | 47.92% | 46.15% | 3.85 pp | -34 | 39 | -0.87 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 442 | 203 | 239 | 45.93% | 46.25% | 45.93% | 4.07 pp | -36 | 39 | -0.92 |
| BTC Daily | nn | NN | 616 | 289 | 327 | 46.92% | 43.33% | 48.96% | 3.08 pp | -38 | 39 | -0.97 |
| BTC Hourly | transformer | Transformer | 794 | 375 | 419 | 47.23% | 44.58% | 46.67% | 2.77 pp | -44 | 43 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 794 | 374 | 420 | 47.10% | 44.58% | 46.88% | 2.90 pp | -46 | 43 | -1.07 |
| BTC Market Hours Daily | nn | NN | 442 | 200 | 242 | 45.25% | 45.83% | 45.25% | 4.75 pp | -42 | 39 | -1.08 |
| BTC Market Hours | lstm | LSTM | 388 | 168 | 220 | 43.30% | 43.75% | 43.30% | 6.70 pp | -52 | 39 | -1.33 |
| BTC Market Hours | rf | RandomForest | 388 | 166 | 222 | 42.78% | 40.42% | 42.78% | 7.22 pp | -56 | 39 | -1.44 |
| Consolidated Hourly | xgb | XGBoost | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 5 | -1.60 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 5 | -1.60 |
| BTC Daily | lstm | LSTM | 616 | 276 | 340 | 44.81% | 44.17% | 44.79% | 5.19 pp | -64 | 39 | -1.64 |
| BTC Hourly | nn | NN | 794 | 358 | 436 | 45.09% | 40.83% | 45.42% | 4.91 pp | -78 | 43 | -1.81 |
| BTC Market Hours | xgb | XGBoost | 388 | 158 | 230 | 40.72% | 38.75% | 40.72% | 9.28 pp | -72 | 39 | -1.85 |
| BTC Hourly | rf | RandomForest | 794 | 353 | 441 | 44.46% | 42.50% | 43.96% | 5.54 pp | -88 | 43 | -2.05 |
| BTC Market Hours Daily | rf | RandomForest | 442 | 181 | 261 | 40.95% | 40.00% | 40.95% | 9.05 pp | -80 | 39 | -2.05 |
| BTC Hourly | lstm | LSTM | 794 | 350 | 444 | 44.08% | 44.17% | 45.62% | 5.92 pp | -94 | 43 | -2.19 |
| BTC Daily | rf | RandomForest | 616 | 263 | 353 | 42.69% | 42.50% | 43.33% | 7.31 pp | -90 | 39 | -2.31 |
| BTC Market Hours Daily | lstm | LSTM | 442 | 176 | 266 | 39.82% | 37.50% | 39.82% | 10.18 pp | -90 | 39 | -2.31 |
| BTC Market Hours Daily | xgb | XGBoost | 442 | 174 | 268 | 39.37% | 37.92% | 39.37% | 10.63 pp | -94 | 39 | -2.41 |
| BTC Hourly | xgb | XGBoost | 794 | 338 | 456 | 42.57% | 39.58% | 44.17% | 7.43 pp | -118 | 43 | -2.74 |
| Consolidated Hourly | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 5 | -2.80 |
| Consolidated Daily/Hourly Refresh | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 5 | -2.80 |
| BTC Daily | xgb | XGBoost | 626 | 248 | 378 | 39.62% | 32.92% | 40.00% | 10.38 pp | -130 | 39 | -3.33 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 794 | 375 | 419 | 47.23% | 44.58% | 46.67% | 2.77 pp | -44 | 43 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 794 | 374 | 420 | 47.10% | 44.58% | 46.88% | 2.90 pp | -46 | 43 | -1.07 |
| BTC Hourly | nn | NN | 794 | 358 | 436 | 45.09% | 40.83% | 45.42% | 4.91 pp | -78 | 43 | -1.81 |
| BTC Hourly | rf | RandomForest | 794 | 353 | 441 | 44.46% | 42.50% | 43.96% | 5.54 pp | -88 | 43 | -2.05 |
| BTC Hourly | lstm | LSTM | 794 | 350 | 444 | 44.08% | 44.17% | 45.62% | 5.92 pp | -94 | 43 | -2.19 |
| BTC Hourly | xgb | XGBoost | 794 | 338 | 456 | 42.57% | 39.58% | 44.17% | 7.43 pp | -118 | 43 | -2.74 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 616 | 303 | 313 | 49.19% | 49.17% | 50.21% | 0.81 pp | -10 | 39 | -0.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 616 | 301 | 315 | 48.86% | 47.50% | 50.21% | 1.14 pp | -14 | 39 | -0.36 |
| BTC Daily | nn | NN | 616 | 289 | 327 | 46.92% | 43.33% | 48.96% | 3.08 pp | -38 | 39 | -0.97 |
| BTC Daily | lstm | LSTM | 616 | 276 | 340 | 44.81% | 44.17% | 44.79% | 5.19 pp | -64 | 39 | -1.64 |
| BTC Daily | rf | RandomForest | 616 | 263 | 353 | 42.69% | 42.50% | 43.33% | 7.31 pp | -90 | 39 | -2.31 |
| BTC Daily | xgb | XGBoost | 626 | 248 | 378 | 39.62% | 32.92% | 40.00% | 10.38 pp | -130 | 39 | -3.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 388 | 192 | 196 | 49.48% | 47.92% | 49.48% | 0.52 pp | -4 | 39 | -0.10 |
| BTC Market Hours | nn | NN | 388 | 181 | 207 | 46.65% | 49.17% | 46.65% | 3.35 pp | -26 | 39 | -0.67 |
| BTC Market Hours | transformer | Transformer | 388 | 181 | 207 | 46.65% | 44.58% | 46.65% | 3.35 pp | -26 | 39 | -0.67 |
| BTC Market Hours | lstm | LSTM | 388 | 168 | 220 | 43.30% | 43.75% | 43.30% | 6.70 pp | -52 | 39 | -1.33 |
| BTC Market Hours | rf | RandomForest | 388 | 166 | 222 | 42.78% | 40.42% | 42.78% | 7.22 pp | -56 | 39 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 388 | 158 | 230 | 40.72% | 38.75% | 40.72% | 9.28 pp | -72 | 39 | -1.85 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 442 | 204 | 238 | 46.15% | 47.92% | 46.15% | 3.85 pp | -34 | 39 | -0.87 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 442 | 203 | 239 | 45.93% | 46.25% | 45.93% | 4.07 pp | -36 | 39 | -0.92 |
| BTC Market Hours Daily | nn | NN | 442 | 200 | 242 | 45.25% | 45.83% | 45.25% | 4.75 pp | -42 | 39 | -1.08 |
| BTC Market Hours Daily | rf | RandomForest | 442 | 181 | 261 | 40.95% | 40.00% | 40.95% | 9.05 pp | -80 | 39 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 442 | 176 | 266 | 39.82% | 37.50% | 39.82% | 10.18 pp | -90 | 39 | -2.31 |
| BTC Market Hours Daily | xgb | XGBoost | 442 | 174 | 268 | 39.37% | 37.92% | 39.37% | 10.63 pp | -94 | 39 | -2.41 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 48 | 27 | 21 | 56.25% | 56.25% | 56.25% | 6.25 pp | 6 | 5 | 1.20 |
| Consolidated Hourly | lstm | LSTM | 48 | 25 | 23 | 52.08% | 52.08% | 52.08% | 2.08 pp | 2 | 5 | 0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 48 | 24 | 24 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Hourly | transformer | Transformer | 48 | 22 | 26 | 45.83% | 45.83% | 45.83% | 4.17 pp | -4 | 5 | -0.80 |
| Consolidated Hourly | xgb | XGBoost | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 5 | -1.60 |
| Consolidated Hourly | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 5 | -2.80 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 48 | 27 | 21 | 56.25% | 56.25% | 56.25% | 6.25 pp | 6 | 5 | 1.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 48 | 25 | 23 | 52.08% | 52.08% | 52.08% | 2.08 pp | 2 | 5 | 0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 48 | 24 | 24 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 48 | 22 | 26 | 45.83% | 45.83% | 45.83% | 4.17 pp | -4 | 5 | -0.80 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 48 | 20 | 28 | 41.67% | 41.67% | 41.67% | 8.33 pp | -8 | 5 | -1.60 |
| Consolidated Daily/Hourly Refresh | nn | NN | 48 | 17 | 31 | 35.42% | 35.42% | 35.42% | 14.58 pp | -14 | 5 | -2.80 |

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
