# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T06:34:47.627696+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1122 | 834 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 997 | 632 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 587 | 394 | 192 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 589 | 448 | 139 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 03:00:00+00:00 | 52 | 52 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 03:00:00+00:00 | 52 | 52 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 03:00:00+00:00 | 52 | 0 | 52 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-23 03:00:00+00:00 | 52 | 0 | 52 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 52 | 29 | 23 | 55.77% | 55.77% | 55.77% | 5.77 pp | 6 | 6 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 52 | 29 | 23 | 55.77% | 55.77% | 55.77% | 5.77 pp | 6 | 6 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 52 | 27 | 25 | 51.92% | 51.92% | 51.92% | 1.92 pp | 2 | 6 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 52 | 27 | 25 | 51.92% | 51.92% | 51.92% | 1.92 pp | 2 | 6 | 0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 52 | 26 | 26 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 52 | 26 | 26 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 394 | 194 | 200 | 49.24% | 47.92% | 49.24% | 0.76 pp | -6 | 40 | -0.15 |
| BTC Daily | transformer | Transformer | 622 | 305 | 317 | 49.04% | 47.92% | 49.79% | 0.96 pp | -12 | 39 | -0.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 622 | 304 | 318 | 48.87% | 47.08% | 50.21% | 1.13 pp | -14 | 39 | -0.36 |
| BTC Market Hours | nn | NN | 394 | 184 | 210 | 46.70% | 48.75% | 46.70% | 3.30 pp | -26 | 40 | -0.65 |
| BTC Market Hours | transformer | Transformer | 394 | 183 | 211 | 46.45% | 43.33% | 46.45% | 3.55 pp | -28 | 40 | -0.70 |
| BTC Market Hours Daily | transformer | Transformer | 448 | 207 | 241 | 46.21% | 48.33% | 46.21% | 3.79 pp | -34 | 40 | -0.85 |
| BTC Daily | nn | NN | 622 | 293 | 329 | 47.11% | 43.75% | 49.17% | 2.89 pp | -36 | 39 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 448 | 205 | 243 | 45.76% | 45.00% | 45.76% | 4.24 pp | -38 | 40 | -0.95 |
| Consolidated Hourly | transformer | Transformer | 52 | 23 | 29 | 44.23% | 44.23% | 44.23% | 5.77 pp | -6 | 6 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 52 | 23 | 29 | 44.23% | 44.23% | 44.23% | 5.77 pp | -6 | 6 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 800 | 377 | 423 | 47.12% | 44.58% | 47.08% | 2.88 pp | -46 | 43 | -1.07 |
| BTC Hourly | transformer | Transformer | 800 | 377 | 423 | 47.12% | 44.58% | 46.67% | 2.88 pp | -46 | 43 | -1.07 |
| BTC Market Hours Daily | nn | NN | 448 | 202 | 246 | 45.09% | 45.00% | 45.09% | 4.91 pp | -44 | 40 | -1.10 |
| BTC Market Hours | lstm | LSTM | 394 | 172 | 222 | 43.65% | 43.75% | 43.65% | 6.35 pp | -50 | 40 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 52 | 22 | 30 | 42.31% | 42.31% | 42.31% | 7.69 pp | -8 | 6 | -1.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 52 | 22 | 30 | 42.31% | 42.31% | 42.31% | 7.69 pp | -8 | 6 | -1.33 |
| BTC Market Hours | rf | RandomForest | 394 | 167 | 227 | 42.39% | 40.83% | 42.39% | 7.61 pp | -60 | 40 | -1.50 |
| BTC Daily | lstm | LSTM | 622 | 277 | 345 | 44.53% | 43.33% | 44.38% | 5.47 pp | -68 | 39 | -1.74 |
| BTC Market Hours | xgb | XGBoost | 394 | 160 | 234 | 40.61% | 39.17% | 40.61% | 9.39 pp | -74 | 40 | -1.85 |
| BTC Hourly | nn | NN | 800 | 359 | 441 | 44.88% | 40.83% | 45.21% | 5.13 pp | -82 | 43 | -1.91 |
| BTC Hourly | rf | RandomForest | 800 | 356 | 444 | 44.50% | 43.33% | 44.38% | 5.50 pp | -88 | 43 | -2.05 |
| BTC Market Hours Daily | rf | RandomForest | 448 | 182 | 266 | 40.62% | 39.58% | 40.62% | 9.38 pp | -84 | 40 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 448 | 180 | 268 | 40.18% | 38.75% | 40.18% | 9.82 pp | -88 | 40 | -2.20 |
| BTC Hourly | lstm | LSTM | 800 | 352 | 448 | 44.00% | 43.75% | 45.42% | 6.00 pp | -96 | 43 | -2.23 |
| BTC Daily | rf | RandomForest | 622 | 266 | 356 | 42.77% | 42.50% | 43.54% | 7.23 pp | -90 | 39 | -2.31 |
| Consolidated Hourly | nn | NN | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 6 | -2.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 6 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 448 | 176 | 272 | 39.29% | 37.50% | 39.29% | 10.71 pp | -96 | 40 | -2.40 |
| BTC Hourly | xgb | XGBoost | 800 | 340 | 460 | 42.50% | 39.58% | 43.96% | 7.50 pp | -120 | 43 | -2.79 |
| BTC Daily | xgb | XGBoost | 632 | 249 | 383 | 39.40% | 32.50% | 39.79% | 10.60 pp | -134 | 39 | -3.44 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 800 | 377 | 423 | 47.12% | 44.58% | 47.08% | 2.88 pp | -46 | 43 | -1.07 |
| BTC Hourly | transformer | Transformer | 800 | 377 | 423 | 47.12% | 44.58% | 46.67% | 2.88 pp | -46 | 43 | -1.07 |
| BTC Hourly | nn | NN | 800 | 359 | 441 | 44.88% | 40.83% | 45.21% | 5.13 pp | -82 | 43 | -1.91 |
| BTC Hourly | rf | RandomForest | 800 | 356 | 444 | 44.50% | 43.33% | 44.38% | 5.50 pp | -88 | 43 | -2.05 |
| BTC Hourly | lstm | LSTM | 800 | 352 | 448 | 44.00% | 43.75% | 45.42% | 6.00 pp | -96 | 43 | -2.23 |
| BTC Hourly | xgb | XGBoost | 800 | 340 | 460 | 42.50% | 39.58% | 43.96% | 7.50 pp | -120 | 43 | -2.79 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 622 | 305 | 317 | 49.04% | 47.92% | 49.79% | 0.96 pp | -12 | 39 | -0.31 |
| BTC Daily | mlp_sklearn | MLPClassifier | 622 | 304 | 318 | 48.87% | 47.08% | 50.21% | 1.13 pp | -14 | 39 | -0.36 |
| BTC Daily | nn | NN | 622 | 293 | 329 | 47.11% | 43.75% | 49.17% | 2.89 pp | -36 | 39 | -0.92 |
| BTC Daily | lstm | LSTM | 622 | 277 | 345 | 44.53% | 43.33% | 44.38% | 5.47 pp | -68 | 39 | -1.74 |
| BTC Daily | rf | RandomForest | 622 | 266 | 356 | 42.77% | 42.50% | 43.54% | 7.23 pp | -90 | 39 | -2.31 |
| BTC Daily | xgb | XGBoost | 632 | 249 | 383 | 39.40% | 32.50% | 39.79% | 10.60 pp | -134 | 39 | -3.44 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 394 | 194 | 200 | 49.24% | 47.92% | 49.24% | 0.76 pp | -6 | 40 | -0.15 |
| BTC Market Hours | nn | NN | 394 | 184 | 210 | 46.70% | 48.75% | 46.70% | 3.30 pp | -26 | 40 | -0.65 |
| BTC Market Hours | transformer | Transformer | 394 | 183 | 211 | 46.45% | 43.33% | 46.45% | 3.55 pp | -28 | 40 | -0.70 |
| BTC Market Hours | lstm | LSTM | 394 | 172 | 222 | 43.65% | 43.75% | 43.65% | 6.35 pp | -50 | 40 | -1.25 |
| BTC Market Hours | rf | RandomForest | 394 | 167 | 227 | 42.39% | 40.83% | 42.39% | 7.61 pp | -60 | 40 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 394 | 160 | 234 | 40.61% | 39.17% | 40.61% | 9.39 pp | -74 | 40 | -1.85 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 448 | 207 | 241 | 46.21% | 48.33% | 46.21% | 3.79 pp | -34 | 40 | -0.85 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 448 | 205 | 243 | 45.76% | 45.00% | 45.76% | 4.24 pp | -38 | 40 | -0.95 |
| BTC Market Hours Daily | nn | NN | 448 | 202 | 246 | 45.09% | 45.00% | 45.09% | 4.91 pp | -44 | 40 | -1.10 |
| BTC Market Hours Daily | rf | RandomForest | 448 | 182 | 266 | 40.62% | 39.58% | 40.62% | 9.38 pp | -84 | 40 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 448 | 180 | 268 | 40.18% | 38.75% | 40.18% | 9.82 pp | -88 | 40 | -2.20 |
| BTC Market Hours Daily | xgb | XGBoost | 448 | 176 | 272 | 39.29% | 37.50% | 39.29% | 10.71 pp | -96 | 40 | -2.40 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 52 | 29 | 23 | 55.77% | 55.77% | 55.77% | 5.77 pp | 6 | 6 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 52 | 27 | 25 | 51.92% | 51.92% | 51.92% | 1.92 pp | 2 | 6 | 0.33 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 52 | 26 | 26 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Hourly | transformer | Transformer | 52 | 23 | 29 | 44.23% | 44.23% | 44.23% | 5.77 pp | -6 | 6 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 52 | 22 | 30 | 42.31% | 42.31% | 42.31% | 7.69 pp | -8 | 6 | -1.33 |
| Consolidated Hourly | nn | NN | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 6 | -2.33 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 52 | 29 | 23 | 55.77% | 55.77% | 55.77% | 5.77 pp | 6 | 6 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 52 | 27 | 25 | 51.92% | 51.92% | 51.92% | 1.92 pp | 2 | 6 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 52 | 26 | 26 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 6 | 0.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 52 | 23 | 29 | 44.23% | 44.23% | 44.23% | 5.77 pp | -6 | 6 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 52 | 22 | 30 | 42.31% | 42.31% | 42.31% | 7.69 pp | -8 | 6 | -1.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 52 | 19 | 33 | 36.54% | 36.54% | 36.54% | 13.46 pp | -14 | 6 | -2.33 |

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
