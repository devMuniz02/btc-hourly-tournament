# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T22:12:35.921053+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1134 | 846 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1010 | 645 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 21:00:00+00:00 | 610 | 407 | 202 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 21:00:00+00:00 | 611 | 460 | 149 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 02:00:00+00:00 | 62 | 62 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 02:00:00+00:00 | 62 | 62 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 02:00:00+00:00 | 62 | 0 | 62 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 02:00:00+00:00 | 62 | 0 | 62 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 62 | 36 | 26 | 58.06% | 58.06% | 58.06% | 8.06 pp | 10 | 7 | 1.43 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 62 | 36 | 26 | 58.06% | 58.06% | 58.06% | 8.06 pp | 10 | 7 | 1.43 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 62 | 32 | 30 | 51.61% | 51.61% | 51.61% | 1.61 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 62 | 32 | 30 | 51.61% | 51.61% | 51.61% | 1.61 pp | 2 | 7 | 0.29 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 407 | 202 | 205 | 49.63% | 49.17% | 49.63% | 0.37 pp | -3 | 41 | -0.07 |
| Consolidated Hourly | lstm | LSTM | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 7 | -0.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 7 | -0.29 |
| BTC Daily | transformer | Transformer | 635 | 311 | 324 | 48.98% | 47.92% | 49.79% | 1.02 pp | -13 | 39 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 635 | 309 | 326 | 48.66% | 45.83% | 50.00% | 1.34 pp | -17 | 39 | -0.44 |
| BTC Market Hours | nn | NN | 407 | 193 | 214 | 47.42% | 51.25% | 47.42% | 2.58 pp | -21 | 41 | -0.51 |
| BTC Market Hours | transformer | Transformer | 407 | 188 | 219 | 46.19% | 42.50% | 46.19% | 3.81 pp | -31 | 41 | -0.76 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 460 | 213 | 247 | 46.30% | 46.25% | 46.30% | 3.70 pp | -34 | 41 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 460 | 212 | 248 | 46.09% | 47.08% | 46.09% | 3.91 pp | -36 | 41 | -0.88 |
| BTC Market Hours Daily | nn | NN | 460 | 210 | 250 | 45.65% | 45.42% | 45.65% | 4.35 pp | -40 | 41 | -0.98 |
| BTC Hourly | transformer | Transformer | 812 | 384 | 428 | 47.29% | 45.83% | 46.46% | 2.71 pp | -44 | 44 | -1.00 |
| BTC Daily | nn | NN | 635 | 298 | 337 | 46.93% | 42.50% | 48.96% | 3.07 pp | -39 | 39 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 812 | 382 | 430 | 47.04% | 43.75% | 47.08% | 2.96 pp | -48 | 44 | -1.09 |
| Consolidated Hourly | xgb | XGBoost | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 7 | -1.14 |
| BTC Market Hours | lstm | LSTM | 407 | 180 | 227 | 44.23% | 45.42% | 44.23% | 5.77 pp | -47 | 41 | -1.15 |
| BTC Market Hours | rf | RandomForest | 407 | 175 | 232 | 43.00% | 42.50% | 43.00% | 7.00 pp | -57 | 41 | -1.39 |
| Consolidated Hourly | transformer | Transformer | 62 | 26 | 36 | 41.94% | 41.94% | 41.94% | 8.06 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 62 | 26 | 36 | 41.94% | 41.94% | 41.94% | 8.06 pp | -10 | 7 | -1.43 |
| BTC Hourly | nn | NN | 812 | 366 | 446 | 45.07% | 41.25% | 45.00% | 4.93 pp | -80 | 44 | -1.82 |
| BTC Daily | lstm | LSTM | 635 | 281 | 354 | 44.25% | 42.50% | 43.54% | 5.75 pp | -73 | 39 | -1.87 |
| BTC Hourly | rf | RandomForest | 812 | 364 | 448 | 44.83% | 44.58% | 44.79% | 5.17 pp | -84 | 44 | -1.91 |
| BTC Market Hours | xgb | XGBoost | 407 | 164 | 243 | 40.29% | 38.75% | 40.29% | 9.71 pp | -79 | 41 | -1.93 |
| BTC Market Hours Daily | rf | RandomForest | 460 | 189 | 271 | 41.09% | 41.25% | 41.09% | 8.91 pp | -82 | 41 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 460 | 188 | 272 | 40.87% | 40.42% | 40.87% | 9.13 pp | -84 | 41 | -2.05 |
| BTC Hourly | lstm | LSTM | 812 | 355 | 457 | 43.72% | 42.50% | 44.58% | 6.28 pp | -102 | 44 | -2.32 |
| BTC Daily | rf | RandomForest | 635 | 271 | 364 | 42.68% | 42.08% | 43.54% | 7.32 pp | -93 | 39 | -2.38 |
| BTC Market Hours Daily | xgb | XGBoost | 460 | 181 | 279 | 39.35% | 37.08% | 39.35% | 10.65 pp | -98 | 41 | -2.39 |
| Consolidated Hourly | nn | NN | 62 | 22 | 40 | 35.48% | 35.48% | 35.48% | 14.52 pp | -18 | 7 | -2.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 62 | 22 | 40 | 35.48% | 35.48% | 35.48% | 14.52 pp | -18 | 7 | -2.57 |
| BTC Hourly | xgb | XGBoost | 812 | 345 | 467 | 42.49% | 40.00% | 42.92% | 7.51 pp | -122 | 44 | -2.77 |
| BTC Daily | xgb | XGBoost | 645 | 252 | 393 | 39.07% | 30.83% | 38.96% | 10.93 pp | -141 | 39 | -3.62 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 812 | 384 | 428 | 47.29% | 45.83% | 46.46% | 2.71 pp | -44 | 44 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 812 | 382 | 430 | 47.04% | 43.75% | 47.08% | 2.96 pp | -48 | 44 | -1.09 |
| BTC Hourly | nn | NN | 812 | 366 | 446 | 45.07% | 41.25% | 45.00% | 4.93 pp | -80 | 44 | -1.82 |
| BTC Hourly | rf | RandomForest | 812 | 364 | 448 | 44.83% | 44.58% | 44.79% | 5.17 pp | -84 | 44 | -1.91 |
| BTC Hourly | lstm | LSTM | 812 | 355 | 457 | 43.72% | 42.50% | 44.58% | 6.28 pp | -102 | 44 | -2.32 |
| BTC Hourly | xgb | XGBoost | 812 | 345 | 467 | 42.49% | 40.00% | 42.92% | 7.51 pp | -122 | 44 | -2.77 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 635 | 311 | 324 | 48.98% | 47.92% | 49.79% | 1.02 pp | -13 | 39 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 635 | 309 | 326 | 48.66% | 45.83% | 50.00% | 1.34 pp | -17 | 39 | -0.44 |
| BTC Daily | nn | NN | 635 | 298 | 337 | 46.93% | 42.50% | 48.96% | 3.07 pp | -39 | 39 | -1.00 |
| BTC Daily | lstm | LSTM | 635 | 281 | 354 | 44.25% | 42.50% | 43.54% | 5.75 pp | -73 | 39 | -1.87 |
| BTC Daily | rf | RandomForest | 635 | 271 | 364 | 42.68% | 42.08% | 43.54% | 7.32 pp | -93 | 39 | -2.38 |
| BTC Daily | xgb | XGBoost | 645 | 252 | 393 | 39.07% | 30.83% | 38.96% | 10.93 pp | -141 | 39 | -3.62 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 407 | 202 | 205 | 49.63% | 49.17% | 49.63% | 0.37 pp | -3 | 41 | -0.07 |
| BTC Market Hours | nn | NN | 407 | 193 | 214 | 47.42% | 51.25% | 47.42% | 2.58 pp | -21 | 41 | -0.51 |
| BTC Market Hours | transformer | Transformer | 407 | 188 | 219 | 46.19% | 42.50% | 46.19% | 3.81 pp | -31 | 41 | -0.76 |
| BTC Market Hours | lstm | LSTM | 407 | 180 | 227 | 44.23% | 45.42% | 44.23% | 5.77 pp | -47 | 41 | -1.15 |
| BTC Market Hours | rf | RandomForest | 407 | 175 | 232 | 43.00% | 42.50% | 43.00% | 7.00 pp | -57 | 41 | -1.39 |
| BTC Market Hours | xgb | XGBoost | 407 | 164 | 243 | 40.29% | 38.75% | 40.29% | 9.71 pp | -79 | 41 | -1.93 |

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
| Consolidated Hourly | rf | RandomForest | 62 | 36 | 26 | 58.06% | 58.06% | 58.06% | 8.06 pp | 10 | 7 | 1.43 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 62 | 32 | 30 | 51.61% | 51.61% | 51.61% | 1.61 pp | 2 | 7 | 0.29 |
| Consolidated Hourly | lstm | LSTM | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 7 | -0.29 |
| Consolidated Hourly | xgb | XGBoost | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 7 | -1.14 |
| Consolidated Hourly | transformer | Transformer | 62 | 26 | 36 | 41.94% | 41.94% | 41.94% | 8.06 pp | -10 | 7 | -1.43 |
| Consolidated Hourly | nn | NN | 62 | 22 | 40 | 35.48% | 35.48% | 35.48% | 14.52 pp | -18 | 7 | -2.57 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 62 | 36 | 26 | 58.06% | 58.06% | 58.06% | 8.06 pp | 10 | 7 | 1.43 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 62 | 32 | 30 | 51.61% | 51.61% | 51.61% | 1.61 pp | 2 | 7 | 0.29 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 62 | 30 | 32 | 48.39% | 48.39% | 48.39% | 1.61 pp | -2 | 7 | -0.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 62 | 27 | 35 | 43.55% | 43.55% | 43.55% | 6.45 pp | -8 | 7 | -1.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 62 | 26 | 36 | 41.94% | 41.94% | 41.94% | 8.06 pp | -10 | 7 | -1.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 62 | 22 | 40 | 35.48% | 35.48% | 35.48% | 14.52 pp | -18 | 7 | -2.57 |

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
