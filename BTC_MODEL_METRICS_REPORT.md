# BTC Model Metrics Report - All Rows

Generated at: 2026-08-21T23:17:03.963595+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-21 22:00:00+00:00 | 1079 | 791 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-21 22:00:00+00:00 | 889 | 565 | 323 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-21 22:00:00+00:00 | 461 | 327 | 133 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-21 22:00:00+00:00 | 463 | 381 | 80 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 07:00:00+00:00 | 1 | 1 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 07:00:00+00:00 | 1 | 1 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 07:00:00+00:00 | 1 | 0 | 1 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-18 07:00:00+00:00 | 1 | 0 | 1 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 327 | 163 | 164 | 49.85% | 48.75% | 49.85% | 0.15 pp | -1 | 35 | -0.03 |
| BTC Daily | transformer | Transformer | 555 | 273 | 282 | 49.19% | 52.50% | 49.38% | 0.81 pp | -9 | 36 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 555 | 272 | 283 | 49.01% | 49.17% | 48.96% | 0.99 pp | -11 | 36 | -0.31 |
| BTC Market Hours | transformer | Transformer | 327 | 154 | 173 | 47.09% | 47.08% | 47.09% | 2.91 pp | -19 | 35 | -0.54 |
| BTC Market Hours Daily | nn | NN | 381 | 176 | 205 | 46.19% | 48.33% | 46.19% | 3.81 pp | -29 | 34 | -0.85 |
| BTC Daily | nn | NN | 555 | 261 | 294 | 47.03% | 45.00% | 47.08% | 2.97 pp | -33 | 36 | -0.92 |
| BTC Market Hours | nn | NN | 327 | 146 | 181 | 44.65% | 46.67% | 44.65% | 5.35 pp | -35 | 35 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 381 | 173 | 208 | 45.41% | 46.67% | 45.41% | 4.59 pp | -35 | 34 | -1.03 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 381 | 172 | 209 | 45.14% | 45.83% | 45.14% | 4.86 pp | -37 | 34 | -1.09 |
| BTC Market Hours | lstm | LSTM | 327 | 144 | 183 | 44.04% | 45.00% | 44.04% | 5.96 pp | -39 | 35 | -1.11 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 757 | 355 | 402 | 46.90% | 43.33% | 47.50% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | transformer | Transformer | 757 | 355 | 402 | 46.90% | 44.17% | 45.62% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Daily | lstm | LSTM | 555 | 252 | 303 | 45.41% | 45.83% | 45.21% | 4.59 pp | -51 | 36 | -1.42 |
| BTC Market Hours | rf | RandomForest | 327 | 135 | 192 | 41.28% | 41.67% | 41.28% | 8.72 pp | -57 | 35 | -1.63 |
| BTC Market Hours | xgb | XGBoost | 327 | 132 | 195 | 40.37% | 40.42% | 40.37% | 9.63 pp | -63 | 35 | -1.80 |
| BTC Hourly | rf | RandomForest | 757 | 340 | 417 | 44.91% | 45.00% | 44.79% | 5.09 pp | -77 | 42 | -1.83 |
| BTC Hourly | nn | NN | 757 | 338 | 419 | 44.65% | 41.25% | 45.00% | 5.35 pp | -81 | 42 | -1.93 |
| BTC Daily | rf | RandomForest | 555 | 241 | 314 | 43.42% | 45.00% | 43.96% | 6.58 pp | -73 | 36 | -2.03 |
| BTC Market Hours Daily | lstm | LSTM | 381 | 155 | 226 | 40.68% | 39.17% | 40.68% | 9.32 pp | -71 | 34 | -2.09 |
| BTC Market Hours Daily | rf | RandomForest | 381 | 154 | 227 | 40.42% | 37.50% | 40.42% | 9.58 pp | -73 | 34 | -2.15 |
| BTC Hourly | lstm | LSTM | 757 | 333 | 424 | 43.99% | 42.92% | 45.42% | 6.01 pp | -91 | 42 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 381 | 150 | 231 | 39.37% | 37.50% | 39.37% | 10.63 pp | -81 | 34 | -2.38 |
| BTC Hourly | xgb | XGBoost | 757 | 326 | 431 | 43.06% | 42.08% | 44.38% | 6.94 pp | -105 | 42 | -2.50 |
| BTC Daily | xgb | XGBoost | 565 | 229 | 336 | 40.53% | 36.25% | 41.04% | 9.47 pp | -107 | 36 | -2.97 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 757 | 355 | 402 | 46.90% | 43.33% | 47.50% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | transformer | Transformer | 757 | 355 | 402 | 46.90% | 44.17% | 45.62% | 3.10 pp | -47 | 42 | -1.12 |
| BTC Hourly | rf | RandomForest | 757 | 340 | 417 | 44.91% | 45.00% | 44.79% | 5.09 pp | -77 | 42 | -1.83 |
| BTC Hourly | nn | NN | 757 | 338 | 419 | 44.65% | 41.25% | 45.00% | 5.35 pp | -81 | 42 | -1.93 |
| BTC Hourly | lstm | LSTM | 757 | 333 | 424 | 43.99% | 42.92% | 45.42% | 6.01 pp | -91 | 42 | -2.17 |
| BTC Hourly | xgb | XGBoost | 757 | 326 | 431 | 43.06% | 42.08% | 44.38% | 6.94 pp | -105 | 42 | -2.50 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 555 | 273 | 282 | 49.19% | 52.50% | 49.38% | 0.81 pp | -9 | 36 | -0.25 |
| BTC Daily | mlp_sklearn | MLPClassifier | 555 | 272 | 283 | 49.01% | 49.17% | 48.96% | 0.99 pp | -11 | 36 | -0.31 |
| BTC Daily | nn | NN | 555 | 261 | 294 | 47.03% | 45.00% | 47.08% | 2.97 pp | -33 | 36 | -0.92 |
| BTC Daily | lstm | LSTM | 555 | 252 | 303 | 45.41% | 45.83% | 45.21% | 4.59 pp | -51 | 36 | -1.42 |
| BTC Daily | rf | RandomForest | 555 | 241 | 314 | 43.42% | 45.00% | 43.96% | 6.58 pp | -73 | 36 | -2.03 |
| BTC Daily | xgb | XGBoost | 565 | 229 | 336 | 40.53% | 36.25% | 41.04% | 9.47 pp | -107 | 36 | -2.97 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 327 | 163 | 164 | 49.85% | 48.75% | 49.85% | 0.15 pp | -1 | 35 | -0.03 |
| BTC Market Hours | transformer | Transformer | 327 | 154 | 173 | 47.09% | 47.08% | 47.09% | 2.91 pp | -19 | 35 | -0.54 |
| BTC Market Hours | nn | NN | 327 | 146 | 181 | 44.65% | 46.67% | 44.65% | 5.35 pp | -35 | 35 | -1.00 |
| BTC Market Hours | lstm | LSTM | 327 | 144 | 183 | 44.04% | 45.00% | 44.04% | 5.96 pp | -39 | 35 | -1.11 |
| BTC Market Hours | rf | RandomForest | 327 | 135 | 192 | 41.28% | 41.67% | 41.28% | 8.72 pp | -57 | 35 | -1.63 |
| BTC Market Hours | xgb | XGBoost | 327 | 132 | 195 | 40.37% | 40.42% | 40.37% | 9.63 pp | -63 | 35 | -1.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 381 | 176 | 205 | 46.19% | 48.33% | 46.19% | 3.81 pp | -29 | 34 | -0.85 |
| BTC Market Hours Daily | transformer | Transformer | 381 | 173 | 208 | 45.41% | 46.67% | 45.41% | 4.59 pp | -35 | 34 | -1.03 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 381 | 172 | 209 | 45.14% | 45.83% | 45.14% | 4.86 pp | -37 | 34 | -1.09 |
| BTC Market Hours Daily | lstm | LSTM | 381 | 155 | 226 | 40.68% | 39.17% | 40.68% | 9.32 pp | -71 | 34 | -2.09 |
| BTC Market Hours Daily | rf | RandomForest | 381 | 154 | 227 | 40.42% | 37.50% | 40.42% | 9.58 pp | -73 | 34 | -2.15 |
| BTC Market Hours Daily | xgb | XGBoost | 381 | 150 | 231 | 39.37% | 37.50% | 39.37% | 10.63 pp | -81 | 34 | -2.38 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Hourly | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

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
