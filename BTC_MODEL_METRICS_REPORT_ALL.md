# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T00:35:55.391778+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 810 | 309 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 974 | 609 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 23:00:00+00:00 | 550 | 371 | 178 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 23:00:00+00:00 | 552 | 425 | 125 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 17:00:00+00:00 | 34 | 34 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 17:00:00+00:00 | 34 | 34 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 17:00:00+00:00 | 34 | 1 | 33 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 17:00:00+00:00 | 34 | 1 | 33 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 34 | 19 | 15 | 55.88% | 55.88% | 55.88% | 5.88 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 34 | 19 | 15 | 55.88% | 55.88% | 55.88% | 5.88 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 34 | 19 | 15 | 55.88% | 55.88% | 55.88% | 5.88 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 34 | 19 | 15 | 55.88% | 55.88% | 55.88% | 5.88 pp | 4 | 4 | 1.00 |
| BTC Daily | transformer | Transformer | 599 | 296 | 303 | 49.42% | 50.83% | 50.21% | 0.58 pp | -7 | 38 | -0.18 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 371 | 182 | 189 | 49.06% | 48.33% | 49.06% | 0.94 pp | -7 | 38 | -0.18 |
| BTC Daily | mlp_sklearn | MLPClassifier | 599 | 293 | 306 | 48.91% | 47.50% | 49.58% | 1.09 pp | -13 | 38 | -0.34 |
| BTC Market Hours | transformer | Transformer | 371 | 174 | 197 | 46.90% | 44.58% | 46.90% | 3.10 pp | -23 | 38 | -0.61 |
| BTC Market Hours | nn | NN | 371 | 170 | 201 | 45.82% | 48.33% | 45.82% | 4.18 pp | -31 | 38 | -0.82 |
| BTC Daily | nn | NN | 599 | 283 | 316 | 47.25% | 45.42% | 48.54% | 2.75 pp | -33 | 38 | -0.87 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 425 | 194 | 231 | 45.65% | 45.42% | 45.65% | 4.35 pp | -37 | 38 | -0.97 |
| BTC Market Hours Daily | nn | NN | 425 | 194 | 231 | 45.65% | 46.67% | 45.65% | 4.35 pp | -37 | 38 | -0.97 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 425 | 193 | 232 | 45.41% | 47.50% | 45.41% | 4.59 pp | -39 | 38 | -1.03 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 776 | 366 | 410 | 47.16% | 44.17% | 47.08% | 2.84 pp | -44 | 42 | -1.05 |
| BTC Hourly | transformer | Transformer | 776 | 364 | 412 | 46.91% | 43.33% | 46.04% | 3.09 pp | -48 | 42 | -1.14 |
| BTC Market Hours | lstm | LSTM | 371 | 158 | 213 | 42.59% | 42.92% | 42.59% | 7.41 pp | -55 | 38 | -1.45 |
| BTC Market Hours | rf | RandomForest | 371 | 157 | 214 | 42.32% | 40.42% | 42.32% | 7.68 pp | -57 | 38 | -1.50 |
| BTC Daily | lstm | LSTM | 599 | 268 | 331 | 44.74% | 43.75% | 45.00% | 5.26 pp | -63 | 38 | -1.66 |
| BTC Hourly | nn | NN | 776 | 348 | 428 | 44.85% | 40.00% | 45.83% | 5.15 pp | -80 | 42 | -1.90 |
| BTC Hourly | rf | RandomForest | 776 | 348 | 428 | 44.85% | 43.75% | 44.58% | 5.15 pp | -80 | 42 | -1.90 |
| BTC Market Hours | xgb | XGBoost | 371 | 149 | 222 | 40.16% | 40.42% | 40.16% | 9.84 pp | -73 | 38 | -1.92 |
| Consolidated Hourly | xgb | XGBoost | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 4 | -2.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 4 | -2.00 |
| BTC Daily | rf | RandomForest | 599 | 260 | 339 | 43.41% | 45.00% | 44.17% | 6.59 pp | -79 | 38 | -2.08 |
| BTC Hourly | lstm | LSTM | 776 | 344 | 432 | 44.33% | 43.75% | 46.25% | 5.67 pp | -88 | 42 | -2.10 |
| BTC Market Hours Daily | rf | RandomForest | 425 | 172 | 253 | 40.47% | 40.00% | 40.47% | 9.53 pp | -81 | 38 | -2.13 |
| BTC Market Hours Daily | xgb | XGBoost | 425 | 168 | 257 | 39.53% | 38.33% | 39.53% | 10.47 pp | -89 | 38 | -2.34 |
| BTC Market Hours Daily | lstm | LSTM | 425 | 167 | 258 | 39.29% | 37.92% | 39.29% | 10.71 pp | -91 | 38 | -2.39 |
| BTC Hourly | xgb | XGBoost | 776 | 333 | 443 | 42.91% | 40.83% | 44.38% | 7.09 pp | -110 | 42 | -2.62 |
| BTC Daily | xgb | XGBoost | 609 | 245 | 364 | 40.23% | 35.83% | 40.21% | 9.77 pp | -119 | 38 | -3.13 |
| Consolidated Hourly | nn | NN | 34 | 10 | 24 | 29.41% | 29.41% | 29.41% | 20.59 pp | -14 | 4 | -3.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 34 | 10 | 24 | 29.41% | 29.41% | 29.41% | 20.59 pp | -14 | 4 | -3.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 776 | 366 | 410 | 47.16% | 44.17% | 47.08% | 2.84 pp | -44 | 42 | -1.05 |
| BTC Hourly | transformer | Transformer | 776 | 364 | 412 | 46.91% | 43.33% | 46.04% | 3.09 pp | -48 | 42 | -1.14 |
| BTC Hourly | nn | NN | 776 | 348 | 428 | 44.85% | 40.00% | 45.83% | 5.15 pp | -80 | 42 | -1.90 |
| BTC Hourly | rf | RandomForest | 776 | 348 | 428 | 44.85% | 43.75% | 44.58% | 5.15 pp | -80 | 42 | -1.90 |
| BTC Hourly | lstm | LSTM | 776 | 344 | 432 | 44.33% | 43.75% | 46.25% | 5.67 pp | -88 | 42 | -2.10 |
| BTC Hourly | xgb | XGBoost | 776 | 333 | 443 | 42.91% | 40.83% | 44.38% | 7.09 pp | -110 | 42 | -2.62 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 599 | 296 | 303 | 49.42% | 50.83% | 50.21% | 0.58 pp | -7 | 38 | -0.18 |
| BTC Daily | mlp_sklearn | MLPClassifier | 599 | 293 | 306 | 48.91% | 47.50% | 49.58% | 1.09 pp | -13 | 38 | -0.34 |
| BTC Daily | nn | NN | 599 | 283 | 316 | 47.25% | 45.42% | 48.54% | 2.75 pp | -33 | 38 | -0.87 |
| BTC Daily | lstm | LSTM | 599 | 268 | 331 | 44.74% | 43.75% | 45.00% | 5.26 pp | -63 | 38 | -1.66 |
| BTC Daily | rf | RandomForest | 599 | 260 | 339 | 43.41% | 45.00% | 44.17% | 6.59 pp | -79 | 38 | -2.08 |
| BTC Daily | xgb | XGBoost | 609 | 245 | 364 | 40.23% | 35.83% | 40.21% | 9.77 pp | -119 | 38 | -3.13 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 371 | 182 | 189 | 49.06% | 48.33% | 49.06% | 0.94 pp | -7 | 38 | -0.18 |
| BTC Market Hours | transformer | Transformer | 371 | 174 | 197 | 46.90% | 44.58% | 46.90% | 3.10 pp | -23 | 38 | -0.61 |
| BTC Market Hours | nn | NN | 371 | 170 | 201 | 45.82% | 48.33% | 45.82% | 4.18 pp | -31 | 38 | -0.82 |
| BTC Market Hours | lstm | LSTM | 371 | 158 | 213 | 42.59% | 42.92% | 42.59% | 7.41 pp | -55 | 38 | -1.45 |
| BTC Market Hours | rf | RandomForest | 371 | 157 | 214 | 42.32% | 40.42% | 42.32% | 7.68 pp | -57 | 38 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 371 | 149 | 222 | 40.16% | 40.42% | 40.16% | 9.84 pp | -73 | 38 | -1.92 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 425 | 194 | 231 | 45.65% | 45.42% | 45.65% | 4.35 pp | -37 | 38 | -0.97 |
| BTC Market Hours Daily | nn | NN | 425 | 194 | 231 | 45.65% | 46.67% | 45.65% | 4.35 pp | -37 | 38 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 425 | 193 | 232 | 45.41% | 47.50% | 45.41% | 4.59 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | rf | RandomForest | 425 | 172 | 253 | 40.47% | 40.00% | 40.47% | 9.53 pp | -81 | 38 | -2.13 |
| BTC Market Hours Daily | xgb | XGBoost | 425 | 168 | 257 | 39.53% | 38.33% | 39.53% | 10.47 pp | -89 | 38 | -2.34 |
| BTC Market Hours Daily | lstm | LSTM | 425 | 167 | 258 | 39.29% | 37.92% | 39.29% | 10.71 pp | -91 | 38 | -2.39 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 34 | 19 | 15 | 55.88% | 55.88% | 55.88% | 5.88 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 34 | 19 | 15 | 55.88% | 55.88% | 55.88% | 5.88 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 4 | -2.00 |
| Consolidated Hourly | nn | NN | 34 | 10 | 24 | 29.41% | 29.41% | 29.41% | 20.59 pp | -14 | 4 | -3.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 34 | 19 | 15 | 55.88% | 55.88% | 55.88% | 5.88 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 34 | 19 | 15 | 55.88% | 55.88% | 55.88% | 5.88 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 4 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 34 | 10 | 24 | 29.41% | 29.41% | 29.41% | 20.59 pp | -14 | 4 | -3.50 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
