# BTC Model Metrics Report - All Rows

Generated at: 2026-08-29T01:03:13.096993+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 829 | 290 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 993 | 628 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 583 | 390 | 192 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-29 00:00:00+00:00 | 585 | 444 | 139 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 23:00:00+00:00 | 50 | 50 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 23:00:00+00:00 | 50 | 50 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 23:00:00+00:00 | 50 | 1 | 49 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 23:00:00+00:00 | 50 | 1 | 49 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 5 | 0.80 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 5 | 0.80 |
| Consolidated Hourly | lstm | LSTM | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 390 | 193 | 197 | 49.49% | 48.33% | 49.49% | 0.51 pp | -4 | 39 | -0.10 |
| BTC Daily | transformer | Transformer | 618 | 304 | 314 | 49.19% | 48.75% | 50.21% | 0.81 pp | -10 | 39 | -0.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 618 | 302 | 316 | 48.87% | 47.50% | 50.21% | 1.13 pp | -14 | 39 | -0.36 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 5 | -0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 5 | -0.40 |
| BTC Market Hours | transformer | Transformer | 390 | 183 | 207 | 46.92% | 44.58% | 46.92% | 3.08 pp | -24 | 39 | -0.62 |
| BTC Market Hours | nn | NN | 390 | 182 | 208 | 46.67% | 49.17% | 46.67% | 3.33 pp | -26 | 39 | -0.67 |
| BTC Market Hours Daily | transformer | Transformer | 444 | 206 | 238 | 46.40% | 48.33% | 46.40% | 3.60 pp | -32 | 39 | -0.82 |
| BTC Daily | nn | NN | 618 | 291 | 327 | 47.09% | 43.75% | 49.17% | 2.91 pp | -36 | 39 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 444 | 204 | 240 | 45.95% | 46.25% | 45.95% | 4.05 pp | -36 | 39 | -0.92 |
| BTC Hourly | transformer | Transformer | 795 | 376 | 419 | 47.30% | 45.00% | 46.67% | 2.70 pp | -43 | 43 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Market Hours Daily | nn | NN | 444 | 201 | 243 | 45.27% | 46.25% | 45.27% | 4.73 pp | -42 | 39 | -1.08 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 795 | 374 | 421 | 47.04% | 44.58% | 46.67% | 2.96 pp | -47 | 43 | -1.09 |
| Consolidated Hourly | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 5 | -1.20 |
| BTC Market Hours | lstm | LSTM | 390 | 169 | 221 | 43.33% | 43.33% | 43.33% | 6.67 pp | -52 | 39 | -1.33 |
| BTC Market Hours | rf | RandomForest | 390 | 167 | 223 | 42.82% | 40.83% | 42.82% | 7.18 pp | -56 | 39 | -1.44 |
| BTC Daily | lstm | LSTM | 618 | 276 | 342 | 44.66% | 43.33% | 44.58% | 5.34 pp | -66 | 39 | -1.69 |
| BTC Hourly | nn | NN | 795 | 359 | 436 | 45.16% | 41.25% | 45.42% | 4.84 pp | -77 | 43 | -1.79 |
| BTC Market Hours | xgb | XGBoost | 390 | 160 | 230 | 41.03% | 39.17% | 41.03% | 8.97 pp | -70 | 39 | -1.79 |
| Consolidated Hourly | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 5 | -2.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 5 | -2.00 |
| BTC Hourly | rf | RandomForest | 795 | 354 | 441 | 44.53% | 42.92% | 43.96% | 5.47 pp | -87 | 43 | -2.02 |
| BTC Market Hours Daily | rf | RandomForest | 444 | 181 | 263 | 40.77% | 40.00% | 40.77% | 9.23 pp | -82 | 39 | -2.10 |
| BTC Hourly | lstm | LSTM | 795 | 351 | 444 | 44.15% | 44.58% | 45.62% | 5.85 pp | -93 | 43 | -2.16 |
| BTC Daily | rf | RandomForest | 618 | 265 | 353 | 42.88% | 42.50% | 43.75% | 7.12 pp | -88 | 39 | -2.26 |
| BTC Market Hours Daily | lstm | LSTM | 444 | 177 | 267 | 39.86% | 37.92% | 39.86% | 10.14 pp | -90 | 39 | -2.31 |
| BTC Market Hours Daily | xgb | XGBoost | 444 | 175 | 269 | 39.41% | 37.92% | 39.41% | 10.59 pp | -94 | 39 | -2.41 |
| BTC Hourly | xgb | XGBoost | 795 | 339 | 456 | 42.64% | 40.00% | 44.17% | 7.36 pp | -117 | 43 | -2.72 |
| Consolidated Hourly | nn | NN | 50 | 17 | 33 | 34.00% | 34.00% | 34.00% | 16.00 pp | -16 | 5 | -3.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 50 | 17 | 33 | 34.00% | 34.00% | 34.00% | 16.00 pp | -16 | 5 | -3.20 |
| BTC Daily | xgb | XGBoost | 628 | 250 | 378 | 39.81% | 33.33% | 40.21% | 10.19 pp | -128 | 39 | -3.28 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 795 | 376 | 419 | 47.30% | 45.00% | 46.67% | 2.70 pp | -43 | 43 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 795 | 374 | 421 | 47.04% | 44.58% | 46.67% | 2.96 pp | -47 | 43 | -1.09 |
| BTC Hourly | nn | NN | 795 | 359 | 436 | 45.16% | 41.25% | 45.42% | 4.84 pp | -77 | 43 | -1.79 |
| BTC Hourly | rf | RandomForest | 795 | 354 | 441 | 44.53% | 42.92% | 43.96% | 5.47 pp | -87 | 43 | -2.02 |
| BTC Hourly | lstm | LSTM | 795 | 351 | 444 | 44.15% | 44.58% | 45.62% | 5.85 pp | -93 | 43 | -2.16 |
| BTC Hourly | xgb | XGBoost | 795 | 339 | 456 | 42.64% | 40.00% | 44.17% | 7.36 pp | -117 | 43 | -2.72 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 618 | 304 | 314 | 49.19% | 48.75% | 50.21% | 0.81 pp | -10 | 39 | -0.26 |
| BTC Daily | mlp_sklearn | MLPClassifier | 618 | 302 | 316 | 48.87% | 47.50% | 50.21% | 1.13 pp | -14 | 39 | -0.36 |
| BTC Daily | nn | NN | 618 | 291 | 327 | 47.09% | 43.75% | 49.17% | 2.91 pp | -36 | 39 | -0.92 |
| BTC Daily | lstm | LSTM | 618 | 276 | 342 | 44.66% | 43.33% | 44.58% | 5.34 pp | -66 | 39 | -1.69 |
| BTC Daily | rf | RandomForest | 618 | 265 | 353 | 42.88% | 42.50% | 43.75% | 7.12 pp | -88 | 39 | -2.26 |
| BTC Daily | xgb | XGBoost | 628 | 250 | 378 | 39.81% | 33.33% | 40.21% | 10.19 pp | -128 | 39 | -3.28 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 390 | 193 | 197 | 49.49% | 48.33% | 49.49% | 0.51 pp | -4 | 39 | -0.10 |
| BTC Market Hours | transformer | Transformer | 390 | 183 | 207 | 46.92% | 44.58% | 46.92% | 3.08 pp | -24 | 39 | -0.62 |
| BTC Market Hours | nn | NN | 390 | 182 | 208 | 46.67% | 49.17% | 46.67% | 3.33 pp | -26 | 39 | -0.67 |
| BTC Market Hours | lstm | LSTM | 390 | 169 | 221 | 43.33% | 43.33% | 43.33% | 6.67 pp | -52 | 39 | -1.33 |
| BTC Market Hours | rf | RandomForest | 390 | 167 | 223 | 42.82% | 40.83% | 42.82% | 7.18 pp | -56 | 39 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 390 | 160 | 230 | 41.03% | 39.17% | 41.03% | 8.97 pp | -70 | 39 | -1.79 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 444 | 206 | 238 | 46.40% | 48.33% | 46.40% | 3.60 pp | -32 | 39 | -0.82 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 444 | 204 | 240 | 45.95% | 46.25% | 45.95% | 4.05 pp | -36 | 39 | -0.92 |
| BTC Market Hours Daily | nn | NN | 444 | 201 | 243 | 45.27% | 46.25% | 45.27% | 4.73 pp | -42 | 39 | -1.08 |
| BTC Market Hours Daily | rf | RandomForest | 444 | 181 | 263 | 40.77% | 40.00% | 40.77% | 9.23 pp | -82 | 39 | -2.10 |
| BTC Market Hours Daily | lstm | LSTM | 444 | 177 | 267 | 39.86% | 37.92% | 39.86% | 10.14 pp | -90 | 39 | -2.31 |
| BTC Market Hours Daily | xgb | XGBoost | 444 | 175 | 269 | 39.41% | 37.92% | 39.41% | 10.59 pp | -94 | 39 | -2.41 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 5 | 0.80 |
| Consolidated Hourly | lstm | LSTM | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 5 | -0.40 |
| Consolidated Hourly | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 5 | -2.00 |
| Consolidated Hourly | nn | NN | 50 | 17 | 33 | 34.00% | 34.00% | 34.00% | 16.00 pp | -16 | 5 | -3.20 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 50 | 27 | 23 | 54.00% | 54.00% | 54.00% | 4.00 pp | 4 | 5 | 0.80 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 50 | 25 | 25 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 50 | 24 | 26 | 48.00% | 48.00% | 48.00% | 2.00 pp | -2 | 5 | -0.40 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 50 | 22 | 28 | 44.00% | 44.00% | 44.00% | 6.00 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 50 | 20 | 30 | 40.00% | 40.00% | 40.00% | 10.00 pp | -10 | 5 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 50 | 17 | 33 | 34.00% | 34.00% | 34.00% | 16.00 pp | -16 | 5 | -3.20 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
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
