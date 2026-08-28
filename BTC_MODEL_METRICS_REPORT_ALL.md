# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T19:16:42.762609+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 825 | 294 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 989 | 624 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 18:00:00+00:00 | 573 | 386 | 186 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 18:00:00+00:00 | 575 | 440 | 133 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 07:00:00+00:00 | 45 | 45 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 07:00:00+00:00 | 45 | 45 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 07:00:00+00:00 | 45 | 0 | 45 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 07:00:00+00:00 | 45 | 0 | 45 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 45 | 25 | 20 | 55.56% | 55.56% | 55.56% | 5.56 pp | 5 | 5 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 45 | 25 | 20 | 55.56% | 55.56% | 55.56% | 5.56 pp | 5 | 5 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 45 | 23 | 22 | 51.11% | 51.11% | 51.11% | 1.11 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 45 | 23 | 22 | 51.11% | 51.11% | 51.11% | 1.11 pp | 1 | 5 | 0.20 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 386 | 190 | 196 | 49.22% | 47.08% | 49.22% | 0.78 pp | -6 | 39 | -0.15 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 45 | 22 | 23 | 48.89% | 48.89% | 48.89% | 1.11 pp | -1 | 5 | -0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 45 | 22 | 23 | 48.89% | 48.89% | 48.89% | 1.11 pp | -1 | 5 | -0.20 |
| BTC Daily | transformer | Transformer | 614 | 303 | 311 | 49.35% | 50.00% | 50.21% | 0.65 pp | -8 | 38 | -0.21 |
| BTC Daily | mlp_sklearn | MLPClassifier | 614 | 301 | 313 | 49.02% | 47.50% | 50.42% | 0.98 pp | -12 | 38 | -0.32 |
| BTC Market Hours | nn | NN | 386 | 179 | 207 | 46.37% | 48.75% | 46.37% | 3.63 pp | -28 | 39 | -0.72 |
| BTC Market Hours | transformer | Transformer | 386 | 179 | 207 | 46.37% | 43.75% | 46.37% | 3.63 pp | -28 | 39 | -0.72 |
| BTC Market Hours Daily | transformer | Transformer | 440 | 203 | 237 | 46.14% | 48.33% | 46.14% | 3.86 pp | -34 | 39 | -0.87 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 440 | 202 | 238 | 45.91% | 46.67% | 45.91% | 4.09 pp | -36 | 39 | -0.92 |
| Consolidated Hourly | transformer | Transformer | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 5 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 791 | 373 | 418 | 47.16% | 44.58% | 46.88% | 2.84 pp | -45 | 43 | -1.05 |
| BTC Hourly | transformer | Transformer | 791 | 373 | 418 | 47.16% | 44.17% | 46.46% | 2.84 pp | -45 | 43 | -1.05 |
| BTC Daily | nn | NN | 614 | 287 | 327 | 46.74% | 42.92% | 48.54% | 3.26 pp | -40 | 38 | -1.05 |
| BTC Market Hours Daily | nn | NN | 440 | 199 | 241 | 45.23% | 46.25% | 45.23% | 4.77 pp | -42 | 39 | -1.08 |
| BTC Market Hours | lstm | LSTM | 386 | 166 | 220 | 43.01% | 43.75% | 43.01% | 6.99 pp | -54 | 39 | -1.38 |
| BTC Market Hours | rf | RandomForest | 386 | 165 | 221 | 42.75% | 40.83% | 42.75% | 7.25 pp | -56 | 39 | -1.44 |
| BTC Daily | lstm | LSTM | 614 | 275 | 339 | 44.79% | 43.75% | 44.58% | 5.21 pp | -64 | 38 | -1.68 |
| BTC Hourly | nn | NN | 791 | 357 | 434 | 45.13% | 40.42% | 45.62% | 4.87 pp | -77 | 43 | -1.79 |
| BTC Market Hours | xgb | XGBoost | 386 | 156 | 230 | 40.41% | 38.75% | 40.41% | 9.59 pp | -74 | 39 | -1.90 |
| BTC Market Hours Daily | rf | RandomForest | 440 | 180 | 260 | 40.91% | 40.00% | 40.91% | 9.09 pp | -80 | 39 | -2.05 |
| BTC Hourly | rf | RandomForest | 791 | 350 | 441 | 44.25% | 42.08% | 43.75% | 5.75 pp | -91 | 43 | -2.12 |
| BTC Hourly | lstm | LSTM | 791 | 348 | 443 | 43.99% | 43.75% | 45.42% | 6.01 pp | -95 | 43 | -2.21 |
| BTC Market Hours Daily | lstm | LSTM | 440 | 175 | 265 | 39.77% | 37.50% | 39.77% | 10.23 pp | -90 | 39 | -2.31 |
| BTC Daily | rf | RandomForest | 614 | 263 | 351 | 42.83% | 42.50% | 43.75% | 7.17 pp | -88 | 38 | -2.32 |
| BTC Market Hours Daily | xgb | XGBoost | 440 | 173 | 267 | 39.32% | 37.50% | 39.32% | 10.68 pp | -94 | 39 | -2.41 |
| BTC Hourly | xgb | XGBoost | 791 | 336 | 455 | 42.48% | 38.75% | 43.96% | 7.52 pp | -119 | 43 | -2.77 |
| Consolidated Hourly | nn | NN | 45 | 15 | 30 | 33.33% | 33.33% | 33.33% | 16.67 pp | -15 | 5 | -3.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 45 | 15 | 30 | 33.33% | 33.33% | 33.33% | 16.67 pp | -15 | 5 | -3.00 |
| BTC Daily | xgb | XGBoost | 624 | 248 | 376 | 39.74% | 33.33% | 40.00% | 10.26 pp | -128 | 38 | -3.37 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 791 | 373 | 418 | 47.16% | 44.58% | 46.88% | 2.84 pp | -45 | 43 | -1.05 |
| BTC Hourly | transformer | Transformer | 791 | 373 | 418 | 47.16% | 44.17% | 46.46% | 2.84 pp | -45 | 43 | -1.05 |
| BTC Hourly | nn | NN | 791 | 357 | 434 | 45.13% | 40.42% | 45.62% | 4.87 pp | -77 | 43 | -1.79 |
| BTC Hourly | rf | RandomForest | 791 | 350 | 441 | 44.25% | 42.08% | 43.75% | 5.75 pp | -91 | 43 | -2.12 |
| BTC Hourly | lstm | LSTM | 791 | 348 | 443 | 43.99% | 43.75% | 45.42% | 6.01 pp | -95 | 43 | -2.21 |
| BTC Hourly | xgb | XGBoost | 791 | 336 | 455 | 42.48% | 38.75% | 43.96% | 7.52 pp | -119 | 43 | -2.77 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 614 | 303 | 311 | 49.35% | 50.00% | 50.21% | 0.65 pp | -8 | 38 | -0.21 |
| BTC Daily | mlp_sklearn | MLPClassifier | 614 | 301 | 313 | 49.02% | 47.50% | 50.42% | 0.98 pp | -12 | 38 | -0.32 |
| BTC Daily | nn | NN | 614 | 287 | 327 | 46.74% | 42.92% | 48.54% | 3.26 pp | -40 | 38 | -1.05 |
| BTC Daily | lstm | LSTM | 614 | 275 | 339 | 44.79% | 43.75% | 44.58% | 5.21 pp | -64 | 38 | -1.68 |
| BTC Daily | rf | RandomForest | 614 | 263 | 351 | 42.83% | 42.50% | 43.75% | 7.17 pp | -88 | 38 | -2.32 |
| BTC Daily | xgb | XGBoost | 624 | 248 | 376 | 39.74% | 33.33% | 40.00% | 10.26 pp | -128 | 38 | -3.37 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 386 | 190 | 196 | 49.22% | 47.08% | 49.22% | 0.78 pp | -6 | 39 | -0.15 |
| BTC Market Hours | nn | NN | 386 | 179 | 207 | 46.37% | 48.75% | 46.37% | 3.63 pp | -28 | 39 | -0.72 |
| BTC Market Hours | transformer | Transformer | 386 | 179 | 207 | 46.37% | 43.75% | 46.37% | 3.63 pp | -28 | 39 | -0.72 |
| BTC Market Hours | lstm | LSTM | 386 | 166 | 220 | 43.01% | 43.75% | 43.01% | 6.99 pp | -54 | 39 | -1.38 |
| BTC Market Hours | rf | RandomForest | 386 | 165 | 221 | 42.75% | 40.83% | 42.75% | 7.25 pp | -56 | 39 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 386 | 156 | 230 | 40.41% | 38.75% | 40.41% | 9.59 pp | -74 | 39 | -1.90 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 440 | 203 | 237 | 46.14% | 48.33% | 46.14% | 3.86 pp | -34 | 39 | -0.87 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 440 | 202 | 238 | 45.91% | 46.67% | 45.91% | 4.09 pp | -36 | 39 | -0.92 |
| BTC Market Hours Daily | nn | NN | 440 | 199 | 241 | 45.23% | 46.25% | 45.23% | 4.77 pp | -42 | 39 | -1.08 |
| BTC Market Hours Daily | rf | RandomForest | 440 | 180 | 260 | 40.91% | 40.00% | 40.91% | 9.09 pp | -80 | 39 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 440 | 175 | 265 | 39.77% | 37.50% | 39.77% | 10.23 pp | -90 | 39 | -2.31 |
| BTC Market Hours Daily | xgb | XGBoost | 440 | 173 | 267 | 39.32% | 37.50% | 39.32% | 10.68 pp | -94 | 39 | -2.41 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 45 | 25 | 20 | 55.56% | 55.56% | 55.56% | 5.56 pp | 5 | 5 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 45 | 23 | 22 | 51.11% | 51.11% | 51.11% | 1.11 pp | 1 | 5 | 0.20 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 45 | 22 | 23 | 48.89% | 48.89% | 48.89% | 1.11 pp | -1 | 5 | -0.20 |
| Consolidated Hourly | transformer | Transformer | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 5 | -1.00 |
| Consolidated Hourly | nn | NN | 45 | 15 | 30 | 33.33% | 33.33% | 33.33% | 16.67 pp | -15 | 5 | -3.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 45 | 25 | 20 | 55.56% | 55.56% | 55.56% | 5.56 pp | 5 | 5 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 45 | 23 | 22 | 51.11% | 51.11% | 51.11% | 1.11 pp | 1 | 5 | 0.20 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 45 | 22 | 23 | 48.89% | 48.89% | 48.89% | 1.11 pp | -1 | 5 | -0.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 5 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 45 | 15 | 30 | 33.33% | 33.33% | 33.33% | 16.67 pp | -15 | 5 | -3.00 |

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
