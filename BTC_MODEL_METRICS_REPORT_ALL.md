# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T12:28:12.528230+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 820 | 299 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 983 | 618 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 560 | 380 | 179 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 562 | 434 | 126 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 02:00:00+00:00 | 40 | 40 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 02:00:00+00:00 | 40 | 40 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 02:00:00+00:00 | 40 | 0 | 40 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-22 02:00:00+00:00 | 40 | 0 | 40 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 40 | 23 | 17 | 57.50% | 57.50% | 57.50% | 7.50 pp | 6 | 5 | 1.20 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 40 | 23 | 17 | 57.50% | 57.50% | 57.50% | 7.50 pp | 6 | 5 | 1.20 |
| Consolidated Hourly | lstm | LSTM | 40 | 21 | 19 | 52.50% | 52.50% | 52.50% | 2.50 pp | 2 | 5 | 0.40 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 40 | 21 | 19 | 52.50% | 52.50% | 52.50% | 2.50 pp | 2 | 5 | 0.40 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 380 | 186 | 194 | 48.95% | 47.50% | 48.95% | 1.05 pp | -8 | 39 | -0.21 |
| BTC Daily | transformer | Transformer | 608 | 298 | 310 | 49.01% | 49.58% | 49.79% | 0.99 pp | -12 | 38 | -0.32 |
| BTC Daily | mlp_sklearn | MLPClassifier | 608 | 297 | 311 | 48.85% | 47.50% | 49.79% | 1.15 pp | -14 | 38 | -0.37 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 5 | -0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 5 | -0.40 |
| BTC Market Hours | transformer | Transformer | 380 | 178 | 202 | 46.84% | 44.17% | 46.84% | 3.16 pp | -24 | 39 | -0.62 |
| BTC Market Hours | nn | NN | 380 | 176 | 204 | 46.32% | 49.17% | 46.32% | 3.68 pp | -28 | 39 | -0.72 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 434 | 198 | 236 | 45.62% | 45.83% | 45.62% | 4.38 pp | -38 | 39 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 434 | 198 | 236 | 45.62% | 47.08% | 45.62% | 4.38 pp | -38 | 39 | -0.97 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 786 | 372 | 414 | 47.33% | 45.00% | 47.50% | 2.67 pp | -42 | 43 | -0.98 |
| BTC Daily | nn | NN | 608 | 285 | 323 | 46.88% | 44.17% | 48.12% | 3.12 pp | -38 | 38 | -1.00 |
| BTC Market Hours Daily | nn | NN | 434 | 197 | 237 | 45.39% | 46.25% | 45.39% | 4.61 pp | -40 | 39 | -1.03 |
| BTC Hourly | transformer | Transformer | 786 | 368 | 418 | 46.82% | 42.92% | 45.83% | 3.18 pp | -50 | 43 | -1.16 |
| Consolidated Hourly | transformer | Transformer | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| BTC Market Hours | lstm | LSTM | 380 | 163 | 217 | 42.89% | 44.17% | 42.89% | 7.11 pp | -54 | 39 | -1.38 |
| BTC Market Hours | rf | RandomForest | 380 | 160 | 220 | 42.11% | 40.42% | 42.11% | 7.89 pp | -60 | 39 | -1.54 |
| BTC Daily | lstm | LSTM | 608 | 272 | 336 | 44.74% | 44.17% | 44.58% | 5.26 pp | -64 | 38 | -1.68 |
| BTC Hourly | nn | NN | 786 | 353 | 433 | 44.91% | 40.42% | 45.83% | 5.09 pp | -80 | 43 | -1.86 |
| BTC Hourly | rf | RandomForest | 786 | 350 | 436 | 44.53% | 43.33% | 43.96% | 5.47 pp | -86 | 43 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 380 | 151 | 229 | 39.74% | 38.33% | 39.74% | 10.26 pp | -78 | 39 | -2.00 |
| BTC Hourly | lstm | LSTM | 786 | 347 | 439 | 44.15% | 43.75% | 45.42% | 5.85 pp | -92 | 43 | -2.14 |
| BTC Market Hours Daily | rf | RandomForest | 434 | 175 | 259 | 40.32% | 39.17% | 40.32% | 9.68 pp | -84 | 39 | -2.15 |
| BTC Daily | rf | RandomForest | 608 | 261 | 347 | 42.93% | 43.33% | 43.33% | 7.07 pp | -86 | 38 | -2.26 |
| BTC Market Hours Daily | lstm | LSTM | 434 | 171 | 263 | 39.40% | 37.92% | 39.40% | 10.60 pp | -92 | 39 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 434 | 170 | 264 | 39.17% | 38.33% | 39.17% | 10.83 pp | -94 | 39 | -2.41 |
| BTC Hourly | xgb | XGBoost | 786 | 336 | 450 | 42.75% | 40.00% | 44.17% | 7.25 pp | -114 | 43 | -2.65 |
| Consolidated Hourly | nn | NN | 40 | 13 | 27 | 32.50% | 32.50% | 32.50% | 17.50 pp | -14 | 5 | -2.80 |
| Consolidated Daily/Hourly Refresh | nn | NN | 40 | 13 | 27 | 32.50% | 32.50% | 32.50% | 17.50 pp | -14 | 5 | -2.80 |
| BTC Daily | xgb | XGBoost | 618 | 246 | 372 | 39.81% | 33.33% | 40.00% | 10.19 pp | -126 | 38 | -3.32 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 786 | 372 | 414 | 47.33% | 45.00% | 47.50% | 2.67 pp | -42 | 43 | -0.98 |
| BTC Hourly | transformer | Transformer | 786 | 368 | 418 | 46.82% | 42.92% | 45.83% | 3.18 pp | -50 | 43 | -1.16 |
| BTC Hourly | nn | NN | 786 | 353 | 433 | 44.91% | 40.42% | 45.83% | 5.09 pp | -80 | 43 | -1.86 |
| BTC Hourly | rf | RandomForest | 786 | 350 | 436 | 44.53% | 43.33% | 43.96% | 5.47 pp | -86 | 43 | -2.00 |
| BTC Hourly | lstm | LSTM | 786 | 347 | 439 | 44.15% | 43.75% | 45.42% | 5.85 pp | -92 | 43 | -2.14 |
| BTC Hourly | xgb | XGBoost | 786 | 336 | 450 | 42.75% | 40.00% | 44.17% | 7.25 pp | -114 | 43 | -2.65 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 608 | 298 | 310 | 49.01% | 49.58% | 49.79% | 0.99 pp | -12 | 38 | -0.32 |
| BTC Daily | mlp_sklearn | MLPClassifier | 608 | 297 | 311 | 48.85% | 47.50% | 49.79% | 1.15 pp | -14 | 38 | -0.37 |
| BTC Daily | nn | NN | 608 | 285 | 323 | 46.88% | 44.17% | 48.12% | 3.12 pp | -38 | 38 | -1.00 |
| BTC Daily | lstm | LSTM | 608 | 272 | 336 | 44.74% | 44.17% | 44.58% | 5.26 pp | -64 | 38 | -1.68 |
| BTC Daily | rf | RandomForest | 608 | 261 | 347 | 42.93% | 43.33% | 43.33% | 7.07 pp | -86 | 38 | -2.26 |
| BTC Daily | xgb | XGBoost | 618 | 246 | 372 | 39.81% | 33.33% | 40.00% | 10.19 pp | -126 | 38 | -3.32 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 380 | 186 | 194 | 48.95% | 47.50% | 48.95% | 1.05 pp | -8 | 39 | -0.21 |
| BTC Market Hours | transformer | Transformer | 380 | 178 | 202 | 46.84% | 44.17% | 46.84% | 3.16 pp | -24 | 39 | -0.62 |
| BTC Market Hours | nn | NN | 380 | 176 | 204 | 46.32% | 49.17% | 46.32% | 3.68 pp | -28 | 39 | -0.72 |
| BTC Market Hours | lstm | LSTM | 380 | 163 | 217 | 42.89% | 44.17% | 42.89% | 7.11 pp | -54 | 39 | -1.38 |
| BTC Market Hours | rf | RandomForest | 380 | 160 | 220 | 42.11% | 40.42% | 42.11% | 7.89 pp | -60 | 39 | -1.54 |
| BTC Market Hours | xgb | XGBoost | 380 | 151 | 229 | 39.74% | 38.33% | 39.74% | 10.26 pp | -78 | 39 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 434 | 198 | 236 | 45.62% | 45.83% | 45.62% | 4.38 pp | -38 | 39 | -0.97 |
| BTC Market Hours Daily | transformer | Transformer | 434 | 198 | 236 | 45.62% | 47.08% | 45.62% | 4.38 pp | -38 | 39 | -0.97 |
| BTC Market Hours Daily | nn | NN | 434 | 197 | 237 | 45.39% | 46.25% | 45.39% | 4.61 pp | -40 | 39 | -1.03 |
| BTC Market Hours Daily | rf | RandomForest | 434 | 175 | 259 | 40.32% | 39.17% | 40.32% | 9.68 pp | -84 | 39 | -2.15 |
| BTC Market Hours Daily | lstm | LSTM | 434 | 171 | 263 | 39.40% | 37.92% | 39.40% | 10.60 pp | -92 | 39 | -2.36 |
| BTC Market Hours Daily | xgb | XGBoost | 434 | 170 | 264 | 39.17% | 38.33% | 39.17% | 10.83 pp | -94 | 39 | -2.41 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 40 | 23 | 17 | 57.50% | 57.50% | 57.50% | 7.50 pp | 6 | 5 | 1.20 |
| Consolidated Hourly | lstm | LSTM | 40 | 21 | 19 | 52.50% | 52.50% | 52.50% | 2.50 pp | 2 | 5 | 0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 5 | -0.40 |
| Consolidated Hourly | transformer | Transformer | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | nn | NN | 40 | 13 | 27 | 32.50% | 32.50% | 32.50% | 17.50 pp | -14 | 5 | -2.80 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 40 | 23 | 17 | 57.50% | 57.50% | 57.50% | 7.50 pp | 6 | 5 | 1.20 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 40 | 21 | 19 | 52.50% | 52.50% | 52.50% | 2.50 pp | 2 | 5 | 0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 5 | -0.40 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | nn | NN | 40 | 13 | 27 | 32.50% | 32.50% | 32.50% | 17.50 pp | -14 | 5 | -2.80 |

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
