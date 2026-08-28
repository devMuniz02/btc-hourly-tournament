# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T12:47:46.976637+00:00
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
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 984 | 619 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 561 | 381 | 179 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 563 | 435 | 126 | 2 |
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
| BTC Market Hours | mlp_sklearn | MLPClassifier | 381 | 186 | 195 | 48.82% | 47.50% | 48.82% | 1.18 pp | -9 | 39 | -0.23 |
| BTC Daily | transformer | Transformer | 609 | 299 | 310 | 49.10% | 49.58% | 50.00% | 0.90 pp | -11 | 38 | -0.29 |
| BTC Daily | mlp_sklearn | MLPClassifier | 609 | 298 | 311 | 48.93% | 47.50% | 50.00% | 1.07 pp | -13 | 38 | -0.34 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 5 | -0.40 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 40 | 19 | 21 | 47.50% | 47.50% | 47.50% | 2.50 pp | -2 | 5 | -0.40 |
| BTC Market Hours | transformer | Transformer | 381 | 178 | 203 | 46.72% | 43.75% | 46.72% | 3.28 pp | -25 | 39 | -0.64 |
| BTC Market Hours | nn | NN | 381 | 176 | 205 | 46.19% | 48.75% | 46.19% | 3.81 pp | -29 | 39 | -0.74 |
| BTC Daily | nn | NN | 609 | 286 | 323 | 46.96% | 44.17% | 48.33% | 3.04 pp | -37 | 38 | -0.97 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 786 | 372 | 414 | 47.33% | 45.00% | 47.50% | 2.67 pp | -42 | 43 | -0.98 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 435 | 198 | 237 | 45.52% | 45.42% | 45.52% | 4.48 pp | -39 | 39 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 435 | 198 | 237 | 45.52% | 46.67% | 45.52% | 4.48 pp | -39 | 39 | -1.00 |
| BTC Market Hours Daily | nn | NN | 435 | 197 | 238 | 45.29% | 45.83% | 45.29% | 4.71 pp | -41 | 39 | -1.05 |
| BTC Hourly | transformer | Transformer | 786 | 368 | 418 | 46.82% | 42.92% | 45.83% | 3.18 pp | -50 | 43 | -1.16 |
| Consolidated Hourly | transformer | Transformer | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Hourly | xgb | XGBoost | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 40 | 17 | 23 | 42.50% | 42.50% | 42.50% | 7.50 pp | -6 | 5 | -1.20 |
| BTC Market Hours | lstm | LSTM | 381 | 163 | 218 | 42.78% | 43.75% | 42.78% | 7.22 pp | -55 | 39 | -1.41 |
| BTC Market Hours | rf | RandomForest | 381 | 161 | 220 | 42.26% | 40.42% | 42.26% | 7.74 pp | -59 | 39 | -1.51 |
| BTC Daily | lstm | LSTM | 609 | 272 | 337 | 44.66% | 44.17% | 44.58% | 5.34 pp | -65 | 38 | -1.71 |
| BTC Hourly | nn | NN | 786 | 353 | 433 | 44.91% | 40.42% | 45.83% | 5.09 pp | -80 | 43 | -1.86 |
| BTC Market Hours | xgb | XGBoost | 381 | 152 | 229 | 39.90% | 38.33% | 39.90% | 10.10 pp | -77 | 39 | -1.97 |
| BTC Hourly | rf | RandomForest | 786 | 350 | 436 | 44.53% | 43.33% | 43.96% | 5.47 pp | -86 | 43 | -2.00 |
| BTC Hourly | lstm | LSTM | 786 | 347 | 439 | 44.15% | 43.75% | 45.42% | 5.85 pp | -92 | 43 | -2.14 |
| BTC Market Hours Daily | rf | RandomForest | 435 | 175 | 260 | 40.23% | 38.75% | 40.23% | 9.77 pp | -85 | 39 | -2.18 |
| BTC Daily | rf | RandomForest | 609 | 262 | 347 | 43.02% | 43.33% | 43.54% | 6.98 pp | -85 | 38 | -2.24 |
| BTC Market Hours Daily | lstm | LSTM | 435 | 172 | 263 | 39.54% | 37.92% | 39.54% | 10.46 pp | -91 | 39 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 435 | 170 | 265 | 39.08% | 37.92% | 39.08% | 10.92 pp | -95 | 39 | -2.44 |
| BTC Hourly | xgb | XGBoost | 786 | 336 | 450 | 42.75% | 40.00% | 44.17% | 7.25 pp | -114 | 43 | -2.65 |
| Consolidated Hourly | nn | NN | 40 | 13 | 27 | 32.50% | 32.50% | 32.50% | 17.50 pp | -14 | 5 | -2.80 |
| Consolidated Daily/Hourly Refresh | nn | NN | 40 | 13 | 27 | 32.50% | 32.50% | 32.50% | 17.50 pp | -14 | 5 | -2.80 |
| BTC Daily | xgb | XGBoost | 619 | 247 | 372 | 39.90% | 33.75% | 40.00% | 10.10 pp | -125 | 38 | -3.29 |

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
| BTC Daily | transformer | Transformer | 609 | 299 | 310 | 49.10% | 49.58% | 50.00% | 0.90 pp | -11 | 38 | -0.29 |
| BTC Daily | mlp_sklearn | MLPClassifier | 609 | 298 | 311 | 48.93% | 47.50% | 50.00% | 1.07 pp | -13 | 38 | -0.34 |
| BTC Daily | nn | NN | 609 | 286 | 323 | 46.96% | 44.17% | 48.33% | 3.04 pp | -37 | 38 | -0.97 |
| BTC Daily | lstm | LSTM | 609 | 272 | 337 | 44.66% | 44.17% | 44.58% | 5.34 pp | -65 | 38 | -1.71 |
| BTC Daily | rf | RandomForest | 609 | 262 | 347 | 43.02% | 43.33% | 43.54% | 6.98 pp | -85 | 38 | -2.24 |
| BTC Daily | xgb | XGBoost | 619 | 247 | 372 | 39.90% | 33.75% | 40.00% | 10.10 pp | -125 | 38 | -3.29 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 381 | 186 | 195 | 48.82% | 47.50% | 48.82% | 1.18 pp | -9 | 39 | -0.23 |
| BTC Market Hours | transformer | Transformer | 381 | 178 | 203 | 46.72% | 43.75% | 46.72% | 3.28 pp | -25 | 39 | -0.64 |
| BTC Market Hours | nn | NN | 381 | 176 | 205 | 46.19% | 48.75% | 46.19% | 3.81 pp | -29 | 39 | -0.74 |
| BTC Market Hours | lstm | LSTM | 381 | 163 | 218 | 42.78% | 43.75% | 42.78% | 7.22 pp | -55 | 39 | -1.41 |
| BTC Market Hours | rf | RandomForest | 381 | 161 | 220 | 42.26% | 40.42% | 42.26% | 7.74 pp | -59 | 39 | -1.51 |
| BTC Market Hours | xgb | XGBoost | 381 | 152 | 229 | 39.90% | 38.33% | 39.90% | 10.10 pp | -77 | 39 | -1.97 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 435 | 198 | 237 | 45.52% | 45.42% | 45.52% | 4.48 pp | -39 | 39 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 435 | 198 | 237 | 45.52% | 46.67% | 45.52% | 4.48 pp | -39 | 39 | -1.00 |
| BTC Market Hours Daily | nn | NN | 435 | 197 | 238 | 45.29% | 45.83% | 45.29% | 4.71 pp | -41 | 39 | -1.05 |
| BTC Market Hours Daily | rf | RandomForest | 435 | 175 | 260 | 40.23% | 38.75% | 40.23% | 9.77 pp | -85 | 39 | -2.18 |
| BTC Market Hours Daily | lstm | LSTM | 435 | 172 | 263 | 39.54% | 37.92% | 39.54% | 10.46 pp | -91 | 39 | -2.33 |
| BTC Market Hours Daily | xgb | XGBoost | 435 | 170 | 265 | 39.08% | 37.92% | 39.08% | 10.92 pp | -95 | 39 | -2.44 |

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
