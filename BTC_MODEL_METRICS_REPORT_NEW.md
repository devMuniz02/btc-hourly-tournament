# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T21:22:21.005247+00:00
Scope: `new`

## Source Files

- `artifacts/btc/hourly/history.csv`
- `artifacts/btc/daily/history.csv`
- `artifacts/btc/market_hours/history.csv`
- `artifacts/btc/market_hours_daily/history.csv`
- `artifacts/consolidated/history.csv`

## Coverage Metadata

| Variation | Source File | Date Range | Rows | Validated | Missing | Failed |
| --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 135 | 75 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 171 | 111 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 20:00:00+00:00 | 199 | 99 | 100 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 20:00:00+00:00 | 198 | 98 | 100 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 07:00:00+00:00 | 78 | 78 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 07:00:00+00:00 | 78 | 78 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 07:00:00+00:00 | 78 | 0 | 78 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 07:00:00+00:00 | 78 | 0 | 78 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 99 | 55 | 44 | 55.56% | 55.56% | 55.56% | 5.56 pp | 11 | 8 | 1.38 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | rf | RandomForest | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| BTC Hourly | transformer | Transformer | 75 | 39 | 36 | 52.00% | 52.00% | 52.00% | 2.00 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 78 | 40 | 38 | 51.28% | 51.28% | 51.28% | 1.28 pp | 2 | 8 | 0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 78 | 40 | 38 | 51.28% | 51.28% | 51.28% | 1.28 pp | 2 | 8 | 0.25 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 98 | 49 | 49 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| BTC Market Hours | rf | RandomForest | 99 | 49 | 50 | 49.49% | 49.49% | 49.49% | 0.51 pp | -1 | 8 | -0.12 |
| BTC Hourly | nn | NN | 75 | 37 | 38 | 49.33% | 49.33% | 49.33% | 0.67 pp | -1 | 4 | -0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 99 | 48 | 51 | 48.48% | 48.48% | 48.48% | 1.52 pp | -3 | 8 | -0.38 |
| Consolidated Hourly | xgb | XGBoost | 78 | 36 | 42 | 46.15% | 46.15% | 46.15% | 3.85 pp | -6 | 8 | -0.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 78 | 36 | 42 | 46.15% | 46.15% | 46.15% | 3.85 pp | -6 | 8 | -0.75 |
| BTC Market Hours Daily | transformer | Transformer | 98 | 45 | 53 | 45.92% | 45.92% | 45.92% | 4.08 pp | -8 | 9 | -0.89 |
| Consolidated Hourly | transformer | Transformer | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 8 | -1.00 |
| BTC Market Hours Daily | rf | RandomForest | 98 | 44 | 54 | 44.90% | 44.90% | 44.90% | 5.10 pp | -10 | 9 | -1.11 |
| BTC Daily | mlp_sklearn | MLPClassifier | 101 | 47 | 54 | 46.53% | 46.53% | 46.53% | 3.47 pp | -7 | 5 | -1.40 |
| BTC Market Hours Daily | nn | NN | 98 | 42 | 56 | 42.86% | 42.86% | 42.86% | 7.14 pp | -14 | 9 | -1.56 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 75 | 34 | 41 | 45.33% | 45.33% | 45.33% | 4.67 pp | -7 | 4 | -1.75 |
| BTC Daily | transformer | Transformer | 101 | 46 | 55 | 45.54% | 45.54% | 45.54% | 4.46 pp | -9 | 5 | -1.80 |
| BTC Market Hours | lstm | LSTM | 99 | 42 | 57 | 42.42% | 42.42% | 42.42% | 7.58 pp | -15 | 8 | -1.88 |
| BTC Daily | nn | NN | 101 | 45 | 56 | 44.55% | 44.55% | 44.55% | 5.45 pp | -11 | 5 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 98 | 39 | 59 | 39.80% | 39.80% | 39.80% | 10.20 pp | -20 | 9 | -2.22 |
| Consolidated Hourly | nn | NN | 78 | 30 | 48 | 38.46% | 38.46% | 38.46% | 11.54 pp | -18 | 8 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 78 | 30 | 48 | 38.46% | 38.46% | 38.46% | 11.54 pp | -18 | 8 | -2.25 |
| BTC Market Hours | transformer | Transformer | 99 | 40 | 59 | 40.40% | 40.40% | 40.40% | 9.60 pp | -19 | 8 | -2.38 |
| BTC Market Hours | xgb | XGBoost | 99 | 38 | 61 | 38.38% | 38.38% | 38.38% | 11.62 pp | -23 | 8 | -2.88 |
| BTC Market Hours Daily | xgb | XGBoost | 98 | 36 | 62 | 36.73% | 36.73% | 36.73% | 13.27 pp | -26 | 9 | -2.89 |
| BTC Hourly | rf | RandomForest | 75 | 31 | 44 | 41.33% | 41.33% | 41.33% | 8.67 pp | -13 | 4 | -3.25 |
| BTC Daily | rf | RandomForest | 101 | 39 | 62 | 38.61% | 38.61% | 38.61% | 11.39 pp | -23 | 5 | -4.60 |
| BTC Hourly | lstm | LSTM | 75 | 26 | 49 | 34.67% | 34.67% | 34.67% | 15.33 pp | -23 | 4 | -5.75 |
| BTC Hourly | xgb | XGBoost | 75 | 26 | 49 | 34.67% | 34.67% | 34.67% | 15.33 pp | -23 | 4 | -5.75 |
| BTC Daily | lstm | LSTM | 101 | 36 | 65 | 35.64% | 35.64% | 35.64% | 14.36 pp | -29 | 5 | -5.80 |
| BTC Daily | xgb | XGBoost | 111 | 38 | 73 | 34.23% | 34.23% | 34.23% | 15.77 pp | -35 | 6 | -5.83 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 75 | 39 | 36 | 52.00% | 52.00% | 52.00% | 2.00 pp | 3 | 4 | 0.75 |
| BTC Hourly | nn | NN | 75 | 37 | 38 | 49.33% | 49.33% | 49.33% | 0.67 pp | -1 | 4 | -0.25 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 75 | 34 | 41 | 45.33% | 45.33% | 45.33% | 4.67 pp | -7 | 4 | -1.75 |
| BTC Hourly | rf | RandomForest | 75 | 31 | 44 | 41.33% | 41.33% | 41.33% | 8.67 pp | -13 | 4 | -3.25 |
| BTC Hourly | lstm | LSTM | 75 | 26 | 49 | 34.67% | 34.67% | 34.67% | 15.33 pp | -23 | 4 | -5.75 |
| BTC Hourly | xgb | XGBoost | 75 | 26 | 49 | 34.67% | 34.67% | 34.67% | 15.33 pp | -23 | 4 | -5.75 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 101 | 47 | 54 | 46.53% | 46.53% | 46.53% | 3.47 pp | -7 | 5 | -1.40 |
| BTC Daily | transformer | Transformer | 101 | 46 | 55 | 45.54% | 45.54% | 45.54% | 4.46 pp | -9 | 5 | -1.80 |
| BTC Daily | nn | NN | 101 | 45 | 56 | 44.55% | 44.55% | 44.55% | 5.45 pp | -11 | 5 | -2.20 |
| BTC Daily | rf | RandomForest | 101 | 39 | 62 | 38.61% | 38.61% | 38.61% | 11.39 pp | -23 | 5 | -4.60 |
| BTC Daily | lstm | LSTM | 101 | 36 | 65 | 35.64% | 35.64% | 35.64% | 14.36 pp | -29 | 5 | -5.80 |
| BTC Daily | xgb | XGBoost | 111 | 38 | 73 | 34.23% | 34.23% | 34.23% | 15.77 pp | -35 | 6 | -5.83 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 99 | 55 | 44 | 55.56% | 55.56% | 55.56% | 5.56 pp | 11 | 8 | 1.38 |
| BTC Market Hours | rf | RandomForest | 99 | 49 | 50 | 49.49% | 49.49% | 49.49% | 0.51 pp | -1 | 8 | -0.12 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 99 | 48 | 51 | 48.48% | 48.48% | 48.48% | 1.52 pp | -3 | 8 | -0.38 |
| BTC Market Hours | lstm | LSTM | 99 | 42 | 57 | 42.42% | 42.42% | 42.42% | 7.58 pp | -15 | 8 | -1.88 |
| BTC Market Hours | transformer | Transformer | 99 | 40 | 59 | 40.40% | 40.40% | 40.40% | 9.60 pp | -19 | 8 | -2.38 |
| BTC Market Hours | xgb | XGBoost | 99 | 38 | 61 | 38.38% | 38.38% | 38.38% | 11.62 pp | -23 | 8 | -2.88 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 98 | 49 | 49 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 9 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 98 | 45 | 53 | 45.92% | 45.92% | 45.92% | 4.08 pp | -8 | 9 | -0.89 |
| BTC Market Hours Daily | rf | RandomForest | 98 | 44 | 54 | 44.90% | 44.90% | 44.90% | 5.10 pp | -10 | 9 | -1.11 |
| BTC Market Hours Daily | nn | NN | 98 | 42 | 56 | 42.86% | 42.86% | 42.86% | 7.14 pp | -14 | 9 | -1.56 |
| BTC Market Hours Daily | lstm | LSTM | 98 | 39 | 59 | 39.80% | 39.80% | 39.80% | 10.20 pp | -20 | 9 | -2.22 |
| BTC Market Hours Daily | xgb | XGBoost | 98 | 36 | 62 | 36.73% | 36.73% | 36.73% | 13.27 pp | -26 | 9 | -2.89 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | rf | RandomForest | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 78 | 40 | 38 | 51.28% | 51.28% | 51.28% | 1.28 pp | 2 | 8 | 0.25 |
| Consolidated Hourly | xgb | XGBoost | 78 | 36 | 42 | 46.15% | 46.15% | 46.15% | 3.85 pp | -6 | 8 | -0.75 |
| Consolidated Hourly | transformer | Transformer | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 8 | -1.00 |
| Consolidated Hourly | nn | NN | 78 | 30 | 48 | 38.46% | 38.46% | 38.46% | 11.54 pp | -18 | 8 | -2.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 78 | 40 | 38 | 51.28% | 51.28% | 51.28% | 1.28 pp | 2 | 8 | 0.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 78 | 36 | 42 | 46.15% | 46.15% | 46.15% | 3.85 pp | -6 | 8 | -0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 78 | 30 | 48 | 38.46% | 38.46% | 38.46% | 11.54 pp | -18 | 8 | -2.25 |

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
