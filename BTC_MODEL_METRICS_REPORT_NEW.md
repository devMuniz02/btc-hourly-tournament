# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-30T22:23:42.248748+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 136 | 76 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 171 | 111 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-30 21:00:00+00:00 | 200 | 99 | 101 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-30 21:00:00+00:00 | 200 | 99 | 101 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 08:00:00+00:00 | 79 | 79 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 08:00:00+00:00 | 79 | 79 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 08:00:00+00:00 | 79 | 0 | 79 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 08:00:00+00:00 | 79 | 0 | 79 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 99 | 56 | 43 | 56.57% | 56.57% | 56.57% | 6.57 pp | 13 | 8 | 1.62 |
| BTC Hourly | transformer | Transformer | 76 | 40 | 36 | 52.63% | 52.63% | 52.63% | 2.63 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | rf | RandomForest | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| BTC Daily | mlp_sklearn | MLPClassifier | 101 | 52 | 49 | 51.49% | 51.49% | 51.49% | 1.49 pp | 3 | 5 | 0.60 |
| Consolidated Hourly | lstm | LSTM | 79 | 40 | 39 | 50.63% | 50.63% | 50.63% | 0.63 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 79 | 40 | 39 | 50.63% | 50.63% | 50.63% | 0.63 pp | 1 | 8 | 0.12 |
| BTC Market Hours | rf | RandomForest | 99 | 50 | 49 | 50.51% | 50.51% | 50.51% | 0.51 pp | 1 | 8 | 0.12 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 99 | 48 | 51 | 48.48% | 48.48% | 48.48% | 1.52 pp | -3 | 9 | -0.33 |
| BTC Hourly | nn | NN | 76 | 37 | 39 | 48.68% | 48.68% | 48.68% | 1.32 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | nn | NN | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| BTC Market Hours Daily | rf | RandomForest | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 8 | -0.62 |
| Consolidated Hourly | xgb | XGBoost | 79 | 37 | 42 | 46.84% | 46.84% | 46.84% | 3.16 pp | -5 | 8 | -0.62 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 79 | 37 | 42 | 46.84% | 46.84% | 46.84% | 3.16 pp | -5 | 8 | -0.62 |
| BTC Daily | nn | NN | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 5 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 99 | 45 | 54 | 45.45% | 45.45% | 45.45% | 4.55 pp | -9 | 9 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 8 | -1.12 |
| BTC Daily | transformer | Transformer | 101 | 46 | 55 | 45.54% | 45.54% | 45.54% | 4.46 pp | -9 | 5 | -1.80 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 4 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 99 | 40 | 59 | 40.40% | 40.40% | 40.40% | 9.60 pp | -19 | 9 | -2.11 |
| Consolidated Hourly | nn | NN | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 8 | -2.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 8 | -2.12 |
| BTC Daily | rf | RandomForest | 101 | 44 | 57 | 43.56% | 43.56% | 43.56% | 6.44 pp | -13 | 5 | -2.60 |
| BTC Market Hours | transformer | Transformer | 99 | 39 | 60 | 39.39% | 39.39% | 39.39% | 10.61 pp | -21 | 8 | -2.62 |
| BTC Market Hours | xgb | XGBoost | 99 | 39 | 60 | 39.39% | 39.39% | 39.39% | 10.61 pp | -21 | 8 | -2.62 |
| BTC Hourly | rf | RandomForest | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 4 | -3.50 |
| BTC Market Hours | lstm | LSTM | 99 | 35 | 64 | 35.35% | 35.35% | 35.35% | 14.65 pp | -29 | 8 | -3.62 |
| BTC Market Hours Daily | lstm | LSTM | 99 | 33 | 66 | 33.33% | 33.33% | 33.33% | 16.67 pp | -33 | 9 | -3.67 |
| BTC Daily | lstm | LSTM | 101 | 40 | 61 | 39.60% | 39.60% | 39.60% | 10.40 pp | -21 | 5 | -4.20 |
| BTC Daily | xgb | XGBoost | 111 | 39 | 72 | 35.14% | 35.14% | 35.14% | 14.86 pp | -33 | 6 | -5.50 |
| BTC Hourly | lstm | LSTM | 76 | 26 | 50 | 34.21% | 34.21% | 34.21% | 15.79 pp | -24 | 4 | -6.00 |
| BTC Hourly | xgb | XGBoost | 76 | 26 | 50 | 34.21% | 34.21% | 34.21% | 15.79 pp | -24 | 4 | -6.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 76 | 40 | 36 | 52.63% | 52.63% | 52.63% | 2.63 pp | 4 | 4 | 1.00 |
| BTC Hourly | nn | NN | 76 | 37 | 39 | 48.68% | 48.68% | 48.68% | 1.32 pp | -2 | 4 | -0.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 76 | 34 | 42 | 44.74% | 44.74% | 44.74% | 5.26 pp | -8 | 4 | -2.00 |
| BTC Hourly | rf | RandomForest | 76 | 31 | 45 | 40.79% | 40.79% | 40.79% | 9.21 pp | -14 | 4 | -3.50 |
| BTC Hourly | lstm | LSTM | 76 | 26 | 50 | 34.21% | 34.21% | 34.21% | 15.79 pp | -24 | 4 | -6.00 |
| BTC Hourly | xgb | XGBoost | 76 | 26 | 50 | 34.21% | 34.21% | 34.21% | 15.79 pp | -24 | 4 | -6.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 101 | 52 | 49 | 51.49% | 51.49% | 51.49% | 1.49 pp | 3 | 5 | 0.60 |
| BTC Daily | nn | NN | 101 | 48 | 53 | 47.52% | 47.52% | 47.52% | 2.48 pp | -5 | 5 | -1.00 |
| BTC Daily | transformer | Transformer | 101 | 46 | 55 | 45.54% | 45.54% | 45.54% | 4.46 pp | -9 | 5 | -1.80 |
| BTC Daily | rf | RandomForest | 101 | 44 | 57 | 43.56% | 43.56% | 43.56% | 6.44 pp | -13 | 5 | -2.60 |
| BTC Daily | lstm | LSTM | 101 | 40 | 61 | 39.60% | 39.60% | 39.60% | 10.40 pp | -21 | 5 | -4.20 |
| BTC Daily | xgb | XGBoost | 111 | 39 | 72 | 35.14% | 35.14% | 35.14% | 14.86 pp | -33 | 6 | -5.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 99 | 56 | 43 | 56.57% | 56.57% | 56.57% | 6.57 pp | 13 | 8 | 1.62 |
| BTC Market Hours | rf | RandomForest | 99 | 50 | 49 | 50.51% | 50.51% | 50.51% | 0.51 pp | 1 | 8 | 0.12 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 8 | -0.62 |
| BTC Market Hours | transformer | Transformer | 99 | 39 | 60 | 39.39% | 39.39% | 39.39% | 10.61 pp | -21 | 8 | -2.62 |
| BTC Market Hours | xgb | XGBoost | 99 | 39 | 60 | 39.39% | 39.39% | 39.39% | 10.61 pp | -21 | 8 | -2.62 |
| BTC Market Hours | lstm | LSTM | 99 | 35 | 64 | 35.35% | 35.35% | 35.35% | 14.65 pp | -29 | 8 | -3.62 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 99 | 48 | 51 | 48.48% | 48.48% | 48.48% | 1.52 pp | -3 | 9 | -0.33 |
| BTC Market Hours Daily | nn | NN | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| BTC Market Hours Daily | rf | RandomForest | 99 | 47 | 52 | 47.47% | 47.47% | 47.47% | 2.53 pp | -5 | 9 | -0.56 |
| BTC Market Hours Daily | transformer | Transformer | 99 | 45 | 54 | 45.45% | 45.45% | 45.45% | 4.55 pp | -9 | 9 | -1.00 |
| BTC Market Hours Daily | xgb | XGBoost | 99 | 40 | 59 | 40.40% | 40.40% | 40.40% | 9.60 pp | -19 | 9 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 99 | 33 | 66 | 33.33% | 33.33% | 33.33% | 16.67 pp | -33 | 9 | -3.67 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | rf | RandomForest | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| Consolidated Hourly | lstm | LSTM | 79 | 40 | 39 | 50.63% | 50.63% | 50.63% | 0.63 pp | 1 | 8 | 0.12 |
| Consolidated Hourly | xgb | XGBoost | 79 | 37 | 42 | 46.84% | 46.84% | 46.84% | 3.16 pp | -5 | 8 | -0.62 |
| Consolidated Hourly | transformer | Transformer | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 8 | -1.12 |
| Consolidated Hourly | nn | NN | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 8 | -2.12 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 79 | 43 | 36 | 54.43% | 54.43% | 54.43% | 4.43 pp | 7 | 8 | 0.88 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 79 | 40 | 39 | 50.63% | 50.63% | 50.63% | 0.63 pp | 1 | 8 | 0.12 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 79 | 37 | 42 | 46.84% | 46.84% | 46.84% | 3.16 pp | -5 | 8 | -0.62 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 79 | 35 | 44 | 44.30% | 44.30% | 44.30% | 5.70 pp | -9 | 8 | -1.12 |
| Consolidated Daily/Hourly Refresh | nn | NN | 79 | 31 | 48 | 39.24% | 39.24% | 39.24% | 10.76 pp | -17 | 8 | -2.12 |

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
