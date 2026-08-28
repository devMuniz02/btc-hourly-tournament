# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-28T02:24:02.750182+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 23 | 79 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 118 | 58 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 111 | 46 | 65 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 111 | 46 | 65 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 07:00:00+00:00 | 34 | 34 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 07:00:00+00:00 | 34 | 34 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 07:00:00+00:00 | 34 | 0 | 34 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 07:00:00+00:00 | 34 | 0 | 34 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 23 | 13 | 10 | 56.52% | 56.52% | 56.52% | 6.52 pp | 3 | 1 | 3.00 |
| Consolidated Hourly | rf | RandomForest | 34 | 19 | 15 | 55.88% | 55.88% | 55.88% | 5.88 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 34 | 19 | 15 | 55.88% | 55.88% | 55.88% | 5.88 pp | 4 | 4 | 1.00 |
| BTC Market Hours | nn | NN | 46 | 25 | 21 | 54.35% | 54.35% | 54.35% | 4.35 pp | 4 | 4 | 1.00 |
| BTC Daily | transformer | Transformer | 48 | 25 | 23 | 52.08% | 52.08% | 52.08% | 2.08 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | lstm | LSTM | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 4 | 0.50 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 4 | 0.50 |
| BTC Market Hours | rf | RandomForest | 46 | 23 | 23 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 46 | 23 | 23 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 5 | -0.40 |
| BTC Daily | mlp_sklearn | MLPClassifier | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 3 | -0.67 |
| BTC Daily | nn | NN | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 3 | -0.67 |
| BTC Hourly | lstm | LSTM | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 1 | -1.00 |
| BTC Hourly | nn | NN | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 1 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| BTC Market Hours | transformer | Transformer | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| BTC Market Hours Daily | nn | NN | 46 | 19 | 27 | 41.30% | 41.30% | 41.30% | 8.70 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | rf | RandomForest | 46 | 19 | 27 | 41.30% | 41.30% | 41.30% | 8.70 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | xgb | XGBoost | 46 | 19 | 27 | 41.30% | 41.30% | 41.30% | 8.70 pp | -8 | 5 | -1.60 |
| BTC Daily | rf | RandomForest | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 3 | -2.00 |
| Consolidated Hourly | xgb | XGBoost | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 4 | -2.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 4 | -2.00 |
| BTC Market Hours | xgb | XGBoost | 46 | 18 | 28 | 39.13% | 39.13% | 39.13% | 10.87 pp | -10 | 4 | -2.50 |
| BTC Hourly | transformer | Transformer | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 1 | -3.00 |
| Consolidated Hourly | nn | NN | 34 | 10 | 24 | 29.41% | 29.41% | 29.41% | 20.59 pp | -14 | 4 | -3.50 |
| Consolidated Daily/Hourly Refresh | nn | NN | 34 | 10 | 24 | 29.41% | 29.41% | 29.41% | 20.59 pp | -14 | 4 | -3.50 |
| BTC Daily | xgb | XGBoost | 58 | 21 | 37 | 36.21% | 36.21% | 36.21% | 13.79 pp | -16 | 4 | -4.00 |
| BTC Market Hours | lstm | LSTM | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |
| BTC Market Hours Daily | lstm | LSTM | 46 | 12 | 34 | 26.09% | 26.09% | 26.09% | 23.91 pp | -22 | 5 | -4.40 |
| BTC Daily | lstm | LSTM | 48 | 16 | 32 | 33.33% | 33.33% | 33.33% | 16.67 pp | -16 | 3 | -5.33 |
| BTC Hourly | rf | RandomForest | 23 | 8 | 15 | 34.78% | 34.78% | 34.78% | 15.22 pp | -7 | 1 | -7.00 |
| BTC Hourly | xgb | XGBoost | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 1 | -9.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 23 | 13 | 10 | 56.52% | 56.52% | 56.52% | 6.52 pp | 3 | 1 | 3.00 |
| BTC Hourly | lstm | LSTM | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 1 | -1.00 |
| BTC Hourly | nn | NN | 23 | 11 | 12 | 47.83% | 47.83% | 47.83% | 2.17 pp | -1 | 1 | -1.00 |
| BTC Hourly | transformer | Transformer | 23 | 10 | 13 | 43.48% | 43.48% | 43.48% | 6.52 pp | -3 | 1 | -3.00 |
| BTC Hourly | rf | RandomForest | 23 | 8 | 15 | 34.78% | 34.78% | 34.78% | 15.22 pp | -7 | 1 | -7.00 |
| BTC Hourly | xgb | XGBoost | 23 | 7 | 16 | 30.43% | 30.43% | 30.43% | 19.57 pp | -9 | 1 | -9.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 48 | 25 | 23 | 52.08% | 52.08% | 52.08% | 2.08 pp | 2 | 3 | 0.67 |
| BTC Daily | mlp_sklearn | MLPClassifier | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 3 | -0.67 |
| BTC Daily | nn | NN | 48 | 23 | 25 | 47.92% | 47.92% | 47.92% | 2.08 pp | -2 | 3 | -0.67 |
| BTC Daily | rf | RandomForest | 48 | 21 | 27 | 43.75% | 43.75% | 43.75% | 6.25 pp | -6 | 3 | -2.00 |
| BTC Daily | xgb | XGBoost | 58 | 21 | 37 | 36.21% | 36.21% | 36.21% | 13.79 pp | -16 | 4 | -4.00 |
| BTC Daily | lstm | LSTM | 48 | 16 | 32 | 33.33% | 33.33% | 33.33% | 16.67 pp | -16 | 3 | -5.33 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 46 | 25 | 21 | 54.35% | 54.35% | 54.35% | 4.35 pp | 4 | 4 | 1.00 |
| BTC Market Hours | rf | RandomForest | 46 | 23 | 23 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| BTC Market Hours | transformer | Transformer | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 46 | 18 | 28 | 39.13% | 39.13% | 39.13% | 10.87 pp | -10 | 4 | -2.50 |
| BTC Market Hours | lstm | LSTM | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 46 | 23 | 23 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 5 | 0.00 |
| BTC Market Hours Daily | transformer | Transformer | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 5 | -0.40 |
| BTC Market Hours Daily | nn | NN | 46 | 19 | 27 | 41.30% | 41.30% | 41.30% | 8.70 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | rf | RandomForest | 46 | 19 | 27 | 41.30% | 41.30% | 41.30% | 8.70 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | xgb | XGBoost | 46 | 19 | 27 | 41.30% | 41.30% | 41.30% | 8.70 pp | -8 | 5 | -1.60 |
| BTC Market Hours Daily | lstm | LSTM | 46 | 12 | 34 | 26.09% | 26.09% | 26.09% | 23.91 pp | -22 | 5 | -4.40 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 34 | 19 | 15 | 55.88% | 55.88% | 55.88% | 5.88 pp | 4 | 4 | 1.00 |
| Consolidated Hourly | lstm | LSTM | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 4 | 0.50 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | transformer | Transformer | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 4 | -2.00 |
| Consolidated Hourly | nn | NN | 34 | 10 | 24 | 29.41% | 29.41% | 29.41% | 20.59 pp | -14 | 4 | -3.50 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 34 | 19 | 15 | 55.88% | 55.88% | 55.88% | 5.88 pp | 4 | 4 | 1.00 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 34 | 18 | 16 | 52.94% | 52.94% | 52.94% | 2.94 pp | 2 | 4 | 0.50 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 34 | 15 | 19 | 44.12% | 44.12% | 44.12% | 5.88 pp | -4 | 4 | -1.00 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 34 | 13 | 21 | 38.24% | 38.24% | 38.24% | 11.76 pp | -8 | 4 | -2.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 34 | 10 | 24 | 29.41% | 29.41% | 29.41% | 20.59 pp | -14 | 4 | -3.50 |

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
