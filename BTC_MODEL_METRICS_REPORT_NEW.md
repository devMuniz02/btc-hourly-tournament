# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-27T12:54:42.405506+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 102 | 13 | 89 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 109 | 49 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 89 | 37 | 52 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 88 | 36 | 52 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 10:00:00+00:00 | 26 | 26 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 10:00:00+00:00 | 26 | 26 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 10:00:00+00:00 | 26 | 0 | 26 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 10:00:00+00:00 | 26 | 0 | 26 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 26 | 17 | 9 | 65.38% | 65.38% | 65.38% | 15.38 pp | 8 | 3 | 2.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 26 | 17 | 9 | 65.38% | 65.38% | 65.38% | 15.38 pp | 8 | 3 | 2.67 |
| BTC Daily | transformer | Transformer | 39 | 22 | 17 | 56.41% | 56.41% | 56.41% | 6.41 pp | 5 | 2 | 2.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 13 | 7 | 6 | 53.85% | 53.85% | 53.85% | 3.85 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | transformer | Transformer | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 3 | 0.67 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 36 | 19 | 17 | 52.78% | 52.78% | 52.78% | 2.78 pp | 2 | 4 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 36 | 19 | 17 | 52.78% | 52.78% | 52.78% | 2.78 pp | 2 | 4 | 0.50 |
| BTC Daily | nn | NN | 39 | 20 | 19 | 51.28% | 51.28% | 51.28% | 1.28 pp | 1 | 2 | 0.50 |
| BTC Market Hours | nn | NN | 37 | 19 | 18 | 51.35% | 51.35% | 51.35% | 1.35 pp | 1 | 3 | 0.33 |
| BTC Market Hours | rf | RandomForest | 37 | 19 | 18 | 51.35% | 51.35% | 51.35% | 1.35 pp | 1 | 3 | 0.33 |
| BTC Market Hours | transformer | Transformer | 37 | 18 | 19 | 48.65% | 48.65% | 48.65% | 1.35 pp | -1 | 3 | -0.33 |
| BTC Daily | mlp_sklearn | MLPClassifier | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 2 | -0.50 |
| BTC Market Hours Daily | rf | RandomForest | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | xgb | XGBoost | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 4 | -0.50 |
| BTC Hourly | lstm | LSTM | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| BTC Market Hours Daily | nn | NN | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 3 | -1.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 3 | -1.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 3 | -1.67 |
| BTC Daily | rf | RandomForest | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 2 | -2.50 |
| BTC Hourly | nn | NN | 13 | 5 | 8 | 38.46% | 38.46% | 38.46% | 11.54 pp | -3 | 1 | -3.00 |
| BTC Market Hours Daily | lstm | LSTM | 36 | 11 | 25 | 30.56% | 30.56% | 30.56% | 19.44 pp | -14 | 4 | -3.50 |
| Consolidated Hourly | nn | NN | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 3 | -4.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 3 | -4.00 |
| BTC Daily | xgb | XGBoost | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 3 | -4.33 |
| BTC Market Hours | lstm | LSTM | 37 | 12 | 25 | 32.43% | 32.43% | 32.43% | 17.57 pp | -13 | 3 | -4.33 |
| BTC Hourly | rf | RandomForest | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| BTC Hourly | transformer | Transformer | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| BTC Daily | lstm | LSTM | 39 | 13 | 26 | 33.33% | 33.33% | 33.33% | 16.67 pp | -13 | 2 | -6.50 |
| BTC Hourly | xgb | XGBoost | 13 | 3 | 10 | 23.08% | 23.08% | 23.08% | 26.92 pp | -7 | 1 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 13 | 7 | 6 | 53.85% | 53.85% | 53.85% | 3.85 pp | 1 | 1 | 1.00 |
| BTC Hourly | lstm | LSTM | 13 | 6 | 7 | 46.15% | 46.15% | 46.15% | 3.85 pp | -1 | 1 | -1.00 |
| BTC Hourly | nn | NN | 13 | 5 | 8 | 38.46% | 38.46% | 38.46% | 11.54 pp | -3 | 1 | -3.00 |
| BTC Hourly | rf | RandomForest | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| BTC Hourly | transformer | Transformer | 13 | 4 | 9 | 30.77% | 30.77% | 30.77% | 19.23 pp | -5 | 1 | -5.00 |
| BTC Hourly | xgb | XGBoost | 13 | 3 | 10 | 23.08% | 23.08% | 23.08% | 26.92 pp | -7 | 1 | -7.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 39 | 22 | 17 | 56.41% | 56.41% | 56.41% | 6.41 pp | 5 | 2 | 2.50 |
| BTC Daily | nn | NN | 39 | 20 | 19 | 51.28% | 51.28% | 51.28% | 1.28 pp | 1 | 2 | 0.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 39 | 19 | 20 | 48.72% | 48.72% | 48.72% | 1.28 pp | -1 | 2 | -0.50 |
| BTC Daily | rf | RandomForest | 39 | 17 | 22 | 43.59% | 43.59% | 43.59% | 6.41 pp | -5 | 2 | -2.50 |
| BTC Daily | xgb | XGBoost | 49 | 18 | 31 | 36.73% | 36.73% | 36.73% | 13.27 pp | -13 | 3 | -4.33 |
| BTC Daily | lstm | LSTM | 39 | 13 | 26 | 33.33% | 33.33% | 33.33% | 16.67 pp | -13 | 2 | -6.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 37 | 19 | 18 | 51.35% | 51.35% | 51.35% | 1.35 pp | 1 | 3 | 0.33 |
| BTC Market Hours | rf | RandomForest | 37 | 19 | 18 | 51.35% | 51.35% | 51.35% | 1.35 pp | 1 | 3 | 0.33 |
| BTC Market Hours | transformer | Transformer | 37 | 18 | 19 | 48.65% | 48.65% | 48.65% | 1.35 pp | -1 | 3 | -0.33 |
| BTC Market Hours | xgb | XGBoost | 37 | 17 | 20 | 45.95% | 45.95% | 45.95% | 4.05 pp | -3 | 3 | -1.00 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 37 | 16 | 21 | 43.24% | 43.24% | 43.24% | 6.76 pp | -5 | 3 | -1.67 |
| BTC Market Hours | lstm | LSTM | 37 | 12 | 25 | 32.43% | 32.43% | 32.43% | 17.57 pp | -13 | 3 | -4.33 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 36 | 19 | 17 | 52.78% | 52.78% | 52.78% | 2.78 pp | 2 | 4 | 0.50 |
| BTC Market Hours Daily | transformer | Transformer | 36 | 19 | 17 | 52.78% | 52.78% | 52.78% | 2.78 pp | 2 | 4 | 0.50 |
| BTC Market Hours Daily | rf | RandomForest | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | xgb | XGBoost | 36 | 17 | 19 | 47.22% | 47.22% | 47.22% | 2.78 pp | -2 | 4 | -0.50 |
| BTC Market Hours Daily | nn | NN | 36 | 16 | 20 | 44.44% | 44.44% | 44.44% | 5.56 pp | -4 | 4 | -1.00 |
| BTC Market Hours Daily | lstm | LSTM | 36 | 11 | 25 | 30.56% | 30.56% | 30.56% | 19.44 pp | -14 | 4 | -3.50 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 26 | 17 | 9 | 65.38% | 65.38% | 65.38% | 15.38 pp | 8 | 3 | 2.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | transformer | Transformer | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | xgb | XGBoost | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 3 | -1.33 |
| Consolidated Hourly | nn | NN | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 3 | -4.00 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 26 | 17 | 9 | 65.38% | 65.38% | 65.38% | 15.38 pp | 8 | 3 | 2.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 3 | -1.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 3 | -4.00 |

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
