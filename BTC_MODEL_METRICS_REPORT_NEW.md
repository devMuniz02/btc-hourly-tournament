# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-08-31T08:08:34.325700+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 142 | 82 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 178 | 118 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 210 | 106 | 104 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 210 | 106 | 104 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T12:00:00+00:00 | 85 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T12:00:00+00:00 | 85 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T12:00:00+00:00 | 85 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-26T12:00:00+00:00 | 86 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| BTC Hourly | nn | NN | 82 | 44 | 38 | 53.66% | 53.66% | 53.66% | 3.66 pp | 6 | 4 | 1.50 |
| BTC Daily | mlp_sklearn | MLPClassifier | 108 | 57 | 51 | 52.78% | 52.78% | 52.78% | 2.78 pp | 6 | 5 | 1.20 |
| Consolidated Market Hours Daily | lstm | LSTM | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| BTC Market Hours | nn | NN | 106 | 57 | 49 | 53.77% | 53.77% | 53.77% | 3.77 pp | 8 | 9 | 0.89 |
| Consolidated Hourly | rf | RandomForest | 85 | 45 | 40 | 52.94% | 52.94% | 52.94% | 2.94 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 85 | 45 | 40 | 52.94% | 52.94% | 52.94% | 2.94 pp | 5 | 9 | 0.56 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 82 | 42 | 40 | 51.22% | 51.22% | 51.22% | 1.22 pp | 2 | 4 | 0.50 |
| BTC Daily | nn | NN | 108 | 55 | 53 | 50.93% | 50.93% | 50.93% | 0.93 pp | 2 | 5 | 0.40 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| BTC Hourly | transformer | Transformer | 82 | 41 | 41 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| Consolidated Market Hours | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Hourly | lstm | LSTM | 85 | 41 | 44 | 48.24% | 48.24% | 48.24% | 1.76 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 85 | 41 | 44 | 48.24% | 48.24% | 48.24% | 1.76 pp | -3 | 9 | -0.33 |
| BTC Market Hours | rf | RandomForest | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 9 | -0.44 |
| Consolidated Hourly | nn | NN | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | nn | NN | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 106 | 50 | 56 | 47.17% | 47.17% | 47.17% | 2.83 pp | -6 | 10 | -0.60 |
| BTC Market Hours Daily | nn | NN | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 10 | -0.80 |
| BTC Market Hours Daily | rf | RandomForest | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 10 | -0.80 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 106 | 48 | 58 | 45.28% | 45.28% | 45.28% | 4.72 pp | -10 | 9 | -1.11 |
| BTC Market Hours Daily | transformer | Transformer | 106 | 47 | 59 | 44.34% | 44.34% | 44.34% | 5.66 pp | -12 | 10 | -1.20 |
| Consolidated Hourly | transformer | Transformer | 85 | 37 | 48 | 43.53% | 43.53% | 43.53% | 6.47 pp | -11 | 9 | -1.22 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 85 | 37 | 48 | 43.53% | 43.53% | 43.53% | 6.47 pp | -11 | 9 | -1.22 |
| BTC Hourly | rf | RandomForest | 82 | 37 | 45 | 45.12% | 45.12% | 45.12% | 4.88 pp | -8 | 4 | -2.00 |
| BTC Market Hours Daily | xgb | XGBoost | 106 | 43 | 63 | 40.57% | 40.57% | 40.57% | 9.43 pp | -20 | 10 | -2.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |
| BTC Market Hours | transformer | Transformer | 106 | 43 | 63 | 40.57% | 40.57% | 40.57% | 9.43 pp | -20 | 9 | -2.22 |
| BTC Daily | rf | RandomForest | 108 | 48 | 60 | 44.44% | 44.44% | 44.44% | 5.56 pp | -12 | 5 | -2.40 |
| BTC Daily | transformer | Transformer | 108 | 48 | 60 | 44.44% | 44.44% | 44.44% | 5.56 pp | -12 | 5 | -2.40 |
| BTC Market Hours | xgb | XGBoost | 106 | 42 | 64 | 39.62% | 39.62% | 39.62% | 10.38 pp | -22 | 9 | -2.44 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 3 | 0 | 3 | 0.00% | 0.00% | 0.00% | 50.00 pp | -3 | 1 | -3.00 |
| BTC Market Hours | lstm | LSTM | 106 | 36 | 70 | 33.96% | 33.96% | 33.96% | 16.04 pp | -34 | 9 | -3.78 |
| BTC Market Hours Daily | lstm | LSTM | 106 | 33 | 73 | 31.13% | 31.13% | 31.13% | 18.87 pp | -40 | 10 | -4.00 |
| BTC Daily | lstm | LSTM | 108 | 42 | 66 | 38.89% | 38.89% | 38.89% | 11.11 pp | -24 | 5 | -4.80 |
| BTC Hourly | xgb | XGBoost | 82 | 31 | 51 | 37.80% | 37.80% | 37.80% | 12.20 pp | -20 | 4 | -5.00 |
| BTC Daily | xgb | XGBoost | 118 | 44 | 74 | 37.29% | 37.29% | 37.29% | 12.71 pp | -30 | 6 | -5.00 |
| BTC Hourly | lstm | LSTM | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 4 | -6.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | nn | NN | 82 | 44 | 38 | 53.66% | 53.66% | 53.66% | 3.66 pp | 6 | 4 | 1.50 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 82 | 42 | 40 | 51.22% | 51.22% | 51.22% | 1.22 pp | 2 | 4 | 0.50 |
| BTC Hourly | transformer | Transformer | 82 | 41 | 41 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 4 | 0.00 |
| BTC Hourly | rf | RandomForest | 82 | 37 | 45 | 45.12% | 45.12% | 45.12% | 4.88 pp | -8 | 4 | -2.00 |
| BTC Hourly | xgb | XGBoost | 82 | 31 | 51 | 37.80% | 37.80% | 37.80% | 12.20 pp | -20 | 4 | -5.00 |
| BTC Hourly | lstm | LSTM | 82 | 29 | 53 | 35.37% | 35.37% | 35.37% | 14.63 pp | -24 | 4 | -6.00 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 108 | 57 | 51 | 52.78% | 52.78% | 52.78% | 2.78 pp | 6 | 5 | 1.20 |
| BTC Daily | nn | NN | 108 | 55 | 53 | 50.93% | 50.93% | 50.93% | 0.93 pp | 2 | 5 | 0.40 |
| BTC Daily | rf | RandomForest | 108 | 48 | 60 | 44.44% | 44.44% | 44.44% | 5.56 pp | -12 | 5 | -2.40 |
| BTC Daily | transformer | Transformer | 108 | 48 | 60 | 44.44% | 44.44% | 44.44% | 5.56 pp | -12 | 5 | -2.40 |
| BTC Daily | lstm | LSTM | 108 | 42 | 66 | 38.89% | 38.89% | 38.89% | 11.11 pp | -24 | 5 | -4.80 |
| BTC Daily | xgb | XGBoost | 118 | 44 | 74 | 37.29% | 37.29% | 37.29% | 12.71 pp | -30 | 6 | -5.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 106 | 57 | 49 | 53.77% | 53.77% | 53.77% | 3.77 pp | 8 | 9 | 0.89 |
| BTC Market Hours | rf | RandomForest | 106 | 51 | 55 | 48.11% | 48.11% | 48.11% | 1.89 pp | -4 | 9 | -0.44 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 106 | 48 | 58 | 45.28% | 45.28% | 45.28% | 4.72 pp | -10 | 9 | -1.11 |
| BTC Market Hours | transformer | Transformer | 106 | 43 | 63 | 40.57% | 40.57% | 40.57% | 9.43 pp | -20 | 9 | -2.22 |
| BTC Market Hours | xgb | XGBoost | 106 | 42 | 64 | 39.62% | 39.62% | 39.62% | 10.38 pp | -22 | 9 | -2.44 |
| BTC Market Hours | lstm | LSTM | 106 | 36 | 70 | 33.96% | 33.96% | 33.96% | 16.04 pp | -34 | 9 | -3.78 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 106 | 50 | 56 | 47.17% | 47.17% | 47.17% | 2.83 pp | -6 | 10 | -0.60 |
| BTC Market Hours Daily | nn | NN | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 10 | -0.80 |
| BTC Market Hours Daily | rf | RandomForest | 106 | 49 | 57 | 46.23% | 46.23% | 46.23% | 3.77 pp | -8 | 10 | -0.80 |
| BTC Market Hours Daily | transformer | Transformer | 106 | 47 | 59 | 44.34% | 44.34% | 44.34% | 5.66 pp | -12 | 10 | -1.20 |
| BTC Market Hours Daily | xgb | XGBoost | 106 | 43 | 63 | 40.57% | 40.57% | 40.57% | 9.43 pp | -20 | 10 | -2.00 |
| BTC Market Hours Daily | lstm | LSTM | 106 | 33 | 73 | 31.13% | 31.13% | 31.13% | 18.87 pp | -40 | 10 | -4.00 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 85 | 45 | 40 | 52.94% | 52.94% | 52.94% | 2.94 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | lstm | LSTM | 85 | 41 | 44 | 48.24% | 48.24% | 48.24% | 1.76 pp | -3 | 9 | -0.33 |
| Consolidated Hourly | nn | NN | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | transformer | Transformer | 85 | 37 | 48 | 43.53% | 43.53% | 43.53% | 6.47 pp | -11 | 9 | -1.22 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 85 | 45 | 40 | 52.94% | 52.94% | 52.94% | 2.94 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 85 | 44 | 41 | 51.76% | 51.76% | 51.76% | 1.76 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 85 | 41 | 44 | 48.24% | 48.24% | 48.24% | 1.76 pp | -3 | 9 | -0.33 |
| Consolidated Daily/Hourly Refresh | nn | NN | 85 | 40 | 45 | 47.06% | 47.06% | 47.06% | 2.94 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 85 | 37 | 48 | 43.53% | 43.53% | 43.53% | 6.47 pp | -11 | 9 | -1.22 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | nn | NN | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | rf | RandomForest | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | transformer | Transformer | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | xgb | XGBoost | 2 | 2 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 2 | 1 | 2.00 |
| Consolidated Market Hours | lstm | LSTM | 2 | 1 | 1 | 50.00% | 50.00% | 50.00% | 0.00 pp | 0 | 1 | 0.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 2 | 0 | 2 | 0.00% | 0.00% | 0.00% | 50.00 pp | -2 | 1 | -2.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | nn | NN | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 3 | 3 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 3 | 1 | 3.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 3 | 2 | 1 | 66.67% | 66.67% | 66.67% | 16.67 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 3 | 0 | 3 | 0.00% | 0.00% | 0.00% | 50.00 pp | -3 | 1 | -3.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
