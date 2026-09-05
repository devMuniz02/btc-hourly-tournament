# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T15:58:58.281883+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 229 | 169 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 265 | 205 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 14:00:00+00:00 | 365 | 193 | 172 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 14:00:00+00:00 | 365 | 193 | 172 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 16:00:00+00:00 | 165 | 165 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 16:00:00+00:00 | 165 | 165 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 16:00:00+00:00 | 165 | 45 | 120 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-29 16:00:00+00:00 | 165 | 45 | 120 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 193 | 102 | 91 | 52.85% | 52.85% | 52.85% | 2.85 pp | 11 | 16 | 0.69 |
| BTC Market Hours | nn | NN | 193 | 100 | 93 | 51.81% | 51.81% | 51.81% | 1.81 pp | 7 | 15 | 0.47 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 169 | 86 | 83 | 50.89% | 50.89% | 50.89% | 0.89 pp | 3 | 7 | 0.43 |
| BTC Market Hours | transformer | Transformer | 193 | 96 | 97 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 15 | -0.07 |
| Consolidated Hourly | rf | RandomForest | 165 | 82 | 83 | 49.70% | 49.70% | 49.70% | 0.30 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 165 | 82 | 83 | 49.70% | 49.70% | 49.70% | 0.30 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 193 | 92 | 101 | 47.67% | 47.67% | 47.67% | 2.33 pp | -9 | 16 | -0.56 |
| BTC Market Hours Daily | nn | NN | 193 | 92 | 101 | 47.67% | 47.67% | 47.67% | 2.33 pp | -9 | 16 | -0.56 |
| Consolidated Market Hours | xgb | XGBoost | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | xgb | XGBoost | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 4 | -0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 195 | 94 | 101 | 48.21% | 48.21% | 48.21% | 1.79 pp | -7 | 9 | -0.78 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 193 | 90 | 103 | 46.63% | 46.63% | 46.63% | 3.37 pp | -13 | 15 | -0.87 |
| BTC Market Hours | rf | RandomForest | 193 | 90 | 103 | 46.63% | 46.63% | 46.63% | 3.37 pp | -13 | 15 | -0.87 |
| BTC Hourly | transformer | Transformer | 169 | 81 | 88 | 47.93% | 47.93% | 47.93% | 2.07 pp | -7 | 7 | -1.00 |
| Consolidated Hourly | lstm | LSTM | 165 | 76 | 89 | 46.06% | 46.06% | 46.06% | 3.94 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 165 | 76 | 89 | 46.06% | 46.06% | 46.06% | 3.94 pp | -13 | 12 | -1.08 |
| Consolidated Market Hours | lstm | LSTM | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 193 | 86 | 107 | 44.56% | 44.56% | 44.56% | 5.44 pp | -21 | 16 | -1.31 |
| BTC Market Hours | xgb | XGBoost | 193 | 84 | 109 | 43.52% | 43.52% | 43.52% | 6.48 pp | -25 | 15 | -1.67 |
| Consolidated Hourly | transformer | Transformer | 165 | 72 | 93 | 43.64% | 43.64% | 43.64% | 6.36 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 165 | 72 | 93 | 43.64% | 43.64% | 43.64% | 6.36 pp | -21 | 12 | -1.75 |
| Consolidated Market Hours | rf | RandomForest | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 4 | -1.75 |
| BTC Market Hours | lstm | LSTM | 193 | 83 | 110 | 43.01% | 43.01% | 43.01% | 6.99 pp | -27 | 15 | -1.80 |
| Consolidated Hourly | xgb | XGBoost | 165 | 71 | 94 | 43.03% | 43.03% | 43.03% | 6.97 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 165 | 71 | 94 | 43.03% | 43.03% | 43.03% | 6.97 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | nn | NN | 165 | 70 | 95 | 42.42% | 42.42% | 42.42% | 7.58 pp | -25 | 12 | -2.08 |
| Consolidated Daily/Hourly Refresh | nn | NN | 165 | 70 | 95 | 42.42% | 42.42% | 42.42% | 7.58 pp | -25 | 12 | -2.08 |
| BTC Daily | nn | NN | 195 | 88 | 107 | 45.13% | 45.13% | 45.13% | 4.87 pp | -19 | 9 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 193 | 79 | 114 | 40.93% | 40.93% | 40.93% | 9.07 pp | -35 | 16 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 193 | 79 | 114 | 40.93% | 40.93% | 40.93% | 9.07 pp | -35 | 16 | -2.19 |
| Consolidated Market Hours | nn | NN | 45 | 17 | 28 | 37.78% | 37.78% | 37.78% | 12.22 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 45 | 17 | 28 | 37.78% | 37.78% | 37.78% | 12.22 pp | -11 | 4 | -2.75 |
| BTC Daily | transformer | Transformer | 195 | 84 | 111 | 43.08% | 43.08% | 43.08% | 6.92 pp | -27 | 9 | -3.00 |
| Consolidated Market Hours | transformer | Transformer | 45 | 16 | 29 | 35.56% | 35.56% | 35.56% | 14.44 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 45 | 16 | 29 | 35.56% | 35.56% | 35.56% | 14.44 pp | -13 | 4 | -3.25 |
| BTC Hourly | nn | NN | 169 | 73 | 96 | 43.20% | 43.20% | 43.20% | 6.80 pp | -23 | 7 | -3.29 |
| BTC Hourly | rf | RandomForest | 169 | 72 | 97 | 42.60% | 42.60% | 42.60% | 7.40 pp | -25 | 7 | -3.57 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 45 | 15 | 30 | 33.33% | 33.33% | 33.33% | 16.67 pp | -15 | 4 | -3.75 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 45 | 15 | 30 | 33.33% | 33.33% | 33.33% | 16.67 pp | -15 | 4 | -3.75 |
| BTC Daily | rf | RandomForest | 195 | 76 | 119 | 38.97% | 38.97% | 38.97% | 11.03 pp | -43 | 9 | -4.78 |
| BTC Daily | xgb | XGBoost | 205 | 74 | 131 | 36.10% | 36.10% | 36.10% | 13.90 pp | -57 | 10 | -5.70 |
| BTC Hourly | lstm | LSTM | 169 | 62 | 107 | 36.69% | 36.69% | 36.69% | 13.31 pp | -45 | 7 | -6.43 |
| BTC Hourly | xgb | XGBoost | 169 | 61 | 108 | 36.09% | 36.09% | 36.09% | 13.91 pp | -47 | 7 | -6.71 |
| BTC Daily | lstm | LSTM | 195 | 67 | 128 | 34.36% | 34.36% | 34.36% | 15.64 pp | -61 | 9 | -6.78 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 169 | 86 | 83 | 50.89% | 50.89% | 50.89% | 0.89 pp | 3 | 7 | 0.43 |
| BTC Hourly | transformer | Transformer | 169 | 81 | 88 | 47.93% | 47.93% | 47.93% | 2.07 pp | -7 | 7 | -1.00 |
| BTC Hourly | nn | NN | 169 | 73 | 96 | 43.20% | 43.20% | 43.20% | 6.80 pp | -23 | 7 | -3.29 |
| BTC Hourly | rf | RandomForest | 169 | 72 | 97 | 42.60% | 42.60% | 42.60% | 7.40 pp | -25 | 7 | -3.57 |
| BTC Hourly | lstm | LSTM | 169 | 62 | 107 | 36.69% | 36.69% | 36.69% | 13.31 pp | -45 | 7 | -6.43 |
| BTC Hourly | xgb | XGBoost | 169 | 61 | 108 | 36.09% | 36.09% | 36.09% | 13.91 pp | -47 | 7 | -6.71 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 195 | 94 | 101 | 48.21% | 48.21% | 48.21% | 1.79 pp | -7 | 9 | -0.78 |
| BTC Daily | nn | NN | 195 | 88 | 107 | 45.13% | 45.13% | 45.13% | 4.87 pp | -19 | 9 | -2.11 |
| BTC Daily | transformer | Transformer | 195 | 84 | 111 | 43.08% | 43.08% | 43.08% | 6.92 pp | -27 | 9 | -3.00 |
| BTC Daily | rf | RandomForest | 195 | 76 | 119 | 38.97% | 38.97% | 38.97% | 11.03 pp | -43 | 9 | -4.78 |
| BTC Daily | xgb | XGBoost | 205 | 74 | 131 | 36.10% | 36.10% | 36.10% | 13.90 pp | -57 | 10 | -5.70 |
| BTC Daily | lstm | LSTM | 195 | 67 | 128 | 34.36% | 34.36% | 34.36% | 15.64 pp | -61 | 9 | -6.78 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 193 | 100 | 93 | 51.81% | 51.81% | 51.81% | 1.81 pp | 7 | 15 | 0.47 |
| BTC Market Hours | transformer | Transformer | 193 | 96 | 97 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 15 | -0.07 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 193 | 90 | 103 | 46.63% | 46.63% | 46.63% | 3.37 pp | -13 | 15 | -0.87 |
| BTC Market Hours | rf | RandomForest | 193 | 90 | 103 | 46.63% | 46.63% | 46.63% | 3.37 pp | -13 | 15 | -0.87 |
| BTC Market Hours | xgb | XGBoost | 193 | 84 | 109 | 43.52% | 43.52% | 43.52% | 6.48 pp | -25 | 15 | -1.67 |
| BTC Market Hours | lstm | LSTM | 193 | 83 | 110 | 43.01% | 43.01% | 43.01% | 6.99 pp | -27 | 15 | -1.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 193 | 102 | 91 | 52.85% | 52.85% | 52.85% | 2.85 pp | 11 | 16 | 0.69 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 193 | 92 | 101 | 47.67% | 47.67% | 47.67% | 2.33 pp | -9 | 16 | -0.56 |
| BTC Market Hours Daily | nn | NN | 193 | 92 | 101 | 47.67% | 47.67% | 47.67% | 2.33 pp | -9 | 16 | -0.56 |
| BTC Market Hours Daily | rf | RandomForest | 193 | 86 | 107 | 44.56% | 44.56% | 44.56% | 5.44 pp | -21 | 16 | -1.31 |
| BTC Market Hours Daily | lstm | LSTM | 193 | 79 | 114 | 40.93% | 40.93% | 40.93% | 9.07 pp | -35 | 16 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 193 | 79 | 114 | 40.93% | 40.93% | 40.93% | 9.07 pp | -35 | 16 | -2.19 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 165 | 82 | 83 | 49.70% | 49.70% | 49.70% | 0.30 pp | -1 | 12 | -0.08 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | lstm | LSTM | 165 | 76 | 89 | 46.06% | 46.06% | 46.06% | 3.94 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | transformer | Transformer | 165 | 72 | 93 | 43.64% | 43.64% | 43.64% | 6.36 pp | -21 | 12 | -1.75 |
| Consolidated Hourly | xgb | XGBoost | 165 | 71 | 94 | 43.03% | 43.03% | 43.03% | 6.97 pp | -23 | 12 | -1.92 |
| Consolidated Hourly | nn | NN | 165 | 70 | 95 | 42.42% | 42.42% | 42.42% | 7.58 pp | -25 | 12 | -2.08 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 165 | 82 | 83 | 49.70% | 49.70% | 49.70% | 0.30 pp | -1 | 12 | -0.08 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 165 | 80 | 85 | 48.48% | 48.48% | 48.48% | 1.52 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 165 | 76 | 89 | 46.06% | 46.06% | 46.06% | 3.94 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 165 | 72 | 93 | 43.64% | 43.64% | 43.64% | 6.36 pp | -21 | 12 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 165 | 71 | 94 | 43.03% | 43.03% | 43.03% | 6.97 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | nn | NN | 165 | 70 | 95 | 42.42% | 42.42% | 42.42% | 7.58 pp | -25 | 12 | -2.08 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours | lstm | LSTM | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours | rf | RandomForest | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours | nn | NN | 45 | 17 | 28 | 37.78% | 37.78% | 37.78% | 12.22 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours | transformer | Transformer | 45 | 16 | 29 | 35.56% | 35.56% | 35.56% | 14.44 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 45 | 15 | 30 | 33.33% | 33.33% | 33.33% | 16.67 pp | -15 | 4 | -3.75 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 45 | 21 | 24 | 46.67% | 46.67% | 46.67% | 3.33 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | lstm | LSTM | 45 | 20 | 25 | 44.44% | 44.44% | 44.44% | 5.56 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | rf | RandomForest | 45 | 19 | 26 | 42.22% | 42.22% | 42.22% | 7.78 pp | -7 | 4 | -1.75 |
| Consolidated Market Hours Daily | nn | NN | 45 | 17 | 28 | 37.78% | 37.78% | 37.78% | 12.22 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | transformer | Transformer | 45 | 16 | 29 | 35.56% | 35.56% | 35.56% | 14.44 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 45 | 15 | 30 | 33.33% | 33.33% | 33.33% | 16.67 pp | -15 | 4 | -3.75 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
