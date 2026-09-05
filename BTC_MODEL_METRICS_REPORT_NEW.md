# BTC Model Metrics Report - New Forward Rows

Generated at: 2026-09-05T19:35:08.719540+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-04-28 00:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 231 | 171 | 60 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 267 | 207 | 60 | 0 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-04-28 00:00:00+00:00 to 2026-09-05 18:00:00+00:00 | 371 | 195 | 176 | 0 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-04-27 23:00:00+00:00 to 2026-09-05 18:00:00+00:00 | 371 | 195 | 176 | 0 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T17:00:00+00:00 | 167 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T17:00:00+00:00 | 167 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T17:00:00+00:00 | 167 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18T07:00:00+00:00 to 2026-05-29T17:00:00+00:00 | 168 | 1 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 195 | 102 | 93 | 52.31% | 52.31% | 52.31% | 2.31 pp | 9 | 16 | 0.56 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 171 | 87 | 84 | 50.88% | 50.88% | 50.88% | 0.88 pp | 3 | 8 | 0.38 |
| BTC Market Hours | nn | NN | 195 | 100 | 95 | 51.28% | 51.28% | 51.28% | 1.28 pp | 5 | 15 | 0.33 |
| BTC Market Hours | transformer | Transformer | 195 | 97 | 98 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 15 | -0.07 |
| Consolidated Hourly | rf | RandomForest | 167 | 82 | 85 | 49.10% | 49.10% | 49.10% | 0.90 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 167 | 82 | 85 | 49.10% | 49.10% | 49.10% | 0.90 pp | -3 | 12 | -0.25 |
| Consolidated Market Hours Daily | xgb | XGBoost | 47 | 23 | 24 | 48.94% | 48.94% | 48.94% | 1.06 pp | -1 | 4 | -0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 195 | 94 | 101 | 48.21% | 48.21% | 48.21% | 1.79 pp | -7 | 16 | -0.44 |
| BTC Market Hours Daily | nn | NN | 195 | 94 | 101 | 48.21% | 48.21% | 48.21% | 1.79 pp | -7 | 16 | -0.44 |
| Consolidated Market Hours | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours Daily | lstm | LSTM | 47 | 22 | 25 | 46.81% | 46.81% | 46.81% | 3.19 pp | -3 | 4 | -0.75 |
| BTC Daily | mlp_sklearn | MLPClassifier | 197 | 95 | 102 | 48.22% | 48.22% | 48.22% | 1.78 pp | -7 | 9 | -0.78 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 195 | 91 | 104 | 46.67% | 46.67% | 46.67% | 3.33 pp | -13 | 15 | -0.87 |
| BTC Market Hours | rf | RandomForest | 195 | 90 | 105 | 46.15% | 46.15% | 46.15% | 3.85 pp | -15 | 15 | -1.00 |
| Consolidated Market Hours | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Hourly | xgb | XGBoost | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| BTC Hourly | transformer | Transformer | 171 | 81 | 90 | 47.37% | 47.37% | 47.37% | 2.63 pp | -9 | 8 | -1.12 |
| Consolidated Market Hours Daily | rf | RandomForest | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 4 | -1.25 |
| BTC Market Hours Daily | rf | RandomForest | 195 | 87 | 108 | 44.62% | 44.62% | 44.62% | 5.38 pp | -21 | 16 | -1.31 |
| Consolidated Hourly | lstm | LSTM | 167 | 75 | 92 | 44.91% | 44.91% | 44.91% | 5.09 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | nn | NN | 167 | 75 | 92 | 44.91% | 44.91% | 44.91% | 5.09 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 167 | 75 | 92 | 44.91% | 44.91% | 44.91% | 5.09 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | nn | NN | 167 | 75 | 92 | 44.91% | 44.91% | 44.91% | 5.09 pp | -17 | 12 | -1.42 |
| Consolidated Market Hours | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 4 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 195 | 85 | 110 | 43.59% | 43.59% | 43.59% | 6.41 pp | -25 | 15 | -1.67 |
| BTC Market Hours | lstm | LSTM | 195 | 84 | 111 | 43.08% | 43.08% | 43.08% | 6.92 pp | -27 | 15 | -1.80 |
| Consolidated Hourly | transformer | Transformer | 167 | 72 | 95 | 43.11% | 43.11% | 43.11% | 6.89 pp | -23 | 12 | -1.92 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 167 | 72 | 95 | 43.11% | 43.11% | 43.11% | 6.89 pp | -23 | 12 | -1.92 |
| BTC Daily | nn | NN | 197 | 89 | 108 | 45.18% | 45.18% | 45.18% | 4.82 pp | -19 | 9 | -2.11 |
| BTC Market Hours Daily | lstm | LSTM | 195 | 80 | 115 | 41.03% | 41.03% | 41.03% | 8.97 pp | -35 | 16 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 195 | 80 | 115 | 41.03% | 41.03% | 41.03% | 8.97 pp | -35 | 16 | -2.19 |
| Consolidated Market Hours Daily | transformer | Transformer | 47 | 18 | 29 | 38.30% | 38.30% | 38.30% | 11.70 pp | -11 | 4 | -2.75 |
| BTC Hourly | rf | RandomForest | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 8 | -2.88 |
| Consolidated Market Hours | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | transformer | Transformer | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| BTC Hourly | nn | NN | 171 | 73 | 98 | 42.69% | 42.69% | 42.69% | 7.31 pp | -25 | 8 | -3.12 |
| BTC Daily | transformer | Transformer | 197 | 84 | 113 | 42.64% | 42.64% | 42.64% | 7.36 pp | -29 | 9 | -3.22 |
| Consolidated Market Hours Daily | nn | NN | 47 | 17 | 30 | 36.17% | 36.17% | 36.17% | 13.83 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 47 | 15 | 32 | 31.91% | 31.91% | 31.91% | 18.09 pp | -17 | 4 | -4.25 |
| BTC Daily | rf | RandomForest | 197 | 76 | 121 | 38.58% | 38.58% | 38.58% | 11.42 pp | -45 | 9 | -5.00 |
| BTC Hourly | lstm | LSTM | 171 | 63 | 108 | 36.84% | 36.84% | 36.84% | 13.16 pp | -45 | 8 | -5.62 |
| BTC Daily | xgb | XGBoost | 207 | 74 | 133 | 35.75% | 35.75% | 35.75% | 14.25 pp | -59 | 10 | -5.90 |
| BTC Hourly | xgb | XGBoost | 171 | 61 | 110 | 35.67% | 35.67% | 35.67% | 14.33 pp | -49 | 8 | -6.12 |
| BTC Daily | lstm | LSTM | 197 | 67 | 130 | 34.01% | 34.01% | 34.01% | 15.99 pp | -63 | 9 | -7.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 171 | 87 | 84 | 50.88% | 50.88% | 50.88% | 0.88 pp | 3 | 8 | 0.38 |
| BTC Hourly | transformer | Transformer | 171 | 81 | 90 | 47.37% | 47.37% | 47.37% | 2.63 pp | -9 | 8 | -1.12 |
| BTC Hourly | rf | RandomForest | 171 | 74 | 97 | 43.27% | 43.27% | 43.27% | 6.73 pp | -23 | 8 | -2.88 |
| BTC Hourly | nn | NN | 171 | 73 | 98 | 42.69% | 42.69% | 42.69% | 7.31 pp | -25 | 8 | -3.12 |
| BTC Hourly | lstm | LSTM | 171 | 63 | 108 | 36.84% | 36.84% | 36.84% | 13.16 pp | -45 | 8 | -5.62 |
| BTC Hourly | xgb | XGBoost | 171 | 61 | 110 | 35.67% | 35.67% | 35.67% | 14.33 pp | -49 | 8 | -6.12 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 197 | 95 | 102 | 48.22% | 48.22% | 48.22% | 1.78 pp | -7 | 9 | -0.78 |
| BTC Daily | nn | NN | 197 | 89 | 108 | 45.18% | 45.18% | 45.18% | 4.82 pp | -19 | 9 | -2.11 |
| BTC Daily | transformer | Transformer | 197 | 84 | 113 | 42.64% | 42.64% | 42.64% | 7.36 pp | -29 | 9 | -3.22 |
| BTC Daily | rf | RandomForest | 197 | 76 | 121 | 38.58% | 38.58% | 38.58% | 11.42 pp | -45 | 9 | -5.00 |
| BTC Daily | xgb | XGBoost | 207 | 74 | 133 | 35.75% | 35.75% | 35.75% | 14.25 pp | -59 | 10 | -5.90 |
| BTC Daily | lstm | LSTM | 197 | 67 | 130 | 34.01% | 34.01% | 34.01% | 15.99 pp | -63 | 9 | -7.00 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | nn | NN | 195 | 100 | 95 | 51.28% | 51.28% | 51.28% | 1.28 pp | 5 | 15 | 0.33 |
| BTC Market Hours | transformer | Transformer | 195 | 97 | 98 | 49.74% | 49.74% | 49.74% | 0.26 pp | -1 | 15 | -0.07 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 195 | 91 | 104 | 46.67% | 46.67% | 46.67% | 3.33 pp | -13 | 15 | -0.87 |
| BTC Market Hours | rf | RandomForest | 195 | 90 | 105 | 46.15% | 46.15% | 46.15% | 3.85 pp | -15 | 15 | -1.00 |
| BTC Market Hours | xgb | XGBoost | 195 | 85 | 110 | 43.59% | 43.59% | 43.59% | 6.41 pp | -25 | 15 | -1.67 |
| BTC Market Hours | lstm | LSTM | 195 | 84 | 111 | 43.08% | 43.08% | 43.08% | 6.92 pp | -27 | 15 | -1.80 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | transformer | Transformer | 195 | 102 | 93 | 52.31% | 52.31% | 52.31% | 2.31 pp | 9 | 16 | 0.56 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 195 | 94 | 101 | 48.21% | 48.21% | 48.21% | 1.79 pp | -7 | 16 | -0.44 |
| BTC Market Hours Daily | nn | NN | 195 | 94 | 101 | 48.21% | 48.21% | 48.21% | 1.79 pp | -7 | 16 | -0.44 |
| BTC Market Hours Daily | rf | RandomForest | 195 | 87 | 108 | 44.62% | 44.62% | 44.62% | 5.38 pp | -21 | 16 | -1.31 |
| BTC Market Hours Daily | lstm | LSTM | 195 | 80 | 115 | 41.03% | 41.03% | 41.03% | 8.97 pp | -35 | 16 | -2.19 |
| BTC Market Hours Daily | xgb | XGBoost | 195 | 80 | 115 | 41.03% | 41.03% | 41.03% | 8.97 pp | -35 | 16 | -2.19 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 167 | 82 | 85 | 49.10% | 49.10% | 49.10% | 0.90 pp | -3 | 12 | -0.25 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| Consolidated Hourly | xgb | XGBoost | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| Consolidated Hourly | lstm | LSTM | 167 | 75 | 92 | 44.91% | 44.91% | 44.91% | 5.09 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | nn | NN | 167 | 75 | 92 | 44.91% | 44.91% | 44.91% | 5.09 pp | -17 | 12 | -1.42 |
| Consolidated Hourly | transformer | Transformer | 167 | 72 | 95 | 43.11% | 43.11% | 43.11% | 6.89 pp | -23 | 12 | -1.92 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 167 | 82 | 85 | 49.10% | 49.10% | 49.10% | 0.90 pp | -3 | 12 | -0.25 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 167 | 81 | 86 | 48.50% | 48.50% | 48.50% | 1.50 pp | -5 | 12 | -0.42 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 167 | 77 | 90 | 46.11% | 46.11% | 46.11% | 3.89 pp | -13 | 12 | -1.08 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 167 | 75 | 92 | 44.91% | 44.91% | 44.91% | 5.09 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | nn | NN | 167 | 75 | 92 | 44.91% | 44.91% | 44.91% | 5.09 pp | -17 | 12 | -1.42 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 167 | 72 | 95 | 43.11% | 43.11% | 43.11% | 6.89 pp | -23 | 12 | -1.92 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | xgb | XGBoost | 46 | 22 | 24 | 47.83% | 47.83% | 47.83% | 2.17 pp | -2 | 4 | -0.50 |
| Consolidated Market Hours | lstm | LSTM | 46 | 21 | 25 | 45.65% | 45.65% | 45.65% | 4.35 pp | -4 | 4 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 46 | 20 | 26 | 43.48% | 43.48% | 43.48% | 6.52 pp | -6 | 4 | -1.50 |
| Consolidated Market Hours | nn | NN | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | transformer | Transformer | 46 | 17 | 29 | 36.96% | 36.96% | 36.96% | 13.04 pp | -12 | 4 | -3.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 46 | 15 | 31 | 32.61% | 32.61% | 32.61% | 17.39 pp | -16 | 4 | -4.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | xgb | XGBoost | 47 | 23 | 24 | 48.94% | 48.94% | 48.94% | 1.06 pp | -1 | 4 | -0.25 |
| Consolidated Market Hours Daily | lstm | LSTM | 47 | 22 | 25 | 46.81% | 46.81% | 46.81% | 3.19 pp | -3 | 4 | -0.75 |
| Consolidated Market Hours Daily | rf | RandomForest | 47 | 21 | 26 | 44.68% | 44.68% | 44.68% | 5.32 pp | -5 | 4 | -1.25 |
| Consolidated Market Hours Daily | transformer | Transformer | 47 | 18 | 29 | 38.30% | 38.30% | 38.30% | 11.70 pp | -11 | 4 | -2.75 |
| Consolidated Market Hours Daily | nn | NN | 47 | 17 | 30 | 36.17% | 36.17% | 36.17% | 13.83 pp | -13 | 4 | -3.25 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 47 | 15 | 32 | 31.91% | 31.91% | 31.91% | 18.09 pp | -17 | 4 | -4.25 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
