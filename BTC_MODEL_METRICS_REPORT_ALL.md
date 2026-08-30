# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T02:47:22.297609+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1137 | 849 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1013 | 648 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 616 | 410 | 205 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 618 | 464 | 152 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 17:00:00+00:00 | 67 | 67 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 17:00:00+00:00 | 67 | 67 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 17:00:00+00:00 | 67 | 1 | 66 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 17:00:00+00:00 | 67 | 1 | 66 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 67 | 38 | 29 | 56.72% | 56.72% | 56.72% | 6.72 pp | 9 | 7 | 1.29 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 67 | 38 | 29 | 56.72% | 56.72% | 56.72% | 6.72 pp | 9 | 7 | 1.29 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 67 | 35 | 32 | 52.24% | 52.24% | 52.24% | 2.24 pp | 3 | 7 | 0.43 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 67 | 35 | 32 | 52.24% | 52.24% | 52.24% | 2.24 pp | 3 | 7 | 0.43 |
| Consolidated Hourly | lstm | LSTM | 67 | 34 | 33 | 50.75% | 50.75% | 50.75% | 0.75 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 67 | 34 | 33 | 50.75% | 50.75% | 50.75% | 0.75 pp | 1 | 7 | 0.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 410 | 204 | 206 | 49.76% | 49.17% | 49.76% | 0.24 pp | -2 | 41 | -0.05 |
| BTC Daily | transformer | Transformer | 638 | 312 | 326 | 48.90% | 47.08% | 49.79% | 1.10 pp | -14 | 39 | -0.36 |
| BTC Daily | mlp_sklearn | MLPClassifier | 638 | 309 | 329 | 48.43% | 45.00% | 49.79% | 1.57 pp | -20 | 39 | -0.51 |
| BTC Market Hours | nn | NN | 410 | 194 | 216 | 47.32% | 50.42% | 47.32% | 2.68 pp | -22 | 41 | -0.54 |
| BTC Market Hours | transformer | Transformer | 410 | 190 | 220 | 46.34% | 42.50% | 46.34% | 3.66 pp | -30 | 41 | -0.73 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 464 | 215 | 249 | 46.34% | 46.25% | 46.34% | 3.66 pp | -34 | 41 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 464 | 213 | 251 | 45.91% | 46.25% | 45.91% | 4.09 pp | -38 | 41 | -0.93 |
| BTC Daily | nn | NN | 638 | 300 | 338 | 47.02% | 42.92% | 48.96% | 2.98 pp | -38 | 39 | -0.97 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | transformer | Transformer | 815 | 385 | 430 | 47.24% | 45.83% | 46.25% | 2.76 pp | -45 | 44 | -1.02 |
| BTC Market Hours Daily | nn | NN | 464 | 211 | 253 | 45.47% | 45.00% | 45.47% | 4.53 pp | -42 | 41 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 815 | 382 | 433 | 46.87% | 42.92% | 46.88% | 3.13 pp | -51 | 44 | -1.16 |
| BTC Market Hours | lstm | LSTM | 410 | 181 | 229 | 44.15% | 45.42% | 44.15% | 5.85 pp | -48 | 41 | -1.17 |
| Consolidated Hourly | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | xgb | XGBoost | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 7 | -1.29 |
| BTC Market Hours | rf | RandomForest | 410 | 176 | 234 | 42.93% | 42.08% | 42.93% | 7.07 pp | -58 | 41 | -1.41 |
| BTC Hourly | nn | NN | 815 | 367 | 448 | 45.03% | 40.83% | 44.79% | 4.97 pp | -81 | 44 | -1.84 |
| BTC Daily | lstm | LSTM | 638 | 283 | 355 | 44.36% | 42.08% | 43.96% | 5.64 pp | -72 | 39 | -1.85 |
| BTC Hourly | rf | RandomForest | 815 | 365 | 450 | 44.79% | 44.58% | 44.58% | 5.21 pp | -85 | 44 | -1.93 |
| BTC Market Hours | xgb | XGBoost | 410 | 164 | 246 | 40.00% | 38.33% | 40.00% | 10.00 pp | -82 | 41 | -2.00 |
| BTC Market Hours Daily | rf | RandomForest | 464 | 190 | 274 | 40.95% | 41.25% | 40.95% | 9.05 pp | -84 | 41 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 464 | 189 | 275 | 40.73% | 39.58% | 40.73% | 9.27 pp | -86 | 41 | -2.10 |
| BTC Hourly | lstm | LSTM | 815 | 355 | 460 | 43.56% | 41.67% | 44.17% | 6.44 pp | -105 | 44 | -2.39 |
| BTC Daily | rf | RandomForest | 638 | 272 | 366 | 42.63% | 41.25% | 43.54% | 7.37 pp | -94 | 39 | -2.41 |
| BTC Market Hours Daily | xgb | XGBoost | 464 | 181 | 283 | 39.01% | 35.83% | 39.01% | 10.99 pp | -102 | 41 | -2.49 |
| Consolidated Hourly | nn | NN | 67 | 24 | 43 | 35.82% | 35.82% | 35.82% | 14.18 pp | -19 | 7 | -2.71 |
| Consolidated Daily/Hourly Refresh | nn | NN | 67 | 24 | 43 | 35.82% | 35.82% | 35.82% | 14.18 pp | -19 | 7 | -2.71 |
| BTC Hourly | xgb | XGBoost | 815 | 345 | 470 | 42.33% | 39.58% | 42.71% | 7.67 pp | -125 | 44 | -2.84 |
| BTC Daily | xgb | XGBoost | 648 | 253 | 395 | 39.04% | 30.42% | 38.96% | 10.96 pp | -142 | 39 | -3.64 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 815 | 385 | 430 | 47.24% | 45.83% | 46.25% | 2.76 pp | -45 | 44 | -1.02 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 815 | 382 | 433 | 46.87% | 42.92% | 46.88% | 3.13 pp | -51 | 44 | -1.16 |
| BTC Hourly | nn | NN | 815 | 367 | 448 | 45.03% | 40.83% | 44.79% | 4.97 pp | -81 | 44 | -1.84 |
| BTC Hourly | rf | RandomForest | 815 | 365 | 450 | 44.79% | 44.58% | 44.58% | 5.21 pp | -85 | 44 | -1.93 |
| BTC Hourly | lstm | LSTM | 815 | 355 | 460 | 43.56% | 41.67% | 44.17% | 6.44 pp | -105 | 44 | -2.39 |
| BTC Hourly | xgb | XGBoost | 815 | 345 | 470 | 42.33% | 39.58% | 42.71% | 7.67 pp | -125 | 44 | -2.84 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 638 | 312 | 326 | 48.90% | 47.08% | 49.79% | 1.10 pp | -14 | 39 | -0.36 |
| BTC Daily | mlp_sklearn | MLPClassifier | 638 | 309 | 329 | 48.43% | 45.00% | 49.79% | 1.57 pp | -20 | 39 | -0.51 |
| BTC Daily | nn | NN | 638 | 300 | 338 | 47.02% | 42.92% | 48.96% | 2.98 pp | -38 | 39 | -0.97 |
| BTC Daily | lstm | LSTM | 638 | 283 | 355 | 44.36% | 42.08% | 43.96% | 5.64 pp | -72 | 39 | -1.85 |
| BTC Daily | rf | RandomForest | 638 | 272 | 366 | 42.63% | 41.25% | 43.54% | 7.37 pp | -94 | 39 | -2.41 |
| BTC Daily | xgb | XGBoost | 648 | 253 | 395 | 39.04% | 30.42% | 38.96% | 10.96 pp | -142 | 39 | -3.64 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 410 | 204 | 206 | 49.76% | 49.17% | 49.76% | 0.24 pp | -2 | 41 | -0.05 |
| BTC Market Hours | nn | NN | 410 | 194 | 216 | 47.32% | 50.42% | 47.32% | 2.68 pp | -22 | 41 | -0.54 |
| BTC Market Hours | transformer | Transformer | 410 | 190 | 220 | 46.34% | 42.50% | 46.34% | 3.66 pp | -30 | 41 | -0.73 |
| BTC Market Hours | lstm | LSTM | 410 | 181 | 229 | 44.15% | 45.42% | 44.15% | 5.85 pp | -48 | 41 | -1.17 |
| BTC Market Hours | rf | RandomForest | 410 | 176 | 234 | 42.93% | 42.08% | 42.93% | 7.07 pp | -58 | 41 | -1.41 |
| BTC Market Hours | xgb | XGBoost | 410 | 164 | 246 | 40.00% | 38.33% | 40.00% | 10.00 pp | -82 | 41 | -2.00 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 464 | 215 | 249 | 46.34% | 46.25% | 46.34% | 3.66 pp | -34 | 41 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 464 | 213 | 251 | 45.91% | 46.25% | 45.91% | 4.09 pp | -38 | 41 | -0.93 |
| BTC Market Hours Daily | nn | NN | 464 | 211 | 253 | 45.47% | 45.00% | 45.47% | 4.53 pp | -42 | 41 | -1.02 |
| BTC Market Hours Daily | rf | RandomForest | 464 | 190 | 274 | 40.95% | 41.25% | 40.95% | 9.05 pp | -84 | 41 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 464 | 189 | 275 | 40.73% | 39.58% | 40.73% | 9.27 pp | -86 | 41 | -2.10 |
| BTC Market Hours Daily | xgb | XGBoost | 464 | 181 | 283 | 39.01% | 35.83% | 39.01% | 10.99 pp | -102 | 41 | -2.49 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 67 | 38 | 29 | 56.72% | 56.72% | 56.72% | 6.72 pp | 9 | 7 | 1.29 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 67 | 35 | 32 | 52.24% | 52.24% | 52.24% | 2.24 pp | 3 | 7 | 0.43 |
| Consolidated Hourly | lstm | LSTM | 67 | 34 | 33 | 50.75% | 50.75% | 50.75% | 0.75 pp | 1 | 7 | 0.14 |
| Consolidated Hourly | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | xgb | XGBoost | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | nn | NN | 67 | 24 | 43 | 35.82% | 35.82% | 35.82% | 14.18 pp | -19 | 7 | -2.71 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 67 | 38 | 29 | 56.72% | 56.72% | 56.72% | 6.72 pp | 9 | 7 | 1.29 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 67 | 35 | 32 | 52.24% | 52.24% | 52.24% | 2.24 pp | 3 | 7 | 0.43 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 67 | 34 | 33 | 50.75% | 50.75% | 50.75% | 0.75 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 67 | 29 | 38 | 43.28% | 43.28% | 43.28% | 6.72 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | nn | NN | 67 | 24 | 43 | 35.82% | 35.82% | 35.82% | 14.18 pp | -19 | 7 | -2.71 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
