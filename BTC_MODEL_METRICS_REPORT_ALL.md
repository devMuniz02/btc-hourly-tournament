# BTC Model Metrics Report - All Rows

Generated at: 2026-08-31T07:20:17.469470+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1159 | 871 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1035 | 670 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 651 | 432 | 218 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-31 00:00:00+00:00 | 653 | 486 | 165 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 00:00:00+00:00 | 83 | 83 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 00:00:00+00:00 | 83 | 83 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 00:00:00+00:00 | 83 | 1 | 82 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-26 00:00:00+00:00 | 83 | 1 | 82 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Hourly | rf | RandomForest | 83 | 46 | 37 | 55.42% | 55.42% | 55.42% | 5.42 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 83 | 46 | 37 | 55.42% | 55.42% | 55.42% | 5.42 pp | 9 | 9 | 1.00 |
| BTC Daily | mlp_sklearn | MLPClassifier | 660 | 344 | 316 | 52.12% | 50.00% | 53.33% | 2.12 pp | 28 | 40 | 0.70 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | lstm | LSTM | 83 | 43 | 40 | 51.81% | 51.81% | 51.81% | 1.81 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 83 | 43 | 40 | 51.81% | 51.81% | 51.81% | 1.81 pp | 3 | 9 | 0.33 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 432 | 219 | 213 | 50.69% | 47.08% | 50.69% | 0.69 pp | 6 | 43 | 0.14 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 837 | 415 | 422 | 49.58% | 48.33% | 49.58% | 0.42 pp | -7 | 45 | -0.16 |
| BTC Market Hours | nn | NN | 432 | 212 | 220 | 49.07% | 50.00% | 49.07% | 0.93 pp | -8 | 43 | -0.19 |
| BTC Daily | nn | NN | 660 | 319 | 341 | 48.33% | 47.08% | 50.42% | 1.67 pp | -22 | 40 | -0.55 |
| Consolidated Hourly | xgb | XGBoost | 83 | 39 | 44 | 46.99% | 46.99% | 46.99% | 3.01 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 83 | 39 | 44 | 46.99% | 46.99% | 46.99% | 3.01 pp | -5 | 9 | -0.56 |
| BTC Market Hours Daily | nn | NN | 486 | 228 | 258 | 46.91% | 44.58% | 47.29% | 3.09 pp | -30 | 43 | -0.70 |
| Consolidated Hourly | transformer | Transformer | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 9 | -0.78 |
| BTC Daily | transformer | Transformer | 660 | 312 | 348 | 47.27% | 44.17% | 49.17% | 2.73 pp | -36 | 40 | -0.90 |
| BTC Hourly | nn | NN | 837 | 398 | 439 | 47.55% | 47.50% | 47.08% | 2.45 pp | -41 | 45 | -0.91 |
| BTC Hourly | transformer | Transformer | 837 | 398 | 439 | 47.55% | 47.08% | 45.42% | 2.45 pp | -41 | 45 | -0.91 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 486 | 223 | 263 | 45.88% | 47.08% | 46.04% | 4.12 pp | -40 | 43 | -0.93 |
| BTC Market Hours Daily | transformer | Transformer | 486 | 223 | 263 | 45.88% | 43.33% | 46.04% | 4.12 pp | -40 | 43 | -0.93 |
| BTC Market Hours | transformer | Transformer | 432 | 196 | 236 | 45.37% | 40.83% | 45.37% | 4.63 pp | -40 | 43 | -0.93 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |
| BTC Hourly | rf | RandomForest | 837 | 393 | 444 | 46.95% | 46.25% | 46.04% | 3.05 pp | -51 | 45 | -1.13 |
| BTC Market Hours | rf | RandomForest | 432 | 189 | 243 | 43.75% | 44.17% | 43.75% | 6.25 pp | -54 | 43 | -1.26 |
| BTC Market Hours Daily | rf | RandomForest | 486 | 214 | 272 | 44.03% | 43.33% | 44.17% | 5.97 pp | -58 | 43 | -1.35 |
| BTC Daily | lstm | LSTM | 660 | 303 | 357 | 45.91% | 40.00% | 45.83% | 4.09 pp | -54 | 40 | -1.35 |
| BTC Market Hours | lstm | LSTM | 432 | 185 | 247 | 42.82% | 40.83% | 42.82% | 7.18 pp | -62 | 43 | -1.44 |
| BTC Daily | rf | RandomForest | 660 | 298 | 362 | 45.15% | 42.50% | 46.25% | 4.85 pp | -64 | 40 | -1.60 |
| Consolidated Hourly | nn | NN | 83 | 34 | 49 | 40.96% | 40.96% | 40.96% | 9.04 pp | -15 | 9 | -1.67 |
| Consolidated Daily/Hourly Refresh | nn | NN | 83 | 34 | 49 | 40.96% | 40.96% | 40.96% | 9.04 pp | -15 | 9 | -1.67 |
| BTC Market Hours Daily | xgb | XGBoost | 486 | 202 | 284 | 41.56% | 40.00% | 41.67% | 8.44 pp | -82 | 43 | -1.91 |
| BTC Market Hours | xgb | XGBoost | 432 | 174 | 258 | 40.28% | 38.33% | 40.28% | 9.72 pp | -84 | 43 | -1.95 |
| BTC Hourly | lstm | LSTM | 837 | 374 | 463 | 44.68% | 43.75% | 45.21% | 5.32 pp | -89 | 45 | -1.98 |
| BTC Hourly | xgb | XGBoost | 837 | 369 | 468 | 44.09% | 43.75% | 43.96% | 5.91 pp | -99 | 45 | -2.20 |
| BTC Market Hours Daily | lstm | LSTM | 486 | 194 | 292 | 39.92% | 36.25% | 40.00% | 10.08 pp | -98 | 43 | -2.28 |
| BTC Daily | xgb | XGBoost | 670 | 271 | 399 | 40.45% | 35.00% | 41.04% | 9.55 pp | -128 | 40 | -3.20 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 837 | 415 | 422 | 49.58% | 48.33% | 49.58% | 0.42 pp | -7 | 45 | -0.16 |
| BTC Hourly | nn | NN | 837 | 398 | 439 | 47.55% | 47.50% | 47.08% | 2.45 pp | -41 | 45 | -0.91 |
| BTC Hourly | transformer | Transformer | 837 | 398 | 439 | 47.55% | 47.08% | 45.42% | 2.45 pp | -41 | 45 | -0.91 |
| BTC Hourly | rf | RandomForest | 837 | 393 | 444 | 46.95% | 46.25% | 46.04% | 3.05 pp | -51 | 45 | -1.13 |
| BTC Hourly | lstm | LSTM | 837 | 374 | 463 | 44.68% | 43.75% | 45.21% | 5.32 pp | -89 | 45 | -1.98 |
| BTC Hourly | xgb | XGBoost | 837 | 369 | 468 | 44.09% | 43.75% | 43.96% | 5.91 pp | -99 | 45 | -2.20 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 660 | 344 | 316 | 52.12% | 50.00% | 53.33% | 2.12 pp | 28 | 40 | 0.70 |
| BTC Daily | nn | NN | 660 | 319 | 341 | 48.33% | 47.08% | 50.42% | 1.67 pp | -22 | 40 | -0.55 |
| BTC Daily | transformer | Transformer | 660 | 312 | 348 | 47.27% | 44.17% | 49.17% | 2.73 pp | -36 | 40 | -0.90 |
| BTC Daily | lstm | LSTM | 660 | 303 | 357 | 45.91% | 40.00% | 45.83% | 4.09 pp | -54 | 40 | -1.35 |
| BTC Daily | rf | RandomForest | 660 | 298 | 362 | 45.15% | 42.50% | 46.25% | 4.85 pp | -64 | 40 | -1.60 |
| BTC Daily | xgb | XGBoost | 670 | 271 | 399 | 40.45% | 35.00% | 41.04% | 9.55 pp | -128 | 40 | -3.20 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 432 | 219 | 213 | 50.69% | 47.08% | 50.69% | 0.69 pp | 6 | 43 | 0.14 |
| BTC Market Hours | nn | NN | 432 | 212 | 220 | 49.07% | 50.00% | 49.07% | 0.93 pp | -8 | 43 | -0.19 |
| BTC Market Hours | transformer | Transformer | 432 | 196 | 236 | 45.37% | 40.83% | 45.37% | 4.63 pp | -40 | 43 | -0.93 |
| BTC Market Hours | rf | RandomForest | 432 | 189 | 243 | 43.75% | 44.17% | 43.75% | 6.25 pp | -54 | 43 | -1.26 |
| BTC Market Hours | lstm | LSTM | 432 | 185 | 247 | 42.82% | 40.83% | 42.82% | 7.18 pp | -62 | 43 | -1.44 |
| BTC Market Hours | xgb | XGBoost | 432 | 174 | 258 | 40.28% | 38.33% | 40.28% | 9.72 pp | -84 | 43 | -1.95 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 486 | 228 | 258 | 46.91% | 44.58% | 47.29% | 3.09 pp | -30 | 43 | -0.70 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 486 | 223 | 263 | 45.88% | 47.08% | 46.04% | 4.12 pp | -40 | 43 | -0.93 |
| BTC Market Hours Daily | transformer | Transformer | 486 | 223 | 263 | 45.88% | 43.33% | 46.04% | 4.12 pp | -40 | 43 | -0.93 |
| BTC Market Hours Daily | rf | RandomForest | 486 | 214 | 272 | 44.03% | 43.33% | 44.17% | 5.97 pp | -58 | 43 | -1.35 |
| BTC Market Hours Daily | xgb | XGBoost | 486 | 202 | 284 | 41.56% | 40.00% | 41.67% | 8.44 pp | -82 | 43 | -1.91 |
| BTC Market Hours Daily | lstm | LSTM | 486 | 194 | 292 | 39.92% | 36.25% | 40.00% | 10.08 pp | -98 | 43 | -2.28 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 83 | 46 | 37 | 55.42% | 55.42% | 55.42% | 5.42 pp | 9 | 9 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 9 | 0.56 |
| Consolidated Hourly | lstm | LSTM | 83 | 43 | 40 | 51.81% | 51.81% | 51.81% | 1.81 pp | 3 | 9 | 0.33 |
| Consolidated Hourly | xgb | XGBoost | 83 | 39 | 44 | 46.99% | 46.99% | 46.99% | 3.01 pp | -5 | 9 | -0.56 |
| Consolidated Hourly | transformer | Transformer | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 9 | -0.78 |
| Consolidated Hourly | nn | NN | 83 | 34 | 49 | 40.96% | 40.96% | 40.96% | 9.04 pp | -15 | 9 | -1.67 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 83 | 46 | 37 | 55.42% | 55.42% | 55.42% | 5.42 pp | 9 | 9 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 83 | 44 | 39 | 53.01% | 53.01% | 53.01% | 3.01 pp | 5 | 9 | 0.56 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 83 | 43 | 40 | 51.81% | 51.81% | 51.81% | 1.81 pp | 3 | 9 | 0.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 83 | 39 | 44 | 46.99% | 46.99% | 46.99% | 3.01 pp | -5 | 9 | -0.56 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 83 | 38 | 45 | 45.78% | 45.78% | 45.78% | 4.22 pp | -7 | 9 | -0.78 |
| Consolidated Daily/Hourly Refresh | nn | NN | 83 | 34 | 49 | 40.96% | 40.96% | 40.96% | 9.04 pp | -15 | 9 | -1.67 |

### Consolidated Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

### Consolidated Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Market Hours Daily | lstm | LSTM | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | nn | NN | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | rf | RandomForest | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | transformer | Transformer | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | xgb | XGBoost | 1 | 1 | 0 | 100.00% | 100.00% | 100.00% | 50.00 pp | 1 | 1 | 1.00 |
| Consolidated Market Hours Daily | mlp_sklearn | MLPClassifier | 1 | 0 | 1 | 0.00% | 0.00% | 0.00% | 50.00 pp | -1 | 1 | -1.00 |

## Metric Definitions

- Accuracy is wins divided by evaluated predictions.
- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.
- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.
- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.
- Net wins is wins minus losses.
- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.
- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.
- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.
