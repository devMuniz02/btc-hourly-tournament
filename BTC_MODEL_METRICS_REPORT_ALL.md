# BTC Model Metrics Report - All Rows

Generated at: 2026-08-28T03:47:26.583799+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 813 | 306 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 976 | 611 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 553 | 373 | 179 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-28 00:00:00+00:00 | 555 | 427 | 126 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 08:00:00+00:00 | 35 | 35 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 08:00:00+00:00 | 35 | 35 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 08:00:00+00:00 | 35 | 0 | 35 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-21 08:00:00+00:00 | 35 | 0 | 35 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 35 | 20 | 15 | 57.14% | 57.14% | 57.14% | 7.14 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 35 | 20 | 15 | 57.14% | 57.14% | 57.14% | 7.14 pp | 5 | 4 | 1.25 |
| Consolidated Hourly | lstm | LSTM | 35 | 19 | 16 | 54.29% | 54.29% | 54.29% | 4.29 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 35 | 19 | 16 | 54.29% | 54.29% | 54.29% | 4.29 pp | 3 | 4 | 0.75 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 373 | 183 | 190 | 49.06% | 48.33% | 49.06% | 0.94 pp | -7 | 38 | -0.18 |
| BTC Daily | transformer | Transformer | 601 | 296 | 305 | 49.25% | 50.42% | 50.00% | 0.75 pp | -9 | 38 | -0.24 |
| BTC Daily | mlp_sklearn | MLPClassifier | 601 | 294 | 307 | 48.92% | 47.92% | 49.79% | 1.08 pp | -13 | 38 | -0.34 |
| BTC Market Hours | transformer | Transformer | 373 | 175 | 198 | 46.92% | 44.58% | 46.92% | 3.08 pp | -23 | 38 | -0.61 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 4 | -0.75 |
| BTC Market Hours | nn | NN | 373 | 172 | 201 | 46.11% | 48.75% | 46.11% | 3.89 pp | -29 | 38 | -0.76 |
| BTC Daily | nn | NN | 601 | 282 | 319 | 46.92% | 44.58% | 48.12% | 3.08 pp | -37 | 38 | -0.97 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 427 | 195 | 232 | 45.67% | 45.83% | 45.67% | 4.33 pp | -37 | 38 | -0.97 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 779 | 368 | 411 | 47.24% | 44.58% | 47.50% | 2.76 pp | -43 | 42 | -1.02 |
| BTC Market Hours Daily | nn | NN | 427 | 194 | 233 | 45.43% | 46.67% | 45.43% | 4.57 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | transformer | Transformer | 427 | 194 | 233 | 45.43% | 47.92% | 45.43% | 4.57 pp | -39 | 38 | -1.03 |
| BTC Hourly | transformer | Transformer | 779 | 365 | 414 | 46.85% | 42.92% | 45.83% | 3.15 pp | -49 | 42 | -1.17 |
| Consolidated Hourly | transformer | Transformer | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 4 | -1.25 |
| BTC Market Hours | lstm | LSTM | 373 | 159 | 214 | 42.63% | 42.92% | 42.63% | 7.37 pp | -55 | 38 | -1.45 |
| BTC Market Hours | rf | RandomForest | 373 | 158 | 215 | 42.36% | 40.42% | 42.36% | 7.64 pp | -57 | 38 | -1.50 |
| BTC Daily | lstm | LSTM | 601 | 269 | 332 | 44.76% | 43.33% | 45.00% | 5.24 pp | -63 | 38 | -1.66 |
| Consolidated Hourly | xgb | XGBoost | 35 | 14 | 21 | 40.00% | 40.00% | 40.00% | 10.00 pp | -7 | 4 | -1.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 35 | 14 | 21 | 40.00% | 40.00% | 40.00% | 10.00 pp | -7 | 4 | -1.75 |
| BTC Hourly | nn | NN | 779 | 350 | 429 | 44.93% | 40.83% | 45.83% | 5.07 pp | -79 | 42 | -1.88 |
| BTC Hourly | rf | RandomForest | 779 | 349 | 430 | 44.80% | 43.75% | 44.38% | 5.20 pp | -81 | 42 | -1.93 |
| BTC Market Hours | xgb | XGBoost | 373 | 149 | 224 | 39.95% | 39.58% | 39.95% | 10.05 pp | -75 | 38 | -1.97 |
| BTC Hourly | lstm | LSTM | 779 | 345 | 434 | 44.29% | 43.75% | 45.83% | 5.71 pp | -89 | 42 | -2.12 |
| BTC Daily | rf | RandomForest | 601 | 260 | 341 | 43.26% | 44.58% | 43.96% | 6.74 pp | -81 | 38 | -2.13 |
| BTC Market Hours Daily | rf | RandomForest | 427 | 173 | 254 | 40.52% | 40.42% | 40.52% | 9.48 pp | -81 | 38 | -2.13 |
| BTC Market Hours Daily | lstm | LSTM | 427 | 168 | 259 | 39.34% | 38.33% | 39.34% | 10.66 pp | -91 | 38 | -2.39 |
| BTC Market Hours Daily | xgb | XGBoost | 427 | 168 | 259 | 39.34% | 38.33% | 39.34% | 10.66 pp | -91 | 38 | -2.39 |
| BTC Hourly | xgb | XGBoost | 779 | 334 | 445 | 42.88% | 40.42% | 43.96% | 7.12 pp | -111 | 42 | -2.64 |
| BTC Daily | xgb | XGBoost | 611 | 245 | 366 | 40.10% | 35.42% | 40.21% | 9.90 pp | -121 | 38 | -3.18 |
| Consolidated Hourly | nn | NN | 35 | 11 | 24 | 31.43% | 31.43% | 31.43% | 18.57 pp | -13 | 4 | -3.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 35 | 11 | 24 | 31.43% | 31.43% | 31.43% | 18.57 pp | -13 | 4 | -3.25 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 779 | 368 | 411 | 47.24% | 44.58% | 47.50% | 2.76 pp | -43 | 42 | -1.02 |
| BTC Hourly | transformer | Transformer | 779 | 365 | 414 | 46.85% | 42.92% | 45.83% | 3.15 pp | -49 | 42 | -1.17 |
| BTC Hourly | nn | NN | 779 | 350 | 429 | 44.93% | 40.83% | 45.83% | 5.07 pp | -79 | 42 | -1.88 |
| BTC Hourly | rf | RandomForest | 779 | 349 | 430 | 44.80% | 43.75% | 44.38% | 5.20 pp | -81 | 42 | -1.93 |
| BTC Hourly | lstm | LSTM | 779 | 345 | 434 | 44.29% | 43.75% | 45.83% | 5.71 pp | -89 | 42 | -2.12 |
| BTC Hourly | xgb | XGBoost | 779 | 334 | 445 | 42.88% | 40.42% | 43.96% | 7.12 pp | -111 | 42 | -2.64 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 601 | 296 | 305 | 49.25% | 50.42% | 50.00% | 0.75 pp | -9 | 38 | -0.24 |
| BTC Daily | mlp_sklearn | MLPClassifier | 601 | 294 | 307 | 48.92% | 47.92% | 49.79% | 1.08 pp | -13 | 38 | -0.34 |
| BTC Daily | nn | NN | 601 | 282 | 319 | 46.92% | 44.58% | 48.12% | 3.08 pp | -37 | 38 | -0.97 |
| BTC Daily | lstm | LSTM | 601 | 269 | 332 | 44.76% | 43.33% | 45.00% | 5.24 pp | -63 | 38 | -1.66 |
| BTC Daily | rf | RandomForest | 601 | 260 | 341 | 43.26% | 44.58% | 43.96% | 6.74 pp | -81 | 38 | -2.13 |
| BTC Daily | xgb | XGBoost | 611 | 245 | 366 | 40.10% | 35.42% | 40.21% | 9.90 pp | -121 | 38 | -3.18 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 373 | 183 | 190 | 49.06% | 48.33% | 49.06% | 0.94 pp | -7 | 38 | -0.18 |
| BTC Market Hours | transformer | Transformer | 373 | 175 | 198 | 46.92% | 44.58% | 46.92% | 3.08 pp | -23 | 38 | -0.61 |
| BTC Market Hours | nn | NN | 373 | 172 | 201 | 46.11% | 48.75% | 46.11% | 3.89 pp | -29 | 38 | -0.76 |
| BTC Market Hours | lstm | LSTM | 373 | 159 | 214 | 42.63% | 42.92% | 42.63% | 7.37 pp | -55 | 38 | -1.45 |
| BTC Market Hours | rf | RandomForest | 373 | 158 | 215 | 42.36% | 40.42% | 42.36% | 7.64 pp | -57 | 38 | -1.50 |
| BTC Market Hours | xgb | XGBoost | 373 | 149 | 224 | 39.95% | 39.58% | 39.95% | 10.05 pp | -75 | 38 | -1.97 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 427 | 195 | 232 | 45.67% | 45.83% | 45.67% | 4.33 pp | -37 | 38 | -0.97 |
| BTC Market Hours Daily | nn | NN | 427 | 194 | 233 | 45.43% | 46.67% | 45.43% | 4.57 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | transformer | Transformer | 427 | 194 | 233 | 45.43% | 47.92% | 45.43% | 4.57 pp | -39 | 38 | -1.03 |
| BTC Market Hours Daily | rf | RandomForest | 427 | 173 | 254 | 40.52% | 40.42% | 40.52% | 9.48 pp | -81 | 38 | -2.13 |
| BTC Market Hours Daily | lstm | LSTM | 427 | 168 | 259 | 39.34% | 38.33% | 39.34% | 10.66 pp | -91 | 38 | -2.39 |
| BTC Market Hours Daily | xgb | XGBoost | 427 | 168 | 259 | 39.34% | 38.33% | 39.34% | 10.66 pp | -91 | 38 | -2.39 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 35 | 20 | 15 | 57.14% | 57.14% | 57.14% | 7.14 pp | 5 | 4 | 1.25 |
| Consolidated Hourly | lstm | LSTM | 35 | 19 | 16 | 54.29% | 54.29% | 54.29% | 4.29 pp | 3 | 4 | 0.75 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 4 | -0.75 |
| Consolidated Hourly | transformer | Transformer | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 4 | -1.25 |
| Consolidated Hourly | xgb | XGBoost | 35 | 14 | 21 | 40.00% | 40.00% | 40.00% | 10.00 pp | -7 | 4 | -1.75 |
| Consolidated Hourly | nn | NN | 35 | 11 | 24 | 31.43% | 31.43% | 31.43% | 18.57 pp | -13 | 4 | -3.25 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 35 | 20 | 15 | 57.14% | 57.14% | 57.14% | 7.14 pp | 5 | 4 | 1.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 35 | 19 | 16 | 54.29% | 54.29% | 54.29% | 4.29 pp | 3 | 4 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 35 | 16 | 19 | 45.71% | 45.71% | 45.71% | 4.29 pp | -3 | 4 | -0.75 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 35 | 15 | 20 | 42.86% | 42.86% | 42.86% | 7.14 pp | -5 | 4 | -1.25 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 35 | 14 | 21 | 40.00% | 40.00% | 40.00% | 10.00 pp | -7 | 4 | -1.75 |
| Consolidated Daily/Hourly Refresh | nn | NN | 35 | 11 | 24 | 31.43% | 31.43% | 31.43% | 18.57 pp | -13 | 4 | -3.25 |

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
