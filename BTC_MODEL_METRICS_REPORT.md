# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T21:31:19.331544+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1152 | 864 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1028 | 663 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 20:00:00+00:00 | 640 | 425 | 214 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 20:00:00+00:00 | 642 | 479 | 161 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 07:00:00+00:00 | 78 | 78 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 07:00:00+00:00 | 78 | 78 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 07:00:00+00:00 | 78 | 0 | 78 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-25 07:00:00+00:00 | 78 | 0 | 78 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | rf | RandomForest | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 78 | 42 | 36 | 53.85% | 53.85% | 53.85% | 3.85 pp | 6 | 8 | 0.75 |
| Consolidated Hourly | lstm | LSTM | 78 | 40 | 38 | 51.28% | 51.28% | 51.28% | 1.28 pp | 2 | 8 | 0.25 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 78 | 40 | 38 | 51.28% | 51.28% | 51.28% | 1.28 pp | 2 | 8 | 0.25 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 425 | 210 | 215 | 49.41% | 47.08% | 49.41% | 0.59 pp | -5 | 42 | -0.12 |
| BTC Daily | mlp_sklearn | MLPClassifier | 653 | 317 | 336 | 48.55% | 45.83% | 49.58% | 1.45 pp | -19 | 40 | -0.47 |
| BTC Daily | transformer | Transformer | 653 | 317 | 336 | 48.55% | 46.25% | 49.79% | 1.45 pp | -19 | 40 | -0.47 |
| BTC Market Hours | nn | NN | 425 | 201 | 224 | 47.29% | 50.42% | 47.29% | 2.71 pp | -23 | 42 | -0.55 |
| Consolidated Hourly | xgb | XGBoost | 78 | 36 | 42 | 46.15% | 46.15% | 46.15% | 3.85 pp | -6 | 8 | -0.75 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 78 | 36 | 42 | 46.15% | 46.15% | 46.15% | 3.85 pp | -6 | 8 | -0.75 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 479 | 221 | 258 | 46.14% | 46.67% | 46.14% | 3.86 pp | -37 | 42 | -0.88 |
| BTC Market Hours | transformer | Transformer | 425 | 194 | 231 | 45.65% | 41.67% | 45.65% | 4.35 pp | -37 | 42 | -0.88 |
| BTC Hourly | transformer | Transformer | 830 | 394 | 436 | 47.47% | 47.50% | 46.67% | 2.53 pp | -42 | 45 | -0.93 |
| Consolidated Hourly | transformer | Transformer | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 8 | -1.00 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 78 | 35 | 43 | 44.87% | 44.87% | 44.87% | 5.13 pp | -8 | 8 | -1.00 |
| BTC Market Hours Daily | nn | NN | 479 | 218 | 261 | 45.51% | 44.17% | 45.51% | 4.49 pp | -43 | 42 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 479 | 217 | 262 | 45.30% | 45.00% | 45.30% | 4.70 pp | -45 | 42 | -1.07 |
| BTC Daily | nn | NN | 653 | 304 | 349 | 46.55% | 41.25% | 48.54% | 3.45 pp | -45 | 40 | -1.12 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 830 | 389 | 441 | 46.87% | 42.50% | 46.67% | 3.13 pp | -52 | 45 | -1.16 |
| BTC Market Hours | lstm | LSTM | 425 | 186 | 239 | 43.76% | 43.75% | 43.76% | 6.24 pp | -53 | 42 | -1.26 |
| BTC Market Hours | rf | RandomForest | 425 | 184 | 241 | 43.29% | 43.75% | 43.29% | 6.71 pp | -57 | 42 | -1.36 |
| BTC Hourly | nn | NN | 830 | 375 | 455 | 45.18% | 42.92% | 44.58% | 4.82 pp | -80 | 45 | -1.78 |
| BTC Daily | lstm | LSTM | 653 | 288 | 365 | 44.10% | 40.83% | 43.33% | 5.90 pp | -77 | 40 | -1.93 |
| BTC Hourly | rf | RandomForest | 830 | 371 | 459 | 44.70% | 43.33% | 44.38% | 5.30 pp | -88 | 45 | -1.96 |
| BTC Market Hours Daily | rf | RandomForest | 479 | 198 | 281 | 41.34% | 41.67% | 41.34% | 8.66 pp | -83 | 42 | -1.98 |
| BTC Market Hours | xgb | XGBoost | 425 | 169 | 256 | 39.76% | 37.92% | 39.76% | 10.24 pp | -87 | 42 | -2.07 |
| BTC Market Hours Daily | lstm | LSTM | 479 | 194 | 285 | 40.50% | 39.17% | 40.50% | 9.50 pp | -91 | 42 | -2.17 |
| Consolidated Hourly | nn | NN | 78 | 30 | 48 | 38.46% | 38.46% | 38.46% | 11.54 pp | -18 | 8 | -2.25 |
| Consolidated Daily/Hourly Refresh | nn | NN | 78 | 30 | 48 | 38.46% | 38.46% | 38.46% | 11.54 pp | -18 | 8 | -2.25 |
| BTC Daily | rf | RandomForest | 653 | 278 | 375 | 42.57% | 40.83% | 43.33% | 7.43 pp | -97 | 40 | -2.42 |
| BTC Hourly | lstm | LSTM | 830 | 359 | 471 | 43.25% | 40.42% | 43.12% | 6.75 pp | -112 | 45 | -2.49 |
| BTC Market Hours Daily | xgb | XGBoost | 479 | 185 | 294 | 38.62% | 35.00% | 38.62% | 11.38 pp | -109 | 42 | -2.60 |
| BTC Hourly | xgb | XGBoost | 830 | 352 | 478 | 42.41% | 39.17% | 42.71% | 7.59 pp | -126 | 45 | -2.80 |
| BTC Daily | xgb | XGBoost | 663 | 262 | 401 | 39.52% | 32.50% | 39.79% | 10.48 pp | -139 | 40 | -3.48 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 830 | 394 | 436 | 47.47% | 47.50% | 46.67% | 2.53 pp | -42 | 45 | -0.93 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 830 | 389 | 441 | 46.87% | 42.50% | 46.67% | 3.13 pp | -52 | 45 | -1.16 |
| BTC Hourly | nn | NN | 830 | 375 | 455 | 45.18% | 42.92% | 44.58% | 4.82 pp | -80 | 45 | -1.78 |
| BTC Hourly | rf | RandomForest | 830 | 371 | 459 | 44.70% | 43.33% | 44.38% | 5.30 pp | -88 | 45 | -1.96 |
| BTC Hourly | lstm | LSTM | 830 | 359 | 471 | 43.25% | 40.42% | 43.12% | 6.75 pp | -112 | 45 | -2.49 |
| BTC Hourly | xgb | XGBoost | 830 | 352 | 478 | 42.41% | 39.17% | 42.71% | 7.59 pp | -126 | 45 | -2.80 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 653 | 317 | 336 | 48.55% | 45.83% | 49.58% | 1.45 pp | -19 | 40 | -0.47 |
| BTC Daily | transformer | Transformer | 653 | 317 | 336 | 48.55% | 46.25% | 49.79% | 1.45 pp | -19 | 40 | -0.47 |
| BTC Daily | nn | NN | 653 | 304 | 349 | 46.55% | 41.25% | 48.54% | 3.45 pp | -45 | 40 | -1.12 |
| BTC Daily | lstm | LSTM | 653 | 288 | 365 | 44.10% | 40.83% | 43.33% | 5.90 pp | -77 | 40 | -1.93 |
| BTC Daily | rf | RandomForest | 653 | 278 | 375 | 42.57% | 40.83% | 43.33% | 7.43 pp | -97 | 40 | -2.42 |
| BTC Daily | xgb | XGBoost | 663 | 262 | 401 | 39.52% | 32.50% | 39.79% | 10.48 pp | -139 | 40 | -3.48 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 425 | 210 | 215 | 49.41% | 47.08% | 49.41% | 0.59 pp | -5 | 42 | -0.12 |
| BTC Market Hours | nn | NN | 425 | 201 | 224 | 47.29% | 50.42% | 47.29% | 2.71 pp | -23 | 42 | -0.55 |
| BTC Market Hours | transformer | Transformer | 425 | 194 | 231 | 45.65% | 41.67% | 45.65% | 4.35 pp | -37 | 42 | -0.88 |
| BTC Market Hours | lstm | LSTM | 425 | 186 | 239 | 43.76% | 43.75% | 43.76% | 6.24 pp | -53 | 42 | -1.26 |
| BTC Market Hours | rf | RandomForest | 425 | 184 | 241 | 43.29% | 43.75% | 43.29% | 6.71 pp | -57 | 42 | -1.36 |
| BTC Market Hours | xgb | XGBoost | 425 | 169 | 256 | 39.76% | 37.92% | 39.76% | 10.24 pp | -87 | 42 | -2.07 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 479 | 221 | 258 | 46.14% | 46.67% | 46.14% | 3.86 pp | -37 | 42 | -0.88 |
| BTC Market Hours Daily | nn | NN | 479 | 218 | 261 | 45.51% | 44.17% | 45.51% | 4.49 pp | -43 | 42 | -1.02 |
| BTC Market Hours Daily | transformer | Transformer | 479 | 217 | 262 | 45.30% | 45.00% | 45.30% | 4.70 pp | -45 | 42 | -1.07 |
| BTC Market Hours Daily | rf | RandomForest | 479 | 198 | 281 | 41.34% | 41.67% | 41.34% | 8.66 pp | -83 | 42 | -1.98 |
| BTC Market Hours Daily | lstm | LSTM | 479 | 194 | 285 | 40.50% | 39.17% | 40.50% | 9.50 pp | -91 | 42 | -2.17 |
| BTC Market Hours Daily | xgb | XGBoost | 479 | 185 | 294 | 38.62% | 35.00% | 38.62% | 11.38 pp | -109 | 42 | -2.60 |

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
