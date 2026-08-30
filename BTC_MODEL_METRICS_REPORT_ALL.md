# BTC Model Metrics Report - All Rows

Generated at: 2026-08-30T10:53:50.281818+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1144 | 856 | 288 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1019 | 654 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 622 | 416 | 205 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-30 00:00:00+00:00 | 624 | 470 | 152 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 11:00:00+00:00 | 71 | 71 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 11:00:00+00:00 | 71 | 71 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 11:00:00+00:00 | 71 | 0 | 71 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-24 11:00:00+00:00 | 71 | 0 | 71 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 71 | 38 | 33 | 53.52% | 53.52% | 53.52% | 3.52 pp | 5 | 7 | 0.71 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 71 | 38 | 33 | 53.52% | 53.52% | 53.52% | 3.52 pp | 5 | 7 | 0.71 |
| Consolidated Hourly | lstm | LSTM | 71 | 36 | 35 | 50.70% | 50.70% | 50.70% | 0.70 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 71 | 36 | 35 | 50.70% | 50.70% | 50.70% | 0.70 pp | 1 | 7 | 0.14 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 416 | 206 | 210 | 49.52% | 47.92% | 49.52% | 0.48 pp | -4 | 41 | -0.10 |
| BTC Daily | mlp_sklearn | MLPClassifier | 644 | 313 | 331 | 48.60% | 45.83% | 50.00% | 1.40 pp | -18 | 40 | -0.45 |
| BTC Daily | transformer | Transformer | 644 | 312 | 332 | 48.45% | 45.42% | 49.17% | 1.55 pp | -20 | 40 | -0.50 |
| BTC Market Hours | nn | NN | 416 | 197 | 219 | 47.36% | 50.83% | 47.36% | 2.64 pp | -22 | 41 | -0.54 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 470 | 218 | 252 | 46.38% | 46.67% | 46.38% | 3.62 pp | -34 | 41 | -0.83 |
| BTC Market Hours | transformer | Transformer | 416 | 191 | 225 | 45.91% | 41.67% | 45.91% | 4.09 pp | -34 | 41 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 470 | 215 | 255 | 45.74% | 45.42% | 45.74% | 4.26 pp | -40 | 41 | -0.98 |
| BTC Hourly | transformer | Transformer | 822 | 389 | 433 | 47.32% | 46.67% | 46.46% | 2.68 pp | -44 | 44 | -1.00 |
| BTC Market Hours Daily | nn | NN | 470 | 214 | 256 | 45.53% | 44.58% | 45.53% | 4.47 pp | -42 | 41 | -1.02 |
| BTC Daily | nn | NN | 644 | 301 | 343 | 46.74% | 42.08% | 48.96% | 3.26 pp | -42 | 40 | -1.05 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 822 | 386 | 436 | 46.96% | 42.50% | 46.88% | 3.04 pp | -50 | 44 | -1.14 |
| BTC Market Hours | lstm | LSTM | 416 | 183 | 233 | 43.99% | 44.17% | 43.99% | 6.01 pp | -50 | 41 | -1.22 |
| Consolidated Hourly | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| BTC Market Hours | rf | RandomForest | 416 | 180 | 236 | 43.27% | 42.50% | 43.27% | 6.73 pp | -56 | 41 | -1.37 |
| Consolidated Hourly | transformer | Transformer | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 7 | -1.57 |
| BTC Hourly | nn | NN | 822 | 372 | 450 | 45.26% | 42.50% | 45.00% | 4.74 pp | -78 | 44 | -1.77 |
| BTC Daily | lstm | LSTM | 644 | 285 | 359 | 44.25% | 41.67% | 43.75% | 5.75 pp | -74 | 40 | -1.85 |
| BTC Market Hours Daily | rf | RandomForest | 470 | 195 | 275 | 41.49% | 42.50% | 41.49% | 8.51 pp | -80 | 41 | -1.95 |
| BTC Hourly | rf | RandomForest | 822 | 368 | 454 | 44.77% | 44.58% | 44.58% | 5.23 pp | -86 | 44 | -1.95 |
| BTC Market Hours | xgb | XGBoost | 416 | 166 | 250 | 39.90% | 37.50% | 39.90% | 10.10 pp | -84 | 41 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 470 | 192 | 278 | 40.85% | 39.58% | 40.85% | 9.15 pp | -86 | 41 | -2.10 |
| BTC Daily | rf | RandomForest | 644 | 274 | 370 | 42.55% | 40.83% | 43.33% | 7.45 pp | -96 | 40 | -2.40 |
| Consolidated Hourly | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 7 | -2.43 |
| Consolidated Daily/Hourly Refresh | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 7 | -2.43 |
| BTC Hourly | lstm | LSTM | 822 | 357 | 465 | 43.43% | 41.25% | 43.75% | 6.57 pp | -108 | 44 | -2.45 |
| BTC Market Hours Daily | xgb | XGBoost | 470 | 182 | 288 | 38.72% | 35.42% | 38.72% | 11.28 pp | -106 | 41 | -2.59 |
| BTC Hourly | xgb | XGBoost | 822 | 347 | 475 | 42.21% | 39.58% | 42.50% | 7.79 pp | -128 | 44 | -2.91 |
| BTC Daily | xgb | XGBoost | 654 | 257 | 397 | 39.30% | 31.25% | 39.38% | 10.70 pp | -140 | 40 | -3.50 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | transformer | Transformer | 822 | 389 | 433 | 47.32% | 46.67% | 46.46% | 2.68 pp | -44 | 44 | -1.00 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 822 | 386 | 436 | 46.96% | 42.50% | 46.88% | 3.04 pp | -50 | 44 | -1.14 |
| BTC Hourly | nn | NN | 822 | 372 | 450 | 45.26% | 42.50% | 45.00% | 4.74 pp | -78 | 44 | -1.77 |
| BTC Hourly | rf | RandomForest | 822 | 368 | 454 | 44.77% | 44.58% | 44.58% | 5.23 pp | -86 | 44 | -1.95 |
| BTC Hourly | lstm | LSTM | 822 | 357 | 465 | 43.43% | 41.25% | 43.75% | 6.57 pp | -108 | 44 | -2.45 |
| BTC Hourly | xgb | XGBoost | 822 | 347 | 475 | 42.21% | 39.58% | 42.50% | 7.79 pp | -128 | 44 | -2.91 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | mlp_sklearn | MLPClassifier | 644 | 313 | 331 | 48.60% | 45.83% | 50.00% | 1.40 pp | -18 | 40 | -0.45 |
| BTC Daily | transformer | Transformer | 644 | 312 | 332 | 48.45% | 45.42% | 49.17% | 1.55 pp | -20 | 40 | -0.50 |
| BTC Daily | nn | NN | 644 | 301 | 343 | 46.74% | 42.08% | 48.96% | 3.26 pp | -42 | 40 | -1.05 |
| BTC Daily | lstm | LSTM | 644 | 285 | 359 | 44.25% | 41.67% | 43.75% | 5.75 pp | -74 | 40 | -1.85 |
| BTC Daily | rf | RandomForest | 644 | 274 | 370 | 42.55% | 40.83% | 43.33% | 7.45 pp | -96 | 40 | -2.40 |
| BTC Daily | xgb | XGBoost | 654 | 257 | 397 | 39.30% | 31.25% | 39.38% | 10.70 pp | -140 | 40 | -3.50 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 416 | 206 | 210 | 49.52% | 47.92% | 49.52% | 0.48 pp | -4 | 41 | -0.10 |
| BTC Market Hours | nn | NN | 416 | 197 | 219 | 47.36% | 50.83% | 47.36% | 2.64 pp | -22 | 41 | -0.54 |
| BTC Market Hours | transformer | Transformer | 416 | 191 | 225 | 45.91% | 41.67% | 45.91% | 4.09 pp | -34 | 41 | -0.83 |
| BTC Market Hours | lstm | LSTM | 416 | 183 | 233 | 43.99% | 44.17% | 43.99% | 6.01 pp | -50 | 41 | -1.22 |
| BTC Market Hours | rf | RandomForest | 416 | 180 | 236 | 43.27% | 42.50% | 43.27% | 6.73 pp | -56 | 41 | -1.37 |
| BTC Market Hours | xgb | XGBoost | 416 | 166 | 250 | 39.90% | 37.50% | 39.90% | 10.10 pp | -84 | 41 | -2.05 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 470 | 218 | 252 | 46.38% | 46.67% | 46.38% | 3.62 pp | -34 | 41 | -0.83 |
| BTC Market Hours Daily | transformer | Transformer | 470 | 215 | 255 | 45.74% | 45.42% | 45.74% | 4.26 pp | -40 | 41 | -0.98 |
| BTC Market Hours Daily | nn | NN | 470 | 214 | 256 | 45.53% | 44.58% | 45.53% | 4.47 pp | -42 | 41 | -1.02 |
| BTC Market Hours Daily | rf | RandomForest | 470 | 195 | 275 | 41.49% | 42.50% | 41.49% | 8.51 pp | -80 | 41 | -1.95 |
| BTC Market Hours Daily | lstm | LSTM | 470 | 192 | 278 | 40.85% | 39.58% | 40.85% | 9.15 pp | -86 | 41 | -2.10 |
| BTC Market Hours Daily | xgb | XGBoost | 470 | 182 | 288 | 38.72% | 35.42% | 38.72% | 11.28 pp | -106 | 41 | -2.59 |

### Consolidated Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | rf | RandomForest | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 71 | 38 | 33 | 53.52% | 53.52% | 53.52% | 3.52 pp | 5 | 7 | 0.71 |
| Consolidated Hourly | lstm | LSTM | 71 | 36 | 35 | 50.70% | 50.70% | 50.70% | 0.70 pp | 1 | 7 | 0.14 |
| Consolidated Hourly | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| Consolidated Hourly | transformer | Transformer | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 7 | -1.57 |
| Consolidated Hourly | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 7 | -2.43 |

### Consolidated Daily/Hourly Refresh

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 71 | 39 | 32 | 54.93% | 54.93% | 54.93% | 4.93 pp | 7 | 7 | 1.00 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 71 | 38 | 33 | 53.52% | 53.52% | 53.52% | 3.52 pp | 5 | 7 | 0.71 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 71 | 36 | 35 | 50.70% | 50.70% | 50.70% | 0.70 pp | 1 | 7 | 0.14 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 71 | 31 | 40 | 43.66% | 43.66% | 43.66% | 6.34 pp | -9 | 7 | -1.29 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 71 | 30 | 41 | 42.25% | 42.25% | 42.25% | 7.75 pp | -11 | 7 | -1.57 |
| Consolidated Daily/Hourly Refresh | nn | NN | 71 | 27 | 44 | 38.03% | 38.03% | 38.03% | 11.97 pp | -17 | 7 | -2.43 |

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
