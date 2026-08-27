# BTC Model Metrics Report - All Rows

Generated at: 2026-08-27T12:45:08.274676+00:00
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
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 1119 | 802 | 317 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-08-23 15:00:00+00:00 | 965 | 600 | 364 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 529 | 362 | 166 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-08-27 00:00:00+00:00 | 531 | 416 | 113 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 10:00:00+00:00 | 26 | 26 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 10:00:00+00:00 | 26 | 26 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 10:00:00+00:00 | 26 | 0 | 26 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | 2026-05-18 07:00:00+00:00 to 2026-05-20 10:00:00+00:00 | 26 | 0 | 26 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Consolidated Hourly | lstm | LSTM | 26 | 17 | 9 | 65.38% | 65.38% | 65.38% | 15.38 pp | 8 | 3 | 2.67 |
| Consolidated Daily/Hourly Refresh | lstm | LSTM | 26 | 17 | 9 | 65.38% | 65.38% | 65.38% | 15.38 pp | 8 | 3 | 2.67 |
| Consolidated Hourly | mlp_sklearn | MLPClassifier | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | rf | RandomForest | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 3 | 0.67 |
| Consolidated Hourly | transformer | Transformer | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | mlp_sklearn | MLPClassifier | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | rf | RandomForest | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 3 | 0.67 |
| Consolidated Daily/Hourly Refresh | transformer | Transformer | 26 | 14 | 12 | 53.85% | 53.85% | 53.85% | 3.85 pp | 2 | 3 | 0.67 |
| BTC Daily | transformer | Transformer | 590 | 292 | 298 | 49.49% | 52.08% | 50.00% | 0.51 pp | -6 | 37 | -0.16 |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 362 | 178 | 184 | 49.17% | 47.92% | 49.17% | 0.83 pp | -6 | 37 | -0.16 |
| BTC Daily | mlp_sklearn | MLPClassifier | 590 | 288 | 302 | 48.81% | 47.08% | 49.38% | 1.19 pp | -14 | 37 | -0.38 |
| BTC Market Hours | transformer | Transformer | 362 | 172 | 190 | 47.51% | 46.25% | 47.51% | 2.49 pp | -18 | 37 | -0.49 |
| BTC Daily | nn | NN | 590 | 278 | 312 | 47.12% | 45.42% | 48.12% | 2.88 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | nn | NN | 416 | 191 | 225 | 45.91% | 46.67% | 45.91% | 4.09 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 416 | 191 | 225 | 45.91% | 47.92% | 45.91% | 4.09 pp | -34 | 37 | -0.92 |
| BTC Market Hours | nn | NN | 362 | 164 | 198 | 45.30% | 46.67% | 45.30% | 4.70 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 416 | 190 | 226 | 45.67% | 45.42% | 45.67% | 4.33 pp | -36 | 37 | -0.97 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 768 | 362 | 406 | 47.14% | 43.33% | 47.29% | 2.86 pp | -44 | 42 | -1.05 |
| BTC Hourly | transformer | Transformer | 768 | 359 | 409 | 46.74% | 43.33% | 45.42% | 3.26 pp | -50 | 42 | -1.19 |
| Consolidated Hourly | xgb | XGBoost | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 3 | -1.33 |
| Consolidated Daily/Hourly Refresh | xgb | XGBoost | 26 | 11 | 15 | 42.31% | 42.31% | 42.31% | 7.69 pp | -4 | 3 | -1.33 |
| BTC Market Hours | lstm | LSTM | 362 | 156 | 206 | 43.09% | 43.33% | 43.09% | 6.91 pp | -50 | 37 | -1.35 |
| BTC Market Hours | rf | RandomForest | 362 | 154 | 208 | 42.54% | 41.67% | 42.54% | 7.46 pp | -54 | 37 | -1.46 |
| BTC Daily | lstm | LSTM | 590 | 265 | 325 | 44.92% | 44.58% | 45.00% | 5.08 pp | -60 | 37 | -1.62 |
| BTC Market Hours | xgb | XGBoost | 362 | 148 | 214 | 40.88% | 41.67% | 40.88% | 9.12 pp | -66 | 37 | -1.78 |
| BTC Hourly | rf | RandomForest | 768 | 344 | 424 | 44.79% | 44.58% | 44.58% | 5.21 pp | -80 | 42 | -1.90 |
| BTC Hourly | nn | NN | 768 | 343 | 425 | 44.66% | 40.42% | 45.42% | 5.34 pp | -82 | 42 | -1.95 |
| BTC Market Hours Daily | rf | RandomForest | 416 | 170 | 246 | 40.87% | 40.42% | 40.87% | 9.13 pp | -76 | 37 | -2.05 |
| BTC Hourly | lstm | LSTM | 768 | 339 | 429 | 44.14% | 43.75% | 45.42% | 5.86 pp | -90 | 42 | -2.14 |
| BTC Daily | rf | RandomForest | 590 | 255 | 335 | 43.22% | 44.17% | 43.96% | 6.78 pp | -80 | 37 | -2.16 |
| BTC Market Hours Daily | lstm | LSTM | 416 | 166 | 250 | 39.90% | 38.33% | 39.90% | 10.10 pp | -84 | 37 | -2.27 |
| BTC Market Hours Daily | xgb | XGBoost | 416 | 166 | 250 | 39.90% | 38.75% | 39.90% | 10.10 pp | -84 | 37 | -2.27 |
| BTC Hourly | xgb | XGBoost | 768 | 329 | 439 | 42.84% | 41.25% | 44.38% | 7.16 pp | -110 | 42 | -2.62 |
| BTC Daily | xgb | XGBoost | 600 | 241 | 359 | 40.17% | 36.25% | 40.42% | 9.83 pp | -118 | 37 | -3.19 |
| Consolidated Hourly | nn | NN | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 3 | -4.00 |
| Consolidated Daily/Hourly Refresh | nn | NN | 26 | 7 | 19 | 26.92% | 26.92% | 26.92% | 23.08 pp | -12 | 3 | -4.00 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 768 | 362 | 406 | 47.14% | 43.33% | 47.29% | 2.86 pp | -44 | 42 | -1.05 |
| BTC Hourly | transformer | Transformer | 768 | 359 | 409 | 46.74% | 43.33% | 45.42% | 3.26 pp | -50 | 42 | -1.19 |
| BTC Hourly | rf | RandomForest | 768 | 344 | 424 | 44.79% | 44.58% | 44.58% | 5.21 pp | -80 | 42 | -1.90 |
| BTC Hourly | nn | NN | 768 | 343 | 425 | 44.66% | 40.42% | 45.42% | 5.34 pp | -82 | 42 | -1.95 |
| BTC Hourly | lstm | LSTM | 768 | 339 | 429 | 44.14% | 43.75% | 45.42% | 5.86 pp | -90 | 42 | -2.14 |
| BTC Hourly | xgb | XGBoost | 768 | 329 | 439 | 42.84% | 41.25% | 44.38% | 7.16 pp | -110 | 42 | -2.62 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 590 | 292 | 298 | 49.49% | 52.08% | 50.00% | 0.51 pp | -6 | 37 | -0.16 |
| BTC Daily | mlp_sklearn | MLPClassifier | 590 | 288 | 302 | 48.81% | 47.08% | 49.38% | 1.19 pp | -14 | 37 | -0.38 |
| BTC Daily | nn | NN | 590 | 278 | 312 | 47.12% | 45.42% | 48.12% | 2.88 pp | -34 | 37 | -0.92 |
| BTC Daily | lstm | LSTM | 590 | 265 | 325 | 44.92% | 44.58% | 45.00% | 5.08 pp | -60 | 37 | -1.62 |
| BTC Daily | rf | RandomForest | 590 | 255 | 335 | 43.22% | 44.17% | 43.96% | 6.78 pp | -80 | 37 | -2.16 |
| BTC Daily | xgb | XGBoost | 600 | 241 | 359 | 40.17% | 36.25% | 40.42% | 9.83 pp | -118 | 37 | -3.19 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 362 | 178 | 184 | 49.17% | 47.92% | 49.17% | 0.83 pp | -6 | 37 | -0.16 |
| BTC Market Hours | transformer | Transformer | 362 | 172 | 190 | 47.51% | 46.25% | 47.51% | 2.49 pp | -18 | 37 | -0.49 |
| BTC Market Hours | nn | NN | 362 | 164 | 198 | 45.30% | 46.67% | 45.30% | 4.70 pp | -34 | 37 | -0.92 |
| BTC Market Hours | lstm | LSTM | 362 | 156 | 206 | 43.09% | 43.33% | 43.09% | 6.91 pp | -50 | 37 | -1.35 |
| BTC Market Hours | rf | RandomForest | 362 | 154 | 208 | 42.54% | 41.67% | 42.54% | 7.46 pp | -54 | 37 | -1.46 |
| BTC Market Hours | xgb | XGBoost | 362 | 148 | 214 | 40.88% | 41.67% | 40.88% | 9.12 pp | -66 | 37 | -1.78 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 416 | 191 | 225 | 45.91% | 46.67% | 45.91% | 4.09 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | transformer | Transformer | 416 | 191 | 225 | 45.91% | 47.92% | 45.91% | 4.09 pp | -34 | 37 | -0.92 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 416 | 190 | 226 | 45.67% | 45.42% | 45.67% | 4.33 pp | -36 | 37 | -0.97 |
| BTC Market Hours Daily | rf | RandomForest | 416 | 170 | 246 | 40.87% | 40.42% | 40.87% | 9.13 pp | -76 | 37 | -2.05 |
| BTC Market Hours Daily | lstm | LSTM | 416 | 166 | 250 | 39.90% | 38.33% | 39.90% | 10.10 pp | -84 | 37 | -2.27 |
| BTC Market Hours Daily | xgb | XGBoost | 416 | 166 | 250 | 39.90% | 38.75% | 39.90% | 10.10 pp | -84 | 37 | -2.27 |

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
