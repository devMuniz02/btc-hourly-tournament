# BTC Model Metrics Report - Old Baseline

Generated at: 2026-09-05T21:02:15.982720+00:00
Scope: `old`

## Source Files

- `artifacts/btc/hourly/history.csv`
- `artifacts/btc/daily/history.csv`
- `artifacts/btc/market_hours/history.csv`
- `artifacts/btc/market_hours_daily/history.csv`
- `artifacts/consolidated/history.csv`

## Coverage Metadata

| Variation | Source File | Date Range | Rows | Validated | Missing | Failed |
| --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | artifacts/btc/hourly/history.csv | 2026-03-16 15:00:00+00:00 to 2026-04-27 23:00:00+00:00 | 1017 | 789 | 228 | 0 |
| BTC Daily | artifacts/btc/daily/history.csv | 2026-03-21 04:00:00+00:00 to 2026-04-27 22:00:00+00:00 | 857 | 552 | 304 | 1 |
| BTC Market Hours | artifacts/btc/market_hours/history.csv | 2026-03-24 22:00:00+00:00 to 2026-04-27 23:00:00+00:00 | 441 | 326 | 114 | 1 |
| BTC Market Hours Daily | artifacts/btc/market_hours_daily/history.csv | 2026-03-24 22:00:00+00:00 to 2026-04-27 22:00:00+00:00 | 443 | 380 | 61 | 2 |
| Consolidated Hourly | artifacts/consolidated/history.csv | n/a | 0 | 0 | 0 | 0 |
| Consolidated Daily/Hourly Refresh | artifacts/consolidated/history.csv | n/a | 0 | 0 | 0 | 0 |
| Consolidated Market Hours | artifacts/consolidated/history.csv | n/a | 0 | 0 | 0 | 0 |
| Consolidated Market Hours Daily | artifacts/consolidated/history.csv | n/a | 0 | 0 | 0 | 0 |

## Overall Ranking

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 326 | 162 | 164 | 49.69% | 48.33% | 49.69% | 0.31 pp | -2 | 34 | -0.06 |
| BTC Daily | transformer | Transformer | 552 | 271 | 281 | 49.09% | 52.50% | 49.58% | 0.91 pp | -10 | 35 | -0.29 |
| BTC Daily | mlp_sklearn | MLPClassifier | 552 | 270 | 282 | 48.91% | 49.17% | 48.96% | 1.09 pp | -12 | 35 | -0.34 |
| BTC Market Hours | transformer | Transformer | 326 | 154 | 172 | 47.24% | 47.08% | 47.24% | 2.76 pp | -18 | 34 | -0.53 |
| BTC Market Hours Daily | nn | NN | 380 | 175 | 205 | 46.05% | 47.92% | 46.05% | 3.95 pp | -30 | 34 | -0.88 |
| BTC Daily | nn | NN | 552 | 259 | 293 | 46.92% | 44.58% | 46.88% | 3.08 pp | -34 | 35 | -0.97 |
| BTC Market Hours | nn | NN | 326 | 146 | 180 | 44.79% | 46.67% | 44.79% | 5.21 pp | -34 | 34 | -1.00 |
| BTC Market Hours Daily | transformer | Transformer | 380 | 172 | 208 | 45.26% | 46.25% | 45.26% | 4.74 pp | -36 | 34 | -1.06 |
| BTC Hourly | mlp_sklearn | MLPClassifier | 755 | 355 | 400 | 47.02% | 43.75% | 47.71% | 2.98 pp | -45 | 41 | -1.10 |
| BTC Hourly | transformer | Transformer | 755 | 355 | 400 | 47.02% | 44.17% | 46.04% | 2.98 pp | -45 | 41 | -1.10 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 380 | 171 | 209 | 45.00% | 45.42% | 45.00% | 5.00 pp | -38 | 34 | -1.12 |
| BTC Market Hours | lstm | LSTM | 326 | 144 | 182 | 44.17% | 45.42% | 44.17% | 5.83 pp | -38 | 34 | -1.12 |
| BTC Daily | lstm | LSTM | 552 | 252 | 300 | 45.65% | 46.25% | 45.62% | 4.35 pp | -48 | 35 | -1.37 |
| BTC Market Hours | rf | RandomForest | 326 | 135 | 191 | 41.41% | 41.67% | 41.41% | 8.59 pp | -56 | 34 | -1.65 |
| BTC Hourly | rf | RandomForest | 755 | 340 | 415 | 45.03% | 45.00% | 45.00% | 4.97 pp | -75 | 41 | -1.83 |
| BTC Market Hours | xgb | XGBoost | 326 | 131 | 195 | 40.18% | 40.42% | 40.18% | 9.82 pp | -64 | 34 | -1.88 |
| BTC Hourly | nn | NN | 755 | 338 | 417 | 44.77% | 41.67% | 45.21% | 5.23 pp | -79 | 41 | -1.93 |
| BTC Market Hours Daily | lstm | LSTM | 380 | 155 | 225 | 40.79% | 39.17% | 40.79% | 9.21 pp | -70 | 34 | -2.06 |
| BTC Daily | rf | RandomForest | 552 | 239 | 313 | 43.30% | 44.58% | 43.75% | 6.70 pp | -74 | 35 | -2.11 |
| BTC Hourly | lstm | LSTM | 755 | 333 | 422 | 44.11% | 42.92% | 45.83% | 5.89 pp | -89 | 41 | -2.17 |
| BTC Market Hours Daily | rf | RandomForest | 380 | 153 | 227 | 40.26% | 37.08% | 40.26% | 9.74 pp | -74 | 34 | -2.18 |
| BTC Market Hours Daily | xgb | XGBoost | 380 | 149 | 231 | 39.21% | 37.08% | 39.21% | 10.79 pp | -82 | 34 | -2.41 |
| BTC Hourly | xgb | XGBoost | 755 | 326 | 429 | 43.18% | 42.08% | 44.79% | 6.82 pp | -103 | 41 | -2.51 |
| BTC Daily | xgb | XGBoost | 552 | 224 | 328 | 40.58% | 37.50% | 41.04% | 9.42 pp | -104 | 35 | -2.97 |

## Variation Tables

### BTC Hourly

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Hourly | mlp_sklearn | MLPClassifier | 755 | 355 | 400 | 47.02% | 43.75% | 47.71% | 2.98 pp | -45 | 41 | -1.10 |
| BTC Hourly | transformer | Transformer | 755 | 355 | 400 | 47.02% | 44.17% | 46.04% | 2.98 pp | -45 | 41 | -1.10 |
| BTC Hourly | rf | RandomForest | 755 | 340 | 415 | 45.03% | 45.00% | 45.00% | 4.97 pp | -75 | 41 | -1.83 |
| BTC Hourly | nn | NN | 755 | 338 | 417 | 44.77% | 41.67% | 45.21% | 5.23 pp | -79 | 41 | -1.93 |
| BTC Hourly | lstm | LSTM | 755 | 333 | 422 | 44.11% | 42.92% | 45.83% | 5.89 pp | -89 | 41 | -2.17 |
| BTC Hourly | xgb | XGBoost | 755 | 326 | 429 | 43.18% | 42.08% | 44.79% | 6.82 pp | -103 | 41 | -2.51 |

### BTC Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Daily | transformer | Transformer | 552 | 271 | 281 | 49.09% | 52.50% | 49.58% | 0.91 pp | -10 | 35 | -0.29 |
| BTC Daily | mlp_sklearn | MLPClassifier | 552 | 270 | 282 | 48.91% | 49.17% | 48.96% | 1.09 pp | -12 | 35 | -0.34 |
| BTC Daily | nn | NN | 552 | 259 | 293 | 46.92% | 44.58% | 46.88% | 3.08 pp | -34 | 35 | -0.97 |
| BTC Daily | lstm | LSTM | 552 | 252 | 300 | 45.65% | 46.25% | 45.62% | 4.35 pp | -48 | 35 | -1.37 |
| BTC Daily | rf | RandomForest | 552 | 239 | 313 | 43.30% | 44.58% | 43.75% | 6.70 pp | -74 | 35 | -2.11 |
| BTC Daily | xgb | XGBoost | 552 | 224 | 328 | 40.58% | 37.50% | 41.04% | 9.42 pp | -104 | 35 | -2.97 |

### BTC Market Hours

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours | mlp_sklearn | MLPClassifier | 326 | 162 | 164 | 49.69% | 48.33% | 49.69% | 0.31 pp | -2 | 34 | -0.06 |
| BTC Market Hours | transformer | Transformer | 326 | 154 | 172 | 47.24% | 47.08% | 47.24% | 2.76 pp | -18 | 34 | -0.53 |
| BTC Market Hours | nn | NN | 326 | 146 | 180 | 44.79% | 46.67% | 44.79% | 5.21 pp | -34 | 34 | -1.00 |
| BTC Market Hours | lstm | LSTM | 326 | 144 | 182 | 44.17% | 45.42% | 44.17% | 5.83 pp | -38 | 34 | -1.12 |
| BTC Market Hours | rf | RandomForest | 326 | 135 | 191 | 41.41% | 41.67% | 41.41% | 8.59 pp | -56 | 34 | -1.65 |
| BTC Market Hours | xgb | XGBoost | 326 | 131 | 195 | 40.18% | 40.42% | 40.18% | 9.82 pp | -64 | 34 | -1.88 |

### BTC Market Hours Daily

| Variation | Model Family | Model Name | Evaluated Predictions | Wins | Losses | Accuracy | Accuracy Last 240 | Accuracy Last 480 | Accuracy Delta From 50% | Net Wins | Active Days | Net Wins / Day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC Market Hours Daily | nn | NN | 380 | 175 | 205 | 46.05% | 47.92% | 46.05% | 3.95 pp | -30 | 34 | -0.88 |
| BTC Market Hours Daily | transformer | Transformer | 380 | 172 | 208 | 45.26% | 46.25% | 45.26% | 4.74 pp | -36 | 34 | -1.06 |
| BTC Market Hours Daily | mlp_sklearn | MLPClassifier | 380 | 171 | 209 | 45.00% | 45.42% | 45.00% | 5.00 pp | -38 | 34 | -1.12 |
| BTC Market Hours Daily | lstm | LSTM | 380 | 155 | 225 | 40.79% | 39.17% | 40.79% | 9.21 pp | -70 | 34 | -2.06 |
| BTC Market Hours Daily | rf | RandomForest | 380 | 153 | 227 | 40.26% | 37.08% | 40.26% | 9.74 pp | -74 | 34 | -2.18 |
| BTC Market Hours Daily | xgb | XGBoost | 380 | 149 | 231 | 39.21% | 37.08% | 39.21% | 10.79 pp | -82 | 34 | -2.41 |

### Consolidated Hourly

_No model-level predictions available for this variation._

### Consolidated Daily/Hourly Refresh

_No model-level predictions available for this variation._

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
