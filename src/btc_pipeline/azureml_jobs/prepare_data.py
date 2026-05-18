from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.btc_pipeline import main as tournament
from src.btc_pipeline.azureml_jobs.common import ensure_dir, write_dataframe, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BTC data splits for Azure ML.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lookback-hours", type=int, default=tournament.LOOKBACK_HOURS)
    parser.add_argument("--validation-hours", type=int, default=tournament.VALIDATION_HOURS)
    parser.add_argument("--min-candles", type=int, default=5000)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Optional persistent directory for the raw BTC candle cache. "
            "If reused across runs, only missing recent candles are fetched."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    cache_dir = args.cache_dir or os.getenv("BTC_DATA_CACHE_DIR")
    cache_path = None if cache_dir is None else Path(cache_dir) / "btc_usdt_1h_raw_cache.csv"

    tournament.set_seed()
    raw = tournament.fetch_ohlcv(
        limit=args.lookback_hours,
        min_candles=args.min_candles,
        retry_binanceus=True,
        retry_binanceus_attempts=3,
        cache_path=cache_path,
    )
    featured = tournament.add_features(raw)
    train_df, valid_df, future_row = tournament.split_dataset(featured, args.validation_hours)
    full_labeled_df = featured.iloc[:-1].copy().reset_index(drop=True)

    write_dataframe(raw, output_dir / "raw.csv")
    write_dataframe(featured, output_dir / "featured.csv")
    write_dataframe(train_df, output_dir / "train.csv")
    write_dataframe(valid_df, output_dir / "valid.csv")
    write_dataframe(future_row, output_dir / "future.csv")
    write_dataframe(full_labeled_df, output_dir / "full_labeled.csv")
    write_json(
        {
            "symbol": tournament.SYMBOL,
            "timeframe": tournament.TIMEFRAME,
            "lookback_hours": args.lookback_hours,
            "validation_hours": args.validation_hours,
            "feature_columns": tournament.FEATURE_COLUMNS,
            "validation_start": valid_df["timestamp"].iloc[0].isoformat(),
            "validation_end": valid_df["timestamp"].iloc[-1].isoformat(),
            "reference_candle_timestamp": future_row["timestamp"].iloc[0].isoformat(),
            "cache_path": str(cache_path) if cache_path is not None else None,
        },
        output_dir / "metadata.json",
    )
    print(f"Prepared Azure ML data asset folder at {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
