#!/usr/bin/env python3
"""Generate BTC model metrics report from local artifact CSV files."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "BTC_MODEL_METRICS_REPORT.md"
REPORT_ALL_PATH = ROOT / "BTC_MODEL_METRICS_REPORT_ALL.md"
REPORT_OLD_PATH = ROOT / "BTC_MODEL_METRICS_REPORT_OLD.md"
REPORT_NEW_PATH = ROOT / "BTC_MODEL_METRICS_REPORT_NEW.md"
REPORT_NEWTEST_PATH = ROOT / "BTC_MODEL_METRICS_REPORT_NEWTEST.md"
SOURCES = [
    ("BTC Hourly", ROOT / "artifacts/btc/hourly/history.csv"),
    ("BTC Daily", ROOT / "artifacts/btc/daily/history.csv"),
    ("BTC Market Hours", ROOT / "artifacts/btc/market_hours/history.csv"),
    ("BTC Market Hours Daily", ROOT / "artifacts/btc/market_hours_daily/history.csv"),
]
NEWTEST_SOURCES = [
    ("NEWTEST BTC Daily", ROOT / "artifacts/newtest/btc_daily_history.csv"),
]
CONSOLIDATED = ROOT / "artifacts/consolidated/history.csv"
CONSOLIDATED_NAMES = {
    "consolidated-hourly-24h": "Consolidated Hourly",
    "consolidated-hourly-daily": "Consolidated Daily/Hourly Refresh",
    "consolidated-market-hours": "Consolidated Market Hours",
    "consolidated-market-hours-daily": "Consolidated Market Hours Daily",
}
MODEL_ORDER = ["lstm", "mlp_sklearn", "nn", "rf", "transformer", "xgb"]
OLD_NEW_SPLIT_UTC = {
    "BTC Hourly": "2026-04-27T23:00:00+00:00",
    "BTC Daily": "2026-04-27T22:00:00+00:00",
    "BTC Market Hours": "2026-04-27T23:00:00+00:00",
    "BTC Market Hours Daily": "2026-04-27T22:00:00+00:00",
    "Consolidated Hourly": "2026-05-18T06:00:00+00:00",
    "Consolidated Daily/Hourly Refresh": "2026-05-18T06:00:00+00:00",
    "Consolidated Market Hours": "2026-05-18T06:00:00+00:00",
    "Consolidated Market Hours Daily": "2026-05-18T06:00:00+00:00",
}


def parse_label(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"UP", "1", "1.0", "TRUE"}:
        return 1
    if text in {"DOWN", "0", "0.0", "FALSE"}:
        return 0
    return None


def parse_ts(value: Any) -> datetime:
    text = str(value or "").strip().replace("Z", "+00:00")
    try:
        timestamp = datetime.fromisoformat(text)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def pp(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f} pp"


def empty_model_stats() -> dict[str, Any]:
    return {"model_name": "", "evaluations": [], "dates": set()}


def new_variation(label: str, source: Path) -> dict[str, Any]:
    return {
        "label": label,
        "source": str(source.relative_to(ROOT)),
        "rows": 0,
        "statuses": Counter(),
        "timestamps": [],
        "models": defaultdict(empty_model_stats),
    }


def ingest_row(variation: dict[str, Any], row: dict[str, str]) -> None:
    variation["rows"] += 1
    variation["statuses"][row.get("status", "") or "blank"] += 1
    timestamp_text = row.get("timestamp", "")
    if timestamp_text:
        variation["timestamps"].append(timestamp_text)
    actual = parse_label(row.get("actual"))
    if actual is None:
        return
    raw_predictions = row.get("model_predictions") or ""
    if not raw_predictions.strip():
        return
    try:
        predictions = json.loads(raw_predictions)
    except json.JSONDecodeError:
        return
    if not isinstance(predictions, dict):
        return
    timestamp = parse_ts(timestamp_text)
    for family, payload in predictions.items():
        if not isinstance(payload, dict):
            continue
        predicted = parse_label(payload.get("predicted_label", payload.get("predicted_signal")))
        if predicted is None:
            continue
        stats = variation["models"][family]
        stats["model_name"] = payload.get("name") or stats["model_name"] or family
        stats["dates"].add(timestamp.date().isoformat())
        stats["evaluations"].append((timestamp, predicted == actual))


def latest_accuracy(evaluations: list[tuple[datetime, bool]], window: int) -> float | None:
    if not evaluations:
        return None
    latest = sorted(evaluations, key=lambda item: item[0])[-window:]
    return sum(1 for _, correct in latest if correct) / len(latest)


def model_rows(variation: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    families = sorted(
        variation["models"],
        key=lambda family: (MODEL_ORDER.index(family) if family in MODEL_ORDER else 99, family),
    )
    for family in families:
        stats = variation["models"][family]
        evaluations = sorted(stats["evaluations"], key=lambda item: item[0])
        evaluated = len(evaluations)
        wins = sum(1 for _, correct in evaluations if correct)
        losses = evaluated - wins
        active_days = len(stats["dates"])
        accuracy = None if not evaluated else wins / evaluated
        net_wins = wins - losses
        net_per_day = None if not active_days else net_wins / active_days
        rows.append(
            {
                "Variation": variation["label"],
                "Model Family": family,
                "Model Name": stats["model_name"] or family,
                "Evaluated Predictions": evaluated,
                "Wins": wins,
                "Losses": losses,
                "Accuracy": pct(accuracy),
                "Accuracy Last 240": pct(latest_accuracy(evaluations, 240)),
                "Accuracy Last 480": pct(latest_accuracy(evaluations, 480)),
                "Accuracy Delta From 50%": pp(None if accuracy is None else abs(accuracy - 0.5)),
                "Net Wins": net_wins,
                "Active Days": active_days,
                "Net Wins / Day": "n/a" if net_per_day is None else f"{net_per_day:.2f}",
                "_sort_net_per_day": -10**9 if net_per_day is None else net_per_day,
                "_sort_accuracy": -1 if accuracy is None else accuracy,
                "_sort_evaluated": evaluated,
            }
        )
    return rows


def md_table(headers: list[str], rows: list[dict[str, Any]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(output)


def include_row(label: str, row: dict[str, str], scope: str) -> bool:
    if scope in {"all", "newtest"}:
        return True
    split_text = OLD_NEW_SPLIT_UTC.get(label)
    if split_text is None:
        return True
    timestamp = parse_ts(row.get("timestamp"))
    split = parse_ts(split_text)
    if scope == "old":
        return timestamp <= split
    if scope == "new":
        return timestamp > split
    raise ValueError(f"Unsupported report scope: {scope}")


def load_variations(scope: str = "all") -> tuple[list[Path], list[dict[str, Any]]]:
    source_files: list[Path] = []
    variations: dict[tuple[str, str], dict[str, Any]] = {}

    def ensure(label: str, source: Path) -> dict[str, Any]:
        key = (label, str(source))
        if key not in variations:
            variations[key] = new_variation(label, source)
        return variations[key]

    selected_sources = NEWTEST_SOURCES if scope == "newtest" else SOURCES
    for label, path in selected_sources:
        source_files.append(path)
        variation = ensure(label, path)
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if include_row(label, row, scope):
                    ingest_row(variation, row)

    if scope != "newtest":
        source_files.append(CONSOLIDATED)
        with CONSOLIDATED.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                workflow = row.get("workflow_name") or "consolidated-unknown"
                label = CONSOLIDATED_NAMES.get(workflow, workflow.replace("-", " ").title())
                variation = ensure(label, CONSOLIDATED)
                if include_row(label, row, scope):
                    ingest_row(variation, row)

    return source_files, list(variations.values())


def report_title(scope: str) -> str:
    if scope == "newtest":
        return "BTC Model Metrics Report - NEWTEST"
    if scope == "old":
        return "BTC Model Metrics Report - Old Baseline"
    if scope == "new":
        return "BTC Model Metrics Report - New Forward Rows"
    return "BTC Model Metrics Report - All Rows"


def report_path(scope: str) -> Path:
    if scope == "newtest":
        return REPORT_NEWTEST_PATH
    if scope == "old":
        return REPORT_OLD_PATH
    if scope == "new":
        return REPORT_NEW_PATH
    return REPORT_ALL_PATH


def generate_report(scope: str = "all", output_path: Path | None = None) -> Path:
    source_files, variations = load_variations(scope)
    headers = [
        "Variation",
        "Model Family",
        "Model Name",
        "Evaluated Predictions",
        "Wins",
        "Losses",
        "Accuracy",
        "Accuracy Last 240",
        "Accuracy Last 480",
        "Accuracy Delta From 50%",
        "Net Wins",
        "Active Days",
        "Net Wins / Day",
    ]
    all_rows = [row for variation in variations for row in model_rows(variation)]
    ranking = sorted(
        all_rows,
        key=lambda row: (row["_sort_net_per_day"], row["_sort_accuracy"], row["_sort_evaluated"]),
        reverse=True,
    )
    metadata_rows = []
    for variation in variations:
        timestamps = variation["timestamps"]
        metadata_rows.append(
            {
                "Variation": variation["label"],
                "Source File": variation["source"],
                "Date Range": "n/a" if not timestamps else f"{min(timestamps)} to {max(timestamps)}",
                "Rows": variation["rows"],
                "Validated": variation["statuses"].get("validated", 0),
                "Missing": variation["statuses"].get("missing", 0),
                "Failed": variation["statuses"].get("failed", 0),
            }
        )

    lines = [
        f"# {report_title(scope)}",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"Scope: `{scope}`",
        "",
        "## Source Files",
        "",
    ]
    lines.extend(f"- `{path.relative_to(ROOT)}`" for path in source_files)
    lines.extend(
        [
            "",
            "## Coverage Metadata",
            "",
            md_table(["Variation", "Source File", "Date Range", "Rows", "Validated", "Missing", "Failed"], metadata_rows),
            "",
            "## Overall Ranking",
            "",
            md_table(headers, ranking),
            "",
            "## Variation Tables",
        ]
    )
    for variation in variations:
        rows = sorted(
            model_rows(variation),
            key=lambda row: (row["_sort_net_per_day"], row["_sort_accuracy"], row["_sort_evaluated"]),
            reverse=True,
        )
        lines.extend(["", f"### {variation['label']}", ""])
        lines.append(md_table(headers, rows) if rows else "_No model-level predictions available for this variation._")
    lines.extend(
        [
            "",
            "## Metric Definitions",
            "",
            "- Accuracy is wins divided by evaluated predictions.",
            "- Accuracy Last 240 is the same calculation using only the latest 240 evaluated predictions for that variation and model; if fewer than 240 exist, it uses all available evaluated predictions.",
            "- Accuracy Last 480 is the same calculation using only the latest 480 evaluated predictions for that variation and model; if fewer than 480 exist, it uses all available evaluated predictions.",
            "- Accuracy Delta From 50% is the absolute percentage-point difference between overall accuracy and 50%.",
            "- Net wins is wins minus losses.",
            "- Net wins per day is net wins divided by unique UTC dates with evaluated predictions.",
            "- Missing and failed rows are included in coverage metadata and excluded from model metric calculations.",
            "- Old/new report scopes split rows by the pre-forward-backfill latest prediction timestamp for each variation.",
        ]
    )
    destination = output_path or report_path(scope)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def generate_all_reports() -> list[Path]:
    paths = [
        generate_report("old"),
        generate_report("new"),
        generate_report("all"),
        generate_report("newtest"),
    ]
    REPORT_PATH.write_text(REPORT_ALL_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    paths.append(REPORT_PATH)
    return paths


def main() -> int:
    paths = generate_all_reports()
    for path in paths:
        print(f"Wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
