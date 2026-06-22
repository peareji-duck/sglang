#!/usr/bin/env python3
"""Build a compact Markdown table from diffusion evidence artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


METRICS = (
    "tok_per_s",
    "tokens_per_second",
    "input_throughput",
    "output_throughput",
    "total_token_throughput",
    "request_throughput",
    "request_per_s",
    "request_per_s_mean",
    "request_per_s_stdev",
    "completion_token_per_s",
    "completion_token_per_s_mean",
    "completion_token_per_s_stdev",
    "mean_ttft_ms",
    "median_ttft_ms",
    "std_ttft_ms",
    "p99_ttft_ms",
    "ttft_ms_p50",
    "mean_tpot_ms",
    "median_tpot_ms",
    "std_tpot_ms",
    "p99_tpot_ms",
    "tpot_ms_p50",
    "mean_itl_ms",
    "median_itl_ms",
    "std_itl_ms",
    "p99_itl_ms",
    "mean_e2el_ms",
    "median_e2el_ms",
    "std_e2el_ms",
    "p99_e2el_ms",
    "e2e_latency_ms",
    "mean_e2e_latency_ms",
    "median_e2e_latency_ms",
    "std_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "latency_p95_s",
    "latency_p95_s_mean",
    "latency_ms_p99",
    "speedup",
    "speedup_vs_baseline",
)

ROW_COLLECTION_KEYS = ("rows", "summary", "summaries", "results", "metrics")
CONTEXT_KEYS = (
    "model",
    "model_name",
    "model_path",
    "served_model_name",
    "tp",
    "tp_size",
    "tensor_parallel_size",
    "concurrency",
    "concurrent_requests",
    "request_rate",
    "endpoint",
    "mode",
    "status",
)


def iter_json_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    yield from iter_json_rows_from_payload(payload, {})


def iter_json_rows_from_payload(
    payload: Any, inherited_context: dict[str, Any]
) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            yield from iter_json_rows_from_payload(item, inherited_context)
        return

    if not isinstance(payload, dict):
        return

    context = dict(inherited_context)
    context.update(
        {
            key: value
            for key, value in payload.items()
            if key in CONTEXT_KEYS and value not in (None, "")
        }
    )
    row_payload = dict(context)
    row_payload.update(payload)
    row = normalize_row(row_payload)
    if has_metric(row):
        yield row

    for key in ROW_COLLECTION_KEYS:
        child = payload.get(key)
        if isinstance(child, (dict, list)):
            yield from iter_json_rows_from_payload(child, context)


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    latency = normalized.get("latency_s")
    if isinstance(latency, dict) and "latency_p95_s" not in normalized:
        normalized["latency_p95_s"] = latency.get("p95")
    return normalized


def has_metric(row: dict[str, Any]) -> bool:
    return any(row.get(metric) not in (None, "") for metric in METRICS)


def iter_csv_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield dict(row)


def metric_value(row: dict[str, Any], metric: str) -> Any:
    if metric in row:
        return row[metric]
    return None


def first_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


def markdown_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def collect_rows(artifact_dir: Path) -> list[dict[str, Any]]:
    table_rows = []
    for path in sorted(artifact_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".json":
            source_rows = iter_json_rows(path)
        elif suffix == ".csv":
            source_rows = iter_csv_rows(path)
        else:
            continue
        rel_source = str(path.relative_to(artifact_dir))
        for source_row in source_rows:
            for metric in METRICS:
                value = metric_value(source_row, metric)
                if value in (None, ""):
                    continue
                table_rows.append(
                    {
                        "source": rel_source,
                        "model": first_value(
                            source_row,
                            (
                                "model",
                                "model_name",
                                "model_path",
                                "served_model_name",
                                "endpoint",
                                "mode",
                            ),
                        ),
                        "tp": first_value(
                            source_row,
                            ("tp", "tp_size", "tensor_parallel_size"),
                        ),
                        "concurrency": first_value(
                            source_row,
                            ("concurrency", "concurrent_requests", "request_rate"),
                        ),
                        "metric": metric,
                        "value": value,
                    }
                )
    return table_rows


def write_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    lines = [
        "| source | model | tp | concurrency | metric | value |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {source} | {model} | {tp} | {concurrency} | {metric} | {value} |".format(
                source=markdown_escape(row["source"]),
                model=markdown_escape(row["model"]),
                tp=markdown_escape(row["tp"]),
                concurrency=markdown_escape(row["concurrency"]),
                metric=markdown_escape(row["metric"]),
                value=markdown_escape(row["value"]),
            )
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Markdown metric table from JSON and CSV artifacts."
    )
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    rows = collect_rows(artifact_dir)
    rows.sort(
        key=lambda row: (
            str(row["source"]),
            str(row["model"]),
            str(row["tp"]),
            str(row["concurrency"]),
            str(row["metric"]),
        )
    )
    write_markdown(rows, Path(args.output_md))
    print(args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
