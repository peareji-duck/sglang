#!/usr/bin/env python3
"""Small SGLang /generate benchmark for dLLM TP-local vocab experiments."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class Endpoint:
    name: str
    url: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def finite_or_none(value: float) -> Optional[float]:
    return value if math.isfinite(value) else None


def safe_component(value: Any) -> str:
    text = "none" if value is None else str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("._") or "none"


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


async def post_generate(
    session: Any,
    endpoint: Endpoint,
    prompt: str,
    max_new_tokens: int,
    timeout_s: float,
    include_output: bool = False,
) -> dict[str, Any]:
    import aiohttp

    body = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0,
        },
    }
    start = time.perf_counter()
    try:
        async with session.post(
            f"{endpoint.url}/generate",
            json=body,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            text = await resp.text()
            elapsed = time.perf_counter() - start
            if resp.status != 200:
                return {
                    "ok": False,
                    "status": resp.status,
                    "latency_s": elapsed,
                    "error": text[:1000],
                }
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return {
                    "ok": False,
                    "status": resp.status,
                    "latency_s": elapsed,
                    "error": text[:1000],
                }
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        elapsed = time.perf_counter() - start
        return {
            "ok": False,
            "status": None,
            "latency_s": elapsed,
            "error": f"{type(exc).__name__}: {exc}",
        }

    output_ids = payload.get("output_ids") or []
    meta = payload.get("meta_info") or {}
    result = {
        "ok": True,
        "status": resp.status,
        "latency_s": elapsed,
        "completion_tokens": int(meta.get("completion_tokens") or len(output_ids)),
        "prompt_tokens": int(meta.get("prompt_tokens") or 0),
        "server_e2e_latency_s": meta.get("e2e_latency"),
        "finish_reason": meta.get("finish_reason"),
    }
    if include_output:
        result["output"] = normalize_output(payload)
        result["output_ids"] = output_ids
    return result


def normalize_output(payload: dict[str, Any]) -> Any:
    for key in ("text", "output", "generated_text"):
        if key in payload:
            return payload[key]
    if "outputs" in payload:
        return payload["outputs"]
    return payload.get("output_ids")


async def run_case(
    endpoint: Endpoint,
    prompt: str,
    max_new_tokens: int,
    concurrency: int,
    requests: int,
    warmup: int,
    timeout_s: float,
    repeat_index: int = 0,
    warmup_requests: Optional[int] = None,
) -> dict[str, Any]:
    import aiohttp

    connector = aiohttp.TCPConnector(limit=max(concurrency * 2, 8))
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(concurrency)

        async def one() -> dict[str, Any]:
            async with sem:
                return await post_generate(
                    session, endpoint, prompt, max_new_tokens, timeout_s
                )

        warmup_rows: list[dict[str, Any]] = []
        warmup_total = warmup * concurrency if warmup_requests is None else warmup_requests
        remaining_warmup = warmup_total
        while remaining_warmup > 0:
            batch_size = min(concurrency, remaining_warmup)
            warmup_rows.extend(await asyncio.gather(*(one() for _ in range(batch_size))))
            remaining_warmup -= batch_size

        warmup_failures = [row for row in warmup_rows if not row["ok"]]
        if warmup_failures:
            return {
                "endpoint": endpoint.name,
                "url": endpoint.url,
                "concurrency": concurrency,
                "requests": requests,
                "warmup": warmup,
                "warmup_requests": warmup_total,
                "max_new_tokens": max_new_tokens,
                "repeat_index": repeat_index,
                "status": "warmup_failed",
                "valid_for_summary": False,
                "ok": 0,
                "failed": 0,
                "warmup_ok": len(warmup_rows) - len(warmup_failures),
                "warmup_failed": len(warmup_failures),
                "wall_s": 0.0,
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "request_per_s": None,
                "completion_token_per_s": None,
                "latency_s": empty_latency_summary(),
                "errors": warmup_failures[:5],
            }

        start = time.perf_counter()
        rows = await asyncio.gather(*(one() for _ in range(requests)))
        wall_s = time.perf_counter() - start

    ok_rows = [row for row in rows if row["ok"]]
    latencies = [float(row["latency_s"]) for row in ok_rows]
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in ok_rows)
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in ok_rows)
    failed = len(rows) - len(ok_rows)
    status = "ok" if failed == 0 and ok_rows else "measured_failed"
    return {
        "endpoint": endpoint.name,
        "url": endpoint.url,
        "concurrency": concurrency,
        "requests": requests,
        "warmup": warmup,
        "warmup_requests": warmup_total,
        "max_new_tokens": max_new_tokens,
        "repeat_index": repeat_index,
        "status": status,
        "valid_for_summary": status == "ok",
        "ok": len(ok_rows),
        "failed": failed,
        "warmup_ok": warmup_total,
        "warmup_failed": 0,
        "wall_s": wall_s,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "request_per_s": finite_or_none(len(ok_rows) / wall_s if wall_s > 0 else float("nan")),
        "completion_token_per_s": finite_or_none(
            completion_tokens / wall_s if wall_s > 0 else float("nan")
        ),
        "latency_s": {
            "mean": finite_or_none(statistics.mean(latencies) if latencies else float("nan")),
            "median": finite_or_none(statistics.median(latencies) if latencies else float("nan")),
            "p90": finite_or_none(percentile(latencies, 90)),
            "p95": finite_or_none(percentile(latencies, 95)),
            "min": finite_or_none(min(latencies) if latencies else float("nan")),
            "max": finite_or_none(max(latencies) if latencies else float("nan")),
        },
        "errors": [row for row in rows if not row["ok"]][:5],
    }


def empty_latency_summary() -> dict[str, None]:
    return {
        "mean": None,
        "median": None,
        "p90": None,
        "p95": None,
        "min": None,
        "max": None,
    }


def parse_endpoints(args: argparse.Namespace) -> list[Endpoint]:
    if args.endpoint:
        endpoints = []
        for item in args.endpoint:
            if "=" not in item:
                raise ValueError(f"--endpoint must be name=url, got {item!r}")
            name, url = item.split("=", 1)
            endpoints.append(Endpoint(name, url.rstrip("/")))
        return endpoints
    return [
        Endpoint("baseline_full_logits", args.baseline_url.rstrip("/")),
        Endpoint("tp_local_vocab_state", args.optimized_url.rstrip("/")),
    ]


def parse_variant_metadata(items: Optional[list[str]]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for item in items or []:
        text = item.strip()
        if not text:
            continue
        if text.startswith("@"):
            loaded = json.loads(Path(text[1:]).read_text())
            if not isinstance(loaded, dict):
                raise ValueError("--variant-metadata @file must contain a JSON object")
            metadata.update(loaded)
            continue
        if text.startswith("{"):
            loaded = json.loads(text)
            if not isinstance(loaded, dict):
                raise ValueError("--variant-metadata JSON must be an object")
            metadata.update(loaded)
            continue
        if "=" not in text:
            raise ValueError(
                "--variant-metadata must be key=value, JSON object, or @json_file"
            )
        key, value = text.split("=", 1)
        metadata[key] = value
    return metadata


def discover_git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(
    args: argparse.Namespace,
    endpoints: list[Endpoint],
    variant_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "created_at": now_iso(),
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "model_name": args.model_name,
        "tp_size": args.tp_size,
        "backend": args.backend,
        "max_running_requests": args.max_running_requests,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "warmup": args.warmup,
        "warmup_requests": args.warmup_requests,
        "repeats": args.repeats,
        "timeout_s": args.timeout_s,
        "endpoints": [{"name": endpoint.name, "url": endpoint.url} for endpoint in endpoints],
        "variant_metadata": variant_metadata,
        "git_commit": discover_git_commit(),
    }


async def check_equivalence(
    baseline: Endpoint,
    optimized: Endpoint,
    prompt: str,
    max_new_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    import aiohttp

    connector = aiohttp.TCPConnector(limit=4)
    async with aiohttp.ClientSession(connector=connector) as session:
        baseline_row, optimized_row = await asyncio.gather(
            post_generate(session, baseline, prompt, max_new_tokens, timeout_s, include_output=True),
            post_generate(session, optimized, prompt, max_new_tokens, timeout_s, include_output=True),
        )
    matched = (
        baseline_row.get("ok")
        and optimized_row.get("ok")
        and baseline_row.get("output") == optimized_row.get("output")
        and baseline_row.get("output_ids") == optimized_row.get("output_ids")
    )
    return {
        "enabled": True,
        "baseline_endpoint": baseline.name,
        "optimized_endpoint": optimized.name,
        "ok": bool(baseline_row.get("ok") and optimized_row.get("ok")),
        "matched": bool(matched),
        "baseline": trim_equivalence_row(baseline_row),
        "optimized": trim_equivalence_row(optimized_row),
    }


def trim_equivalence_row(row: dict[str, Any]) -> dict[str, Any]:
    trimmed = dict(row)
    output = trimmed.get("output")
    if isinstance(output, str) and len(output) > 500:
        trimmed["output"] = output[:500] + "...<truncated>"
    output_ids = trimmed.get("output_ids")
    if isinstance(output_ids, list) and len(output_ids) > 128:
        trimmed["output_ids"] = output_ids[:128] + ["...<truncated>"]
    return trimmed


def summarize_results(
    results: list[dict[str, Any]], endpoints: list[Endpoint]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault((int(row["concurrency"]), str(row["endpoint"])), []).append(row)

    summaries: list[dict[str, Any]] = []
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for (concurrency, endpoint), rows in sorted(grouped.items()):
        valid = [row for row in rows if row.get("valid_for_summary")]
        token_rates = [float(row["completion_token_per_s"]) for row in valid]
        request_rates = [float(row["request_per_s"]) for row in valid]
        latency_p95 = [float(row["latency_s"]["p95"]) for row in valid]
        statuses = sorted({str(row.get("status")) for row in rows})
        summary = {
            "concurrency": concurrency,
            "endpoint": endpoint,
            "status": "ok" if len(valid) == len(rows) and rows else "failed",
            "statuses": ",".join(statuses),
            "repeats": len(rows),
            "valid_repeats": len(valid),
            "failed_repeats": len(rows) - len(valid),
            "completion_token_per_s_mean": mean_or_none(token_rates),
            "completion_token_per_s_stdev": stdev_or_none(token_rates),
            "request_per_s_mean": mean_or_none(request_rates),
            "request_per_s_stdev": stdev_or_none(request_rates),
            "latency_p95_s_mean": mean_or_none(latency_p95),
            "speedup_vs_baseline": None,
        }
        summaries.append(summary)
        by_key[(concurrency, endpoint)] = summary

    baseline_name = endpoints[0].name if endpoints else None
    for summary in summaries:
        baseline = by_key.get((summary["concurrency"], baseline_name))
        base_rate = baseline.get("completion_token_per_s_mean") if baseline else None
        rate = summary.get("completion_token_per_s_mean")
        if (
            baseline
            and baseline.get("status") == "ok"
            and summary.get("status") == "ok"
            and isinstance(base_rate, (int, float))
            and base_rate > 0
            and isinstance(rate, (int, float))
        ):
            summary["speedup_vs_baseline"] = rate / base_rate
    return summaries


def mean_or_none(values: list[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def stdev_or_none(values: list[float]) -> Optional[float]:
    return statistics.stdev(values) if len(values) >= 2 else None


def write_run_root(
    run_root: Path,
    output: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "metadata").mkdir(exist_ok=True)
    (run_root / "tables").mkdir(exist_ok=True)
    (run_root / "plots").mkdir(exist_ok=True)
    (run_root / "manifest.json").write_text(
        json.dumps(output["manifest"], indent=2, sort_keys=True)
    )
    (run_root / "metadata" / "environment.json").write_text(
        json.dumps(output["manifest"], indent=2, sort_keys=True)
    )
    (run_root / "results.json").write_text(json.dumps(output, indent=2, sort_keys=True))
    write_summary_csv(run_root / "summary.csv", summaries)
    write_per_case_csv(run_root / "tables" / "per_case.csv", output)
    write_summary_csv(run_root / "tables" / "summary.csv", summaries)
    write_summary_markdown(run_root / "summary.md", summaries)
    write_report_markdown(run_root / "report.md", output, summaries)
    write_raw_cases(run_root, output)
    write_svg_metric(
        run_root / "plots" / "throughput_by_concurrency.svg",
        summaries,
        "completion_token_per_s_mean",
        "Completion Throughput By Concurrency",
        "tok/s",
    )
    write_svg_metric(
        run_root / "plots" / "speedup_by_concurrency.svg",
        summaries,
        "speedup_vs_baseline",
        "Speedup By Concurrency",
        "x baseline",
    )
    write_svg_metric(
        run_root / "plots" / "p95_latency_by_concurrency.svg",
        summaries,
        "latency_p95_s_mean",
        "P95 Latency By Concurrency",
        "seconds",
    )


def write_per_case_csv(path: Path, output: dict[str, Any]) -> None:
    manifest = output["manifest"]
    fieldnames = [
        "timestamp",
        "model",
        "endpoint",
        "tp",
        "concurrency",
        "repeat",
        "requests",
        "warmup",
        "warmup_requests",
        "max_new_tokens",
        "backend",
        "max_running_requests",
        "status",
        "valid_for_summary",
        "ok",
        "failed",
        "wall_s",
        "request_per_s",
        "completion_token_per_s",
        "latency_mean_s",
        "latency_median_s",
        "latency_p90_s",
        "latency_p95_s",
        "latency_max_s",
        "prompt_tokens",
        "completion_tokens",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in output["results"]:
            latency = row.get("latency_s") or {}
            writer.writerow(
                {
                    "timestamp": manifest["created_at"],
                    "model": manifest.get("model_name"),
                    "endpoint": row.get("endpoint"),
                    "tp": manifest.get("tp_size"),
                    "concurrency": row.get("concurrency"),
                    "repeat": row.get("repeat_index"),
                    "requests": row.get("requests"),
                    "warmup": row.get("warmup"),
                    "warmup_requests": row.get("warmup_requests"),
                    "max_new_tokens": row.get("max_new_tokens"),
                    "backend": manifest.get("backend"),
                    "max_running_requests": manifest.get("max_running_requests"),
                    "status": row.get("status"),
                    "valid_for_summary": row.get("valid_for_summary"),
                    "ok": row.get("ok"),
                    "failed": row.get("failed"),
                    "wall_s": row.get("wall_s"),
                    "request_per_s": row.get("request_per_s"),
                    "completion_token_per_s": row.get("completion_token_per_s"),
                    "latency_mean_s": latency.get("mean"),
                    "latency_median_s": latency.get("median"),
                    "latency_p90_s": latency.get("p90"),
                    "latency_p95_s": latency.get("p95"),
                    "latency_max_s": latency.get("max"),
                    "prompt_tokens": row.get("prompt_tokens"),
                    "completion_tokens": row.get("completion_tokens"),
                }
            )


def write_raw_cases(run_root: Path, output: dict[str, Any]) -> None:
    manifest = output["manifest"]
    model = safe_component(manifest.get("model_name"))
    tp = safe_component(manifest.get("tp_size"))
    for row in output["results"]:
        endpoint = safe_component(row.get("endpoint"))
        concurrency = safe_component(row.get("concurrency"))
        repeat = safe_component(row.get("repeat_index"))
        raw_dir = run_root / "raw" / model / endpoint
        raw_dir.mkdir(parents=True, exist_ok=True)
        stem = f"tp{tp}_c{concurrency}_r{repeat}"
        text = json.dumps(row, indent=2, sort_keys=True)
        (raw_dir / f"{stem}.json").write_text(text)
        (raw_dir / f"{stem}.log").write_text(json.dumps(row, sort_keys=True) + "\n")


def write_report_markdown(
    path: Path,
    output: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> None:
    manifest = output["manifest"]
    lines = [
        "# SGLang SDAR TP-Local Vocab Benchmark",
        "",
        "## Environment",
        "",
        f"- Created: `{manifest.get('created_at_local')}`",
        f"- Model: `{manifest.get('model_name')}`",
        f"- TP size: `{manifest.get('tp_size')}`",
        f"- Backend: `{manifest.get('backend')}`",
        f"- Max running requests: `{manifest.get('max_running_requests')}`",
        f"- Git commit: `{manifest.get('git_commit')}`",
        f"- Max new tokens: `{manifest.get('max_new_tokens')}`",
        f"- Requests per case: `{manifest.get('requests')}`",
        f"- Warmup: `{manifest.get('warmup')}` rounds, warmup requests override `{manifest.get('warmup_requests')}`",
        f"- Repeats: `{manifest.get('repeats')}`",
        "",
        "## Endpoints",
        "",
    ]
    for endpoint in manifest.get("endpoints", []):
        lines.append(f"- `{endpoint['name']}`: `{endpoint['url']}`")
    lines.extend(["", "## Summary", ""])
    lines.extend(summary_markdown_lines(summaries))
    lines.extend(
        [
            "",
            "## Validity Notes",
            "",
            "- A row is valid only when all measured requests succeeded.",
            "- Speedup is omitted when the baseline or comparison endpoint has invalid repeats.",
            "- Use c32 as the headline only after all planned repeats are valid.",
            "",
        ]
    )
    equivalence = output.get("equivalence")
    if equivalence is not None:
        lines.extend(
            [
                "## Equivalence Check",
                "",
                f"- Enabled: `{equivalence.get('enabled')}`",
                f"- OK: `{equivalence.get('ok')}`",
                f"- Matched: `{equivalence.get('matched')}`",
                "",
            ]
        )
    path.write_text("\n".join(lines))


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "concurrency",
        "endpoint",
        "status",
        "statuses",
        "repeats",
        "valid_repeats",
        "failed_repeats",
        "completion_token_per_s_mean",
        "completion_token_per_s_stdev",
        "request_per_s_mean",
        "request_per_s_stdev",
        "latency_p95_s_mean",
        "speedup_vs_baseline",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def write_summary_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(summary_markdown_lines(summaries)) + "\n")


def summary_markdown_lines(summaries: list[dict[str, Any]]) -> list[str]:
    headers = [
        "concurrency",
        "endpoint",
        "status",
        "valid/repeats",
        "tok/s mean",
        "tok/s stdev",
        "p95 latency",
        "speedup",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["concurrency"]),
                    str(row["endpoint"]),
                    str(row["status"]),
                    f"{row['valid_repeats']}/{row['repeats']}",
                    format_optional(row["completion_token_per_s_mean"]),
                    format_optional(row["completion_token_per_s_stdev"]),
                    format_optional(row["latency_p95_s_mean"]),
                    format_optional(row["speedup_vs_baseline"]),
                ]
            )
            + " |"
        )
    return lines


def write_svg_metric(
    path: Path,
    summaries: list[dict[str, Any]],
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    rows = [
        row
        for row in summaries
        if isinstance(row.get(metric), (int, float))
        and row.get("valid_repeats", 0) > 0
    ]
    width = 900
    height = 420
    margin_left = 70
    margin_bottom = 60
    plot_width = width - margin_left - 30
    plot_height = height - 80 - margin_bottom
    if not rows:
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<text x="24" y="36" font-family="Arial" font-size="20">{title}</text>'
            '<text x="24" y="72" font-family="Arial" font-size="14">No valid data</text>'
            "</svg>\n"
        )
        return

    concurrencies = sorted({int(row["concurrency"]) for row in rows})
    endpoints = sorted({str(row["endpoint"]) for row in rows})
    max_value = max(float(row[metric]) for row in rows)
    min_value = min(0.0, min(float(row[metric]) for row in rows))
    span = max(max_value - min_value, 1e-9)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    def x_for(concurrency: int) -> float:
        if len(concurrencies) == 1:
            return margin_left + plot_width / 2
        index = concurrencies.index(concurrency)
        return margin_left + plot_width * index / (len(concurrencies) - 1)

    def y_for(value: float) -> float:
        return 80 + plot_height * (1.0 - ((value - min_value) / span))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<text x="24" y="36" font-family="Arial" font-size="20">{title}</text>',
        f'<text x="24" y="58" font-family="Arial" font-size="12">{ylabel}</text>',
        f'<line x1="{margin_left}" y1="80" x2="{margin_left}" y2="{80 + plot_height}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{80 + plot_height}" x2="{margin_left + plot_width}" y2="{80 + plot_height}" stroke="#333"/>',
    ]
    for concurrency in concurrencies:
        x = x_for(concurrency)
        lines.append(
            f'<text x="{x:.1f}" y="{height - 28}" text-anchor="middle" font-family="Arial" font-size="12">{concurrency}</text>'
        )
    for endpoint_index, endpoint in enumerate(endpoints):
        color = colors[endpoint_index % len(colors)]
        points: list[str] = []
        for concurrency in concurrencies:
            matches = [
                row
                for row in rows
                if int(row["concurrency"]) == concurrency and row["endpoint"] == endpoint
            ]
            if not matches:
                continue
            value = float(matches[0][metric])
            x = x_for(concurrency)
            y = y_for(value)
            points.append(f"{x:.1f},{y:.1f}")
            lines.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}">'
                f"<title>{endpoint} c{concurrency}: {value:.4g}</title></circle>"
            )
        if len(points) >= 2:
            lines.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
        legend_y = 86 + endpoint_index * 20
        lines.append(f'<rect x="{width - 260}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        lines.append(
            f'<text x="{width - 242}" y="{legend_y}" font-family="Arial" font-size="12">{endpoint}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def format_optional(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if value is None:
        return "NA"
    return str(value)


def write_json_output(output_path: Path, output: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-url", default="http://127.0.0.1:18200")
    parser.add_argument("--optimized-url", default="http://127.0.0.1:18201")
    parser.add_argument(
        "--endpoint",
        action="append",
        default=None,
        help=(
            "Benchmark a named endpoint as name=url. Can be repeated. "
            "If omitted, benchmark baseline-url and optimized-url."
        ),
    )
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=None,
        help=(
            "Optional total warmup requests per endpoint/concurrency. "
            "If omitted, warmup remains the legacy number of concurrency-sized rounds."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--timeout-s", type=float, default=300)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--max-running-requests", type=int, default=None)
    parser.add_argument(
        "--variant-metadata",
        action="append",
        default=None,
        help="Repeatable key=value, JSON object, or @json_file metadata for the run.",
    )
    parser.add_argument(
        "--run-root",
        default=None,
        help="Optional directory for manifest.json, results.json, summary.csv, summary.md.",
    )
    parser.add_argument(
        "--check-equivalence",
        action="store_true",
        help=(
            "Send one deterministic request to baseline and optimized endpoints and "
            "record whether outputs match. This is not enabled by default."
        ),
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Solve this briefly: A system serves diffusion language model requests "
            "with tensor parallelism. Explain why avoiding full-vocabulary logits "
            "materialization can improve throughput."
        ),
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.warmup_requests is not None and args.warmup_requests < 0:
        raise ValueError("--warmup-requests must be >= 0")
    endpoints = parse_endpoints(args)
    variant_metadata = parse_variant_metadata(args.variant_metadata)
    manifest = build_manifest(args, endpoints, variant_metadata)
    results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    equivalence: Optional[dict[str, Any]] = None
    output_path = Path(args.output)

    if args.check_equivalence:
        if len(endpoints) < 2:
            raise ValueError("--check-equivalence requires at least two endpoints")
        equivalence = await check_equivalence(
            endpoints[0], endpoints[1], args.prompt, args.max_new_tokens, args.timeout_s
        )
        print(json.dumps({"equivalence": equivalence}, sort_keys=True), flush=True)

    run_root = Path(args.run_root) if args.run_root else None
    for repeat_index in range(args.repeats):
        for concurrency in args.concurrency:
            for endpoint in endpoints:
                case = await run_case(
                    endpoint=endpoint,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    concurrency=concurrency,
                    requests=args.requests,
                    warmup=args.warmup,
                    timeout_s=args.timeout_s,
                    repeat_index=repeat_index,
                    warmup_requests=args.warmup_requests,
                )
                print(json.dumps(case, sort_keys=True), flush=True)
                results.append(case)
                summaries = summarize_results(results, endpoints)
                output = {
                    "created_at": manifest["created_at_local"],
                    "manifest": manifest,
                    "prompt": args.prompt,
                    "equivalence": equivalence,
                    "results": results,
                    "summary": summaries,
                }
                write_json_output(output_path, output)
                if run_root:
                    write_run_root(run_root, output, summaries)

    output = {
        "created_at": manifest["created_at_local"],
        "manifest": manifest,
        "prompt": args.prompt,
        "equivalence": equivalence,
        "results": results,
        "summary": summaries,
    }
    write_json_output(output_path, output)
    if run_root:
        write_run_root(run_root, output, summaries)


if __name__ == "__main__":
    asyncio.run(main())
