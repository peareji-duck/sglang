#!/usr/bin/env python3
"""Compare deterministic diffusion endpoint outputs for PR evidence."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests


CASES = [
    {
        "name": "primary_color",
        "prompt": "Name one primary color.",
        "max_new_tokens": 8,
    },
    {
        "name": "fibonacci_prefix",
        "prompt": "Complete the sequence: 1, 1, 2, 3,",
        "max_new_tokens": 8,
    },
    {
        "name": "comparative_word",
        "prompt": "Write the next word only: cold, colder,",
        "max_new_tokens": 8,
    },
    {
        "name": "single_digit_sum",
        "prompt": "Answer with one digit: 2 + 5 =",
        "max_new_tokens": 8,
    },
    {
        "name": "empty_prompt",
        "prompt": "",
        "max_new_tokens": 8,
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def endpoint_url(base_url: str, path: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def request_body(args: argparse.Namespace, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    if args.mode == "sglang":
        return {
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
            },
        }
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_new_tokens,
    }


def request_path(mode: str) -> str:
    if mode == "sglang":
        return "/generate"
    return "/v1/chat/completions"


def stringify(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def extract_text(payload: Any) -> str | None:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        for item in payload:
            text = extract_text(item)
            if text is not None:
                return text
        return None
    if not isinstance(payload, dict):
        return stringify(payload)

    for key in ("text", "output", "content"):
        if key in payload:
            value = payload[key]
            text = value if isinstance(value, str) else extract_text(value)
            if text is None:
                text = stringify(value)
            if text is not None:
                return text

    message = payload.get("message")
    if isinstance(message, dict):
        text = extract_text(message)
        if text is not None:
            return text

    choices = payload.get("choices")
    if choices is not None:
        text = extract_text(choices)
        if text is not None:
            return text

    outputs = payload.get("outputs")
    if outputs is not None:
        text = extract_text(outputs)
        if text is not None:
            return text

    return None


def call_endpoint(
    name: str,
    base_url: str,
    args: argparse.Namespace,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    url = endpoint_url(base_url, request_path(args.mode))
    body = request_body(args, prompt, max_new_tokens)
    try:
        response = requests.post(url, json=body, timeout=args.timeout_s)
    except requests.RequestException as exc:
        return {
            "endpoint": name,
            "url": url,
            "ok": False,
            "status_code": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    row: dict[str, Any] = {
        "endpoint": name,
        "url": url,
        "ok": response.ok,
        "status_code": response.status_code,
    }
    try:
        payload = response.json()
    except ValueError:
        payload = {"raw_response": response.text}
        row["ok"] = False
        row["error"] = "response was not valid JSON"

    text = extract_text(payload)
    row["text"] = text
    row["normalized_text"] = text.strip() if text is not None else None
    row["payload"] = payload
    if response.ok and text is None:
        row["ok"] = False
        row["error"] = "could not extract text from response"
    elif not response.ok and "error" not in row:
        row["error"] = response.text[:1000]
    return row


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    failures = []
    for case in CASES:
        attempts = run_case(case, args)
        case_failures = [
            attempt_failure(case["name"], attempt)
            for attempt in attempts
            if not attempt["match"]
        ]
        row = {
            "case": case["name"],
            "prompt": case["prompt"],
            "max_new_tokens": case["max_new_tokens"],
            "concurrency": args.concurrency,
            "attempts": attempts,
            "failures": case_failures,
            "match": not case_failures,
        }
        rows.append(row)
        failures.extend(case_failures)

    return {
        "created_at": utc_now(),
        "mode": args.mode,
        "model": args.model,
        "baseline_url": args.baseline_url,
        "optimized_url": args.optimized_url,
        "concurrency": args.concurrency,
        "rows": rows,
        "failures": failures,
        "ok": not failures,
    }


def run_case(case: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    futures = {}
    responses: dict[int, dict[str, dict[str, Any]]] = {
        replica: {} for replica in range(args.concurrency)
    }
    with ThreadPoolExecutor(max_workers=max(2, args.concurrency * 2)) as pool:
        for replica in range(args.concurrency):
            futures[
                pool.submit(
                    call_endpoint,
                    "baseline",
                    args.baseline_url,
                    args,
                    case["prompt"],
                    int(case["max_new_tokens"]),
                )
            ] = (replica, "baseline")
            futures[
                pool.submit(
                    call_endpoint,
                    "optimized",
                    args.optimized_url,
                    args,
                    case["prompt"],
                    int(case["max_new_tokens"]),
                )
            ] = (replica, "optimized")

        for future in as_completed(futures):
            replica, endpoint = futures[future]
            row = future.result()
            row["replica"] = replica
            responses[replica][endpoint] = row

    attempts = []
    for replica in range(args.concurrency):
        baseline = responses[replica].get(
            "baseline",
            {
                "endpoint": "baseline",
                "replica": replica,
                "ok": False,
                "error": "missing baseline response",
            },
        )
        optimized = responses[replica].get(
            "optimized",
            {
                "endpoint": "optimized",
                "replica": replica,
                "ok": False,
                "error": "missing optimized response",
            },
        )
        match = (
            baseline.get("ok")
            and optimized.get("ok")
            and baseline.get("normalized_text") == optimized.get("normalized_text")
        )
        attempts.append(
            {
                "replica": replica,
                "baseline": baseline,
                "optimized": optimized,
                "match": bool(match),
            }
        )
    return attempts


def attempt_failure(case_name: str, attempt: dict[str, Any]) -> dict[str, Any]:
    baseline = attempt["baseline"]
    optimized = attempt["optimized"]
    return {
        "case": case_name,
        "replica": attempt["replica"],
        "baseline_ok": baseline.get("ok"),
        "optimized_ok": optimized.get("ok"),
        "baseline_text": baseline.get("normalized_text"),
        "optimized_text": optimized.get("normalized_text"),
        "baseline_error": baseline.get("error"),
        "optimized_error": optimized.get("error"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and optimized diffusion endpoint outputs."
    )
    parser.add_argument("--baseline-url", required=True)
    parser.add_argument("--optimized-url", required=True)
    parser.add_argument("--mode", choices=("sglang", "openai"), required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model", default="adv-gemma4-stage3")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    payload = run_probe(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(output_path)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
