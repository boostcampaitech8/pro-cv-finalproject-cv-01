#!/usr/bin/env python3
"""Scenario-based benchmark runner for serving/edge/main.py.

Runs each scenario for a fixed duration, stores raw logs, and writes
summary json/csv with key metrics parsed from final logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _parse_vars(var_items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in var_items:
        if "=" not in item:
            raise ValueError(f"Invalid --var format: {item}. expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --var key in: {item}")
        out[key] = value
    return out


def _substitute_vars(value: Any, variables: Dict[str, str]) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            return variables.get(key, match.group(0))

        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, value)

    if isinstance(value, list):
        return [_substitute_vars(v, variables) for v in value]

    if isinstance(value, dict):
        return {k: _substitute_vars(v, variables) for k, v in value.items()}

    return value


def _to_number(value: str) -> float | None:
    if value == "-":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_log_metrics(log_text: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    patterns = {
        "processed_frames": r"처리 프레임:\s*([0-9]+)",
        "captured_pcb": r"포착된 PCB:\s*([0-9]+)",
        "runtime_sec": r"실행 시간:\s*([0-9.]+)초",
        "aggregate_processed_fps": r"총 처리량\(Aggregate Processed FPS\):\s*([0-9.]+)",
        "per_camera_avg_fps": r"카메라당 평균 처리 FPS\(참고\):\s*([0-9.]+)",
        "live_delivery_rate": r"live_delivery_rate=([0-9.]+)%",
        "diagnosis": r"diagnosis:\s*(.+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, log_text)
        if not match:
            continue
        value = match.group(1).strip()
        if key == "diagnosis":
            metrics[key] = value
        elif "." in value:
            metrics[key] = float(value)
        else:
            metrics[key] = int(value)

    eff = re.search(r"delivery\(effective\):\s*([0-9]+)\/([0-9]+)\s*\(([0-9.]+)%\)", log_text)
    if eff:
        metrics["effective_delivery_success"] = int(eff.group(1))
        metrics["effective_delivery_total"] = int(eff.group(2))
        metrics["effective_delivery_rate"] = float(eff.group(3))

    q = re.search(
        r"queue_hwm:\s*frame=([0-9]+),\s*crop=([0-9]+),\s*upload=([0-9]+)\s*\|\s*queue_drop\(frame/crop/upload\)=([0-9]+)\/([0-9]+)\/([0-9]+)",
        log_text,
    )
    if q:
        metrics["queue_hwm_frame"] = int(q.group(1))
        metrics["queue_hwm_crop"] = int(q.group(2))
        metrics["queue_hwm_upload"] = int(q.group(3))
        metrics["queue_drop_frame"] = int(q.group(4))
        metrics["queue_drop_crop"] = int(q.group(5))
        metrics["queue_drop_upload"] = int(q.group(6))

    qd = re.search(r"queue_depth\(now\):\s*frame=([0-9]+),\s*crop=([0-9]+),\s*upload=([0-9]+)", log_text)
    if qd:
        metrics["queue_depth_frame"] = int(qd.group(1))
        metrics["queue_depth_crop"] = int(qd.group(2))
        metrics["queue_depth_upload"] = int(qd.group(3))

    lat = re.search(
        r"latency\(ms\) p50/p95:\s*"
        r"frame_wait=([0-9.-]+)/([0-9.-]+),\s*"
        r"preprocess=([0-9.-]+)/([0-9.-]+),\s*"
        r"crop_wait=([0-9.-]+)/([0-9.-]+),\s*"
        r"inference=([0-9.-]+)/([0-9.-]+),\s*"
        r"upload_wait=([0-9.-]+)/([0-9.-]+),\s*"
        r"upload=([0-9.-]+)/([0-9.-]+),\s*"
        r"e2e=([0-9.-]+)/([0-9.-]+)",
        log_text,
    )
    if lat:
        keys = [
            "frame_wait_p50_ms", "frame_wait_p95_ms",
            "preprocess_p50_ms", "preprocess_p95_ms",
            "crop_wait_p50_ms", "crop_wait_p95_ms",
            "inference_p50_ms", "inference_p95_ms",
            "upload_wait_p50_ms", "upload_wait_p95_ms",
            "upload_p50_ms", "upload_p95_ms",
            "e2e_p50_ms", "e2e_p95_ms",
        ]
        for idx, key in enumerate(keys, start=1):
            value = _to_number(lat.group(idx))
            if value is not None:
                metrics[key] = value

    payload = re.search(r"upload_payload\(kb\) p50/p95:\s*([0-9.-]+)/([0-9.-]+)", log_text)
    if payload:
        p50 = _to_number(payload.group(1))
        p95 = _to_number(payload.group(2))
        if p50 is not None:
            metrics["upload_payload_p50_kb"] = p50
        if p95 is not None:
            metrics["upload_payload_p95_kb"] = p95

    cam_matches = re.findall(
        r"\[(cam_[0-9]+)\]\s*input=([0-9]+)\s*\([0-9.]+fps\),\s*processed=([0-9]+)\s*\([0-9.]+fps\),\s*inference=([0-9]+),\s*drop=([0-9]+)\s*\(([0-9.]+)%\)",
        log_text,
    )
    if cam_matches:
        per_cam = []
        for cam, input_cnt, processed_cnt, infer_cnt, drop_cnt, drop_rate in cam_matches:
            per_cam.append(
                {
                    "camera_id": cam,
                    "input": int(input_cnt),
                    "processed": int(processed_cnt),
                    "inference": int(infer_cnt),
                    "drop": int(drop_cnt),
                    "drop_rate": float(drop_rate),
                }
            )
        metrics["per_camera"] = per_cam
        metrics["max_camera_drop_rate"] = max(item["drop_rate"] for item in per_cam)
        metrics["avg_camera_drop_rate"] = sum(item["drop_rate"] for item in per_cam) / len(per_cam)

    return metrics


def _load_scenarios(path: Path, variables: Dict[str, str], only: List[str]) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    # Accept UTF-8 files with or without BOM.
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, list):
        raise ValueError("Scenario file must be a JSON list")

    scenarios: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Scenario #{idx} must be an object")

        if item.get("enabled", True) is False:
            continue

        scenario = _substitute_vars(item, variables)
        name = scenario.get("name")
        args = scenario.get("args")
        duration_sec = int(scenario.get("duration_sec", 120))
        if not name or not isinstance(name, str):
            raise ValueError(f"Scenario #{idx} missing valid 'name'")
        if not args or not isinstance(args, list):
            raise ValueError(f"Scenario '{name}' missing valid 'args' list")

        if only and name not in only:
            continue

        scenarios.append(
            {
                "name": name,
                "description": scenario.get("description", ""),
                "args": [str(a) for a in args],
                "duration_sec": max(1, duration_sec),
            }
        )

    if not scenarios:
        raise ValueError("No scenario selected. Check --only or scenario file.")

    return scenarios


def _safe_stop(process: subprocess.Popen[Any], grace_sec: float) -> None:
    if process.poll() is not None:
        return

    try:
        process.send_signal(signal.SIGINT)
    except Exception:
        process.terminate()

    try:
        process.wait(timeout=grace_sec)
        return
    except subprocess.TimeoutExpired:
        pass

    process.kill()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _run_scenario(
    scenario: Dict[str, Any],
    runner_tokens: List[str],
    workdir: Path,
    output_root: Path,
    stop_grace_sec: float,
    dry_run: bool,
) -> Dict[str, Any]:
    name = scenario["name"]
    duration_sec = int(scenario["duration_sec"])
    cmd = runner_tokens + scenario["args"]

    scenario_dir = output_root / name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    log_path = scenario_dir / "run.log"

    result: Dict[str, Any] = {
        "name": name,
        "description": scenario.get("description", ""),
        "duration_sec": duration_sec,
        "command": cmd,
        "log_path": str(log_path),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }

    print(f"\n[Benchmark] Scenario: {name}")
    print(f"  cmd: {' '.join(shlex.quote(x) for x in cmd)}")
    print(f"  duration: {duration_sec}s")

    if dry_run:
        result["dry_run"] = True
        return result

    start = time.time()
    with log_path.open("w", encoding="utf-8") as fp:
        process = subprocess.Popen(
            cmd,
            cwd=str(workdir),
            stdout=fp,
            stderr=subprocess.STDOUT,
        )

        while True:
            if process.poll() is not None:
                break
            elapsed = time.time() - start
            if elapsed >= duration_sec:
                break
            time.sleep(0.5)

        _safe_stop(process, grace_sec=stop_grace_sec)
        return_code = process.poll()

    elapsed = time.time() - start
    result["elapsed_sec"] = round(elapsed, 2)
    result["return_code"] = return_code
    result["finished_at"] = datetime.now().isoformat(timespec="seconds")

    try:
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
        metrics = _parse_log_metrics(log_text)
        result.update(metrics)
    except Exception as exc:
        result["parse_error"] = str(exc)

    return result


def _write_outputs(results: List[Dict[str, Any]], output_root: Path) -> None:
    json_path = output_root / "summary.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_columns = [
        "name",
        "duration_sec",
        "elapsed_sec",
        "return_code",
        "processed_frames",
        "captured_pcb",
        "aggregate_processed_fps",
        "per_camera_avg_fps",
        "live_delivery_rate",
        "effective_delivery_rate",
        "inference_p95_ms",
        "upload_wait_p95_ms",
        "upload_p95_ms",
        "e2e_p95_ms",
        "upload_payload_p50_kb",
        "upload_payload_p95_kb",
        "max_camera_drop_rate",
        "queue_hwm_frame",
        "queue_hwm_crop",
        "queue_hwm_upload",
        "queue_drop_frame",
        "queue_drop_crop",
        "queue_drop_upload",
        "diagnosis",
        "log_path",
    ]

    csv_path = output_root / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=csv_columns)
        writer.writeheader()
        for item in results:
            writer.writerow({k: item.get(k, "") for k in csv_columns})

    print(f"\n[Benchmark] summary json: {json_path}")
    print(f"[Benchmark] summary csv : {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Edge benchmark runner")
    parser.add_argument(
        "--scenario-file",
        default="benchmarks/scenarios.sample.json",
        help="Path to scenario json file",
    )
    parser.add_argument(
        "--runner",
        default="python3",
        help="Runner command prefix. Example: 'uv run python' or 'python3'",
    )
    parser.add_argument(
        "--workdir",
        default=".",
        help="Working directory to execute scenarios",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results",
        help="Output directory root",
    )
    parser.add_argument(
        "--var",
        action="append",
        default=[],
        help="Scenario variable substitution (KEY=VALUE)",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Run only matching scenario name (repeatable)",
    )
    parser.add_argument(
        "--stop-grace-sec",
        type=float,
        default=10.0,
        help="Grace period after SIGINT before force kill",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    args = parser.parse_args()

    variables = _parse_vars(args.var)
    scenario_file = Path(args.scenario_file)
    workdir = Path(args.workdir).resolve()
    runner_tokens = shlex.split(args.runner)
    if not runner_tokens:
        raise ValueError("Runner cannot be empty")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    scenarios = _load_scenarios(scenario_file, variables, args.only)

    all_results: List[Dict[str, Any]] = []
    for scenario in scenarios:
        result = _run_scenario(
            scenario=scenario,
            runner_tokens=runner_tokens,
            workdir=workdir,
            output_root=output_root,
            stop_grace_sec=args.stop_grace_sec,
            dry_run=args.dry_run,
        )
        all_results.append(result)

    _write_outputs(all_results, output_root)

    has_failure = any(int(item.get("return_code") or 0) not in (0, 130) for item in all_results if not item.get("dry_run"))
    if has_failure:
        print("[Benchmark] WARNING: one or more scenarios returned non-zero status")
        return 1

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
