"""
Variability benchmark runner for LLoCO.

Runs the same dataset N times per (model, temperature) configuration so users
can measure pipeline stability and compare configurations side-by-side.

Outputs:
    benchmarks/<dataset>_<model>_<timestamp>/
        ├── config.json                    # full benchmark config
        ├── temp_<value>/
        │   ├── run_1.csv                  # raw batch_results from each run
        │   ├── run_2.csv
        │   ├── ...
        │   └── per_problem.csv            # per-problem aggregated stats
        ├── overall.csv                    # one row per (temp), aggregated
        └── summary.txt                    # human-readable report
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from UI.utils import box, hr


BENCH_BASE_DIR = "benchmarks"


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    dataset: str
    model: str
    temps: List[float]
    n_runs: int
    selection: Dict[str, Any]      # mode + selection args (--all/--id/--ids/...)
    problem_timeout: int
    solution_timeout: int
    tolerance: float
    overwrite: bool
    verbosity: int
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _read_csv(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(x) -> Optional[float]:
    if x is None or x == "" or x == "None":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def aggregate_per_problem(run_csvs: List[str], tolerance: float = 1e-6) -> List[Dict[str, Any]]:
    """
    Aggregate N run CSVs into per-problem stats.

    Returns a list of dicts, one per problem, with:
      id, source_dir, expected, n_runs, n_passed, pass_rate, n_ok,
      n_unique_objectives, mean_objective, std_objective,
      min_objective, max_objective, statuses
    """
    by_id: Dict[str, Dict[str, Any]] = {}

    for csv_path in run_csvs:
        rows = _read_csv(csv_path)
        for row in rows:
            pid = row.get("id")
            if pid is None or pid == "":
                continue
            entry = by_id.setdefault(pid, {
                "id": pid,
                "source_dir": row.get("source_dir") or "",
                "expected": row.get("expected"),
                "objectives": [],
                "passed": [],
                "statuses": [],
            })
            entry["objectives"].append(_to_float(row.get("objective")))
            entry["passed"].append(row.get("ok") == "True")
            entry["statuses"].append(row.get("status") or "")

    summary: List[Dict[str, Any]] = []
    for pid, entry in sorted(by_id.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        objs = [o for o in entry["objectives"] if o is not None]
        n_runs = len(entry["objectives"])
        n_passed = sum(1 for p in entry["passed"] if p)
        n_ok = sum(1 for s in entry["statuses"] if s == "OK")
        # Round objectives for unique-counting to absorb floating noise
        rounded = [round(o, 6) for o in objs]
        n_unique = len(set(rounded))
        mean = statistics.fmean(objs) if objs else None
        std = statistics.pstdev(objs) if len(objs) > 1 else 0.0 if objs else None
        summary.append({
            "id": pid,
            "source_dir": entry["source_dir"],
            "expected": entry["expected"],
            "n_runs": n_runs,
            "n_ok": n_ok,
            "n_passed": n_passed,
            "pass_rate": round(n_passed / n_runs, 4) if n_runs else 0.0,
            "n_unique_objectives": n_unique,
            "mean_objective": round(mean, 6) if mean is not None else None,
            "std_objective": round(std, 6) if std is not None else None,
            "min_objective": round(min(objs), 6) if objs else None,
            "max_objective": round(max(objs), 6) if objs else None,
            "statuses": "|".join(entry["statuses"]),
        })

    return summary


def aggregate_overall(per_problem: List[Dict[str, Any]], n_runs: int) -> Dict[str, Any]:
    """Aggregate per-problem stats into one overall row for a single config."""
    n_problems = len(per_problem)
    if n_problems == 0:
        return {
            "n_problems": 0, "n_runs": n_runs, "total_evaluations": 0,
            "avg_pass_rate": None, "fully_stable": 0, "fully_passing": 0,
            "always_failing": 0, "avg_unique_objectives": None,
            "avg_std_objective": None,
        }
    pass_rates = [p["pass_rate"] for p in per_problem]
    fully_stable = sum(1 for p in per_problem if p["n_unique_objectives"] == 1)
    fully_passing = sum(1 for p in per_problem if p["pass_rate"] == 1.0)
    always_failing = sum(1 for p in per_problem if p["n_passed"] == 0)
    uniqs = [p["n_unique_objectives"] for p in per_problem]
    stds = [p["std_objective"] for p in per_problem if p["std_objective"] is not None]
    return {
        "n_problems": n_problems,
        "n_runs": n_runs,
        "total_evaluations": n_problems * n_runs,
        "avg_pass_rate": round(statistics.fmean(pass_rates), 4),
        "fully_stable": fully_stable,
        "fully_passing": fully_passing,
        "always_failing": always_failing,
        "avg_unique_objectives": round(statistics.fmean(uniqs), 3),
        "avg_std_objective": round(statistics.fmean(stds), 6) if stds else None,
    }


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def _write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_per_problem_csv(path: str, per_problem: List[Dict[str, Any]]):
    fields = [
        "id", "source_dir", "expected", "n_runs", "n_ok", "n_passed", "pass_rate",
        "n_unique_objectives", "mean_objective", "std_objective",
        "min_objective", "max_objective", "statuses",
    ]
    _write_csv(path, per_problem, fields)


def write_overall_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    fields = ["model", "temperature"] + [
        k for k in rows[0].keys() if k not in ("model", "temperature")
    ]
    _write_csv(path, rows, fields)


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

def _fmt_temp(t: Optional[float]) -> str:
    return "default" if t is None else f"{t}"


def _fmt_pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x*100:.1f}%"


def write_summary_txt(
    path: str,
    config: BenchConfig,
    overall_rows: List[Dict[str, Any]],
    per_problem_by_temp: Dict[str, List[Dict[str, Any]]],
    duration_s: float,
):
    lines: List[str] = []
    lines.append(box(
        "📊 LLoCO Variability Benchmark — Summary",
        [
            f"Dataset:    {config.dataset}",
            f"Model:      {config.model}",
            f"Runs:       {config.n_runs} per temperature",
            f"Temps:      {', '.join(_fmt_temp(t) for t in config.temps)}",
            f"Started:    {config.started_at}",
            f"Duration:   {duration_s:.1f}s",
        ],
    ))
    lines.append("")
    lines.append("=" * 78)
    lines.append("OVERALL — one row per (model, temperature)")
    lines.append("=" * 78)
    if overall_rows:
        header = (
            f"{'temp':>8} | {'problems':>8} | {'runs':>5} | "
            f"{'avg pass':>9} | {'fully ✅':>8} | {'fully ❌':>8} | "
            f"{'stable':>7} | {'avg σ':>10}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for r in overall_rows:
            lines.append(
                f"{_fmt_temp(r.get('temperature')):>8} | "
                f"{r['n_problems']:>8} | {r['n_runs']:>5} | "
                f"{_fmt_pct(r['avg_pass_rate']):>9} | "
                f"{r['fully_passing']:>8} | {r['always_failing']:>8} | "
                f"{r['fully_stable']:>7} | "
                f"{(str(r['avg_std_objective']) if r['avg_std_objective'] is not None else '—'):>10}"
            )
    lines.append("")
    lines.append("Legend:")
    lines.append("  fully ✅  = problems passing on ALL runs")
    lines.append("  fully ❌  = problems failing on ALL runs")
    lines.append("  stable    = problems with the same objective on every run")
    lines.append("  avg σ     = mean standard deviation of the objective across problems")
    lines.append("")

    for temp_label, per_problem in per_problem_by_temp.items():
        lines.append("=" * 78)
        lines.append(f"PER-PROBLEM — temperature={temp_label}")
        lines.append("=" * 78)
        if not per_problem:
            lines.append("  (no data)")
            continue
        h = (
            f"{'id':>5} | {'source':<24} | {'pass':>8} | "
            f"{'unique obj':>10} | {'mean':>14} | {'σ':>10} | {'expected':>10}"
        )
        lines.append(h)
        lines.append("-" * len(h))
        for p in per_problem:
            src = (p.get("source_dir") or "")[:24]
            mean = p["mean_objective"]
            std = p["std_objective"]
            lines.append(
                f"{p['id']:>5} | {src:<24} | "
                f"{p['n_passed']}/{p['n_runs']:<6} | "
                f"{p['n_unique_objectives']:>10} | "
                f"{(str(mean) if mean is not None else '—'):>14} | "
                f"{(str(std) if std is not None else '—'):>10} | "
                f"{(p['expected'] or '—'):>10}"
            )
        lines.append("")

    if len(config.temps) >= 2:
        lines.append("=" * 78)
        lines.append("COMPARISON (per problem, side-by-side pass rate)")
        lines.append("=" * 78)
        # Build problem id → {temp: pass_rate}
        problem_ids = sorted({p["id"] for rows in per_problem_by_temp.values() for p in rows},
                             key=lambda x: int(x) if x.isdigit() else 0)
        temp_labels = list(per_problem_by_temp.keys())
        h = f"{'id':>5} | " + " | ".join(f"{('t='+t):>9}" for t in temp_labels)
        lines.append(h)
        lines.append("-" * len(h))
        index_by_temp = {
            t: {p["id"]: p for p in per_problem_by_temp[t]}
            for t in temp_labels
        }
        for pid in problem_ids:
            cells = []
            for t in temp_labels:
                p = index_by_temp[t].get(pid)
                cells.append(
                    f"{(str(p['n_passed'])+'/'+str(p['n_runs'])):>9}" if p else f"{'—':>9}"
                )
            lines.append(f"{pid:>5} | " + " | ".join(cells))
        lines.append("")

    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_benchmark(
    dataset: str,
    model: str,
    temps: List[Optional[float]],
    n_runs: int,
    selection: Dict[str, Any],
    data_root: str = "datasets",
    problems_root_base: str = "problems",
    problem_timeout: int = 600,
    solution_timeout: int = 120,
    tolerance: float = 1e-6,
    overwrite: bool = True,
    verbosity: int = 1,
    output_root: str = BENCH_BASE_DIR,
):
    """
    Run a full variability benchmark.

    For each temperature in `temps`, runs the dataset `n_runs` times.
    Uses an isolated problems subdir per run to avoid cross-run pollution.
    """
    # Local import to avoid circular dependency with main.py at module load
    from main import batch_run

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_dir = os.path.join(output_root, f"{dataset}_{model}_{timestamp}")
    os.makedirs(bench_dir, exist_ok=True)

    config = BenchConfig(
        dataset=dataset,
        model=model,
        temps=[t for t in temps],
        n_runs=n_runs,
        selection=selection,
        problem_timeout=problem_timeout,
        solution_timeout=solution_timeout,
        tolerance=tolerance,
        overwrite=overwrite,
        verbosity=verbosity,
    )
    with open(os.path.join(bench_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    print(box(
        "📊 LLoCO Variability Benchmark",
        [
            f"Dataset:    {dataset}",
            f"Model:      {model}",
            f"Temps:      {', '.join(_fmt_temp(t) for t in temps)}",
            f"Runs:       {n_runs} per temperature",
            f"Output:     {bench_dir}",
        ],
    ))
    print()

    overall_rows: List[Dict[str, Any]] = []
    per_problem_by_temp: Dict[str, List[Dict[str, Any]]] = {}

    started = time.time()
    total_jobs = len(temps) * n_runs
    job_idx = 0

    for temp in temps:
        temp_label = _fmt_temp(temp)
        temp_dir = os.path.join(bench_dir, f"temp_{temp_label}")
        os.makedirs(temp_dir, exist_ok=True)

        run_csvs: List[str] = []
        for run_idx in range(1, n_runs + 1):
            job_idx += 1
            run_csv = os.path.join(temp_dir, f"run_{run_idx}.csv")
            run_problems_root = os.path.join(temp_dir, f"run_{run_idx}_problems")

            print(hr())
            print(
                f"🧪  Bench job [{job_idx}/{total_jobs}] — "
                f"temp={temp_label}, run {run_idx}/{n_runs}"
            )
            print(hr())

            batch_run(
                dataset_name=dataset,
                data_root=data_root,
                problems_root=run_problems_root,
                all_flag=selection.get("all_flag", False),
                single_id=selection.get("single_id"),
                ids_csv=selection.get("ids_csv"),
                range_pair=selection.get("range_pair"),
                start=selection.get("start"),
                limit=selection.get("limit"),
                overwrite=overwrite,
                tolerance=tolerance,
                verbosity=verbosity,
                dry_run=False,
                report_path=run_csv,
                problem_timeout=problem_timeout,
                solution_timeout=solution_timeout,
                model=model,
                temperature=temp,
            )
            run_csvs.append(run_csv)

        # Aggregate this temperature's runs
        per_problem = aggregate_per_problem(run_csvs, tolerance=tolerance)
        per_problem_by_temp[temp_label] = per_problem

        per_problem_path = os.path.join(temp_dir, "per_problem.csv")
        write_per_problem_csv(per_problem_path, per_problem)

        overall = aggregate_overall(per_problem, n_runs)
        overall["model"] = model
        overall["temperature"] = temp_label
        overall_rows.append(overall)

    duration = time.time() - started

    # Write overall CSV (one row per temp)
    write_overall_csv(os.path.join(bench_dir, "overall.csv"), overall_rows)

    # Write human-readable summary
    summary_text = write_summary_txt(
        os.path.join(bench_dir, "summary.txt"),
        config=config,
        overall_rows=overall_rows,
        per_problem_by_temp=per_problem_by_temp,
        duration_s=duration,
    )

    print()
    print(summary_text)
    print(hr())
    print(f"✅ Benchmark complete — output: {bench_dir}")
    return bench_dir
