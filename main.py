import argparse
import os
import sys
import json
import re
import shutil
import subprocess
import csv
import textwrap
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

import llm_utils
import code_utils
import io_utils
from UI.utils import show_logo, SpinnerManager

from wcwidth import wcswidth


PROBLEM_BASE_DIR = "problems"
PROMPT_DIR = "prompts"

# --- Parse objective value from optim_summary.txt ---
OBJ_RE = re.compile(r"Optimization objective value:\s*([+-]?\d+(?:\.\d+)?)")

# ---------------------------------------------------------------------------
# Timeout configuration (seconds)
# ---------------------------------------------------------------------------
# Per-problem timeout for the full pipeline subprocess (LLM calls + solver).
# Increase if your LLM calls are slow.  600 s = 10 min per problem.
BATCH_PROBLEM_TIMEOUT: int = 600

# Timeout for solution.py execution only (solver run).
# Most LP/MIP problems should solve in under 2 min.  Set higher for hard MIPs.
SOLUTION_TIMEOUT: int = 120


# ----------------------------
# Console formatting helpers
# ----------------------------
def _hr(char: str = "─", n: int = 70) -> str:
    return char * n


def _wlen(s: str) -> int:
    return wcswidth(s)


def _box(title: str, lines: List[str]) -> str:
    width = max(_wlen(title), *(_wlen(x) for x in lines)) + 4
    top = "╭" + "─" * width + "╮"
    mid = [f"│  {title}{' ' * (width - 2 - _wlen(title))}│"]
    for ln in lines:
        mid.append(f"│  {ln}{' ' * (width - 2 - _wlen(ln))}│")
    bot = "╰" + "─" * width + "╯"
    return "\n".join([top] + mid + [bot])


def _mode_str(all_flag, single_id, ids_csv, range_pair, start, limit) -> str:
    if all_flag:
        return "--all"
    if single_id is not None:
        return f"--id {single_id}"
    if ids_csv is not None:
        return f"--ids {ids_csv}"
    if range_pair is not None:
        return f"--range {range_pair[0]} {range_pair[1]}"
    # start/limit mode
    s = "" if start is None else f"--start {start} "
    l = "" if limit is None else f"--limit {limit}"
    return (s + l).strip()


def extract_objective_value(optim_summary_path: str) -> Optional[float]:
    if not os.path.exists(optim_summary_path):
        return None
    with open(optim_summary_path, "r", encoding="utf-8") as f:
        txt = f.read()
    m = OBJ_RE.search(txt)
    return float(m.group(1)) if m else None


def find_single_json_file(dataset_dir: str) -> str:
    candidates = [p for p in os.listdir(dataset_dir) if p.lower().endswith(".json")]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly 1 .json in {dataset_dir}, found: {candidates}")
    return os.path.join(dataset_dir, candidates[0])


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


# ---------------------------------------------------------------------------
# Dataset format detection & conversion
# ---------------------------------------------------------------------------

@dataclass
class DatasetFormat:
    """Describes the detected format of a dataset directory."""
    kind: str  # "jsonl" or "subdirs"
    jsonl_path: Optional[str] = None  # set when kind == "jsonl"
    sub_dirs: Optional[List[str]] = None  # sorted list of problem sub-directories


def detect_dataset_format(dataset_dir: str) -> DatasetFormat:
    """
    Detect whether a dataset uses a single JSONL file (IndustryOR style)
    or subdirectories with description.txt + sample.json (ComplexOR / LPWP style).
    """
    json_files = [p for p in os.listdir(dataset_dir) if p.lower().endswith(".json")]
    # A single top-level .json that is a *file* (not a dir) → JSONL format
    if len(json_files) == 1 and os.path.isfile(os.path.join(dataset_dir, json_files[0])):
        return DatasetFormat(kind="jsonl", jsonl_path=os.path.join(dataset_dir, json_files[0]))

    # Look for subdirectories that contain description.txt + sample.json
    sub_dirs = sorted(
        d
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
        and os.path.isfile(os.path.join(dataset_dir, d, "description.txt"))
        and os.path.isfile(os.path.join(dataset_dir, d, "sample.json"))
    )
    if sub_dirs:
        return DatasetFormat(kind="subdirs", sub_dirs=sub_dirs)

    raise ValueError(
        f"Cannot detect dataset format in {dataset_dir}. "
        "Expected either a single .json file or subdirectories with description.txt + sample.json."
    )


def _build_en_question(
    description: str, input_data: dict, code_example: Optional[str] = None
) -> str:
    """
    Combine the textual description with the concrete input parameters
    to form a complete problem statement (like IndustryOR's en_question).
    Optionally includes a code_example.py stub for parameter/return hints.
    """
    parts = [description.strip()]
    parts.append("\n\nInput data:\n```json")
    parts.append(json.dumps(input_data, indent=2, ensure_ascii=False))
    parts.append("```")
    if code_example:
        parts.append(
            "\n\nFunction signature and parameter descriptions "
            "(for reference only — do NOT call this function, "
            "use the optimization library instead):\n```python"
        )
        parts.append(code_example.strip())
        parts.append("```")
    return "\n".join(parts)


def _subdir_id(name: str, index: int) -> int:
    """
    Derive a numeric id from a subdirectory name.
    For LPWP-style names like 'prob_42', extract 42.
    Otherwise use the 1-based index in the sorted list.
    """
    m = re.match(r"^prob_(\d+)$", name)
    if m:
        return int(m.group(1))
    return index


def load_subdir_dataset(
    dataset_dir: str, sub_dirs: List[str]
) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Load a subdirectory-based dataset into the same (id, obj) format used
    by iter_jsonl.  Each obj has: en_question, en_answer, source_dir.
    """
    problems: List[Tuple[int, Dict[str, Any]]] = []

    for idx, dirname in enumerate(sub_dirs, start=1):
        dirpath = os.path.join(dataset_dir, dirname)

        with open(os.path.join(dirpath, "description.txt"), "r", encoding="utf-8") as f:
            description = f.read()

        with open(os.path.join(dirpath, "sample.json"), "r", encoding="utf-8") as f:
            samples = json.load(f)

        if not isinstance(samples, list) or len(samples) == 0:
            continue

        sample = samples[0]
        input_data = sample.get("input", {})
        output_val = sample.get("output", None)

        # Read code_example.py if present (parameter/return hints)
        code_example_path = os.path.join(dirpath, "code_example.py")
        code_example = None
        if os.path.isfile(code_example_path):
            with open(code_example_path, "r", encoding="utf-8") as f:
                code_example = f.read()

        # Normalise output to a single scalar string (like IndustryOR's en_answer)
        if isinstance(output_val, list) and len(output_val) == 1:
            en_answer = str(output_val[0])
        elif isinstance(output_val, list):
            en_answer = str(output_val)
        elif output_val is not None:
            en_answer = str(output_val)
        else:
            en_answer = None

        pid = _subdir_id(dirname, idx)
        problems.append(
            (
                pid,
                {
                    "en_question": _build_en_question(description, input_data, code_example),
                    "en_answer": en_answer,
                    "source_dir": dirname,
                    "id": pid,
                },
            )
        )

    return problems


def safe_rmtree(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)


def select_problem_ids(
    all_flag: bool,
    single_id: Optional[int],
    ids_csv: Optional[str],
    range_pair: Optional[List[int]],
    start: Optional[int],
    limit: Optional[int],
    available_ids: List[int],
) -> List[int]:
    # Enforce one selection mode only
    modes = sum(
        [
            1 if all_flag else 0,
            1 if single_id is not None else 0,
            1 if ids_csv is not None else 0,
            1 if range_pair is not None else 0,
            1 if (start is not None or limit is not None) else 0,
        ]
    )
    if modes == 0:
        raise ValueError("Choose one: --all OR --id OR --ids OR --range OR --start/--limit.")
    if modes > 1:
        raise ValueError("Only one selection mode allowed at a time.")

    ids_sorted = sorted(available_ids)

    if all_flag:
        return ids_sorted

    if single_id is not None:
        return [single_id]

    if ids_csv is not None:
        return [int(x.strip()) for x in ids_csv.split(",") if x.strip()]

    if range_pair is not None:
        a, b = range_pair
        if a > b:
            a, b = b, a
        return [i for i in ids_sorted if a <= i <= b]

    # start/limit mode
    s = start if start is not None else ids_sorted[0]
    selected = [i for i in ids_sorted if i >= s]
    return selected[:limit] if limit is not None else selected


def _extract_error_line(stderr_text: str) -> Optional[str]:
    """
    Extract a short, user-friendly error line from stderr.
    Example: "ModuleNotFoundError: No module named 'data'"
    """
    if not stderr_text:
        return None
    lines = [ln.strip() for ln in stderr_text.splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return None


def batch_run(
    dataset_name: str,
    data_root: str = "datasets",
    problems_root: str = PROBLEM_BASE_DIR,
    all_flag: bool = False,
    single_id: Optional[int] = None,
    ids_csv: Optional[str] = None,
    range_pair: Optional[List[int]] = None,
    start: Optional[int] = None,
    limit: Optional[int] = None,
    overwrite: bool = False,
    tolerance: float = 1e-6,
    verbosity: int = 1,
    dry_run: bool = False,
    report_path: str = "batch_results.csv",
    problem_timeout: int = BATCH_PROBLEM_TIMEOUT,
):
    dataset_dir = os.path.join(data_root, dataset_name)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")

    fmt = detect_dataset_format(dataset_dir)

    problems: List[Tuple[int, Dict[str, Any]]] = []
    if fmt.kind == "jsonl":
        for line_no, obj in iter_jsonl(fmt.jsonl_path):
            if "en_question" not in obj:
                continue
            pid = obj.get("id", line_no)
            problems.append((pid, obj))
        format_label = fmt.jsonl_path
    else:
        problems = load_subdir_dataset(dataset_dir, fmt.sub_dirs)
        format_label = f"{len(fmt.sub_dirs)} subdirectories"

    if not problems:
        raise ValueError(f"No problems found in {dataset_dir}")

    available_ids = [pid for pid, _ in problems]
    selected_ids = select_problem_ids(
        all_flag=all_flag,
        single_id=single_id,
        ids_csv=ids_csv,
        range_pair=range_pair,
        start=start,
        limit=limit,
        available_ids=available_ids,
    )

    by_id: Dict[int, Dict[str, Any]] = {pid: obj for pid, obj in problems}

    results: List[Dict[str, Any]] = []

    if verbosity > 0:
        print(
            _box(
                "🚀 LLoCO Batch Runner",
                [
                    f"Dataset: {dataset_name}",
                    f"Source:  {format_label}",
                    f"Mode:    {_mode_str(all_flag, single_id, ids_csv, range_pair, start, limit)}",
                    f"Root:    {problems_root}",
                    f"Total:   {len(selected_ids)} problem(s)",
                    f"Timeout: {problem_timeout}s per problem",
                ],
            )
        )
        print()

    total = len(selected_ids)
    main_script = os.path.abspath(__file__)

    for idx, pid in enumerate(selected_ids, start=1):
        if pid not in by_id:
            results.append(
                {
                    "id": pid,
                    "folder": None,
                    "source_dir": None,
                    "expected": None,
                    "objective": None,
                    "ok": None,
                    "status": "ID_NOT_FOUND",
                }
            )
            continue

        obj = by_id[pid]
        problem_folder = f"{dataset_name}_{pid}"
        problem_path = os.path.join(problems_root, problem_folder)

        if overwrite:
            safe_rmtree(problem_path)

        os.makedirs(problem_path, exist_ok=True)

        user_input_path = os.path.join(problem_path, "user_input.md")
        with open(user_input_path, "w", encoding="utf-8") as f:
            f.write(obj["en_question"].strip() + "\n")

        expected = obj.get("en_answer", None)

        if dry_run:
            results.append(
                {
                    "id": pid,
                    "folder": problem_folder,
                    "source_dir": obj.get("source_dir"),
                    "expected": expected,
                    "objective": None,
                    "ok": None,
                    "status": "DRY_RUN",
                }
            )
            if verbosity > 0:
                print(f"[{idx}/{total}] 🧪 DRY_RUN {problem_folder}")
            continue

        if verbosity > 0:
            print(_hr())
            print(f"[{idx}/{total}] ▶ Running {problem_folder}  (timeout {problem_timeout}s)")
            print(_hr())

        cmd = [
            sys.executable,
            main_script,
            "-f",
            problem_folder,
            "--problems-root",
            problems_root,
            "-v",
            str(verbosity),
        ]

        # ---------------------------------------------------------------
        # Run with timeout so a hanging solver or LLM call never blocks
        # the entire batch.
        # ---------------------------------------------------------------
        timed_out = False
        try:
            run = subprocess.run(cmd, timeout=problem_timeout)
            returncode = run.returncode
        except subprocess.TimeoutExpired as e:
            # Kill the entire process group so no zombie solver lingers
            timed_out = True
            returncode = -1
            if verbosity > 0:
                print(
                    f"\n⏱  TIMEOUT — {problem_folder} exceeded {problem_timeout}s, killing.",
                    file=sys.stderr,
                )
            # Write a synthetic optim_summary so the error appears in the CSV
            optim_path_early = os.path.join(problem_path, "optim_summary.txt")
            with open(optim_path_early, "w", encoding="utf-8") as f:
                f.write(
                    f"\n\n[stderr]\n"
                    f"TIMEOUT: problem exceeded {problem_timeout}s and was killed.\n"
                )

        optim_path = os.path.join(problem_path, "optim_summary.txt")
        objective = extract_objective_value(optim_path)

        ok = None
        if timed_out:
            status = "TIMEOUT"
        elif returncode != 0:
            status = "RUN_FAILED"
        elif objective is None:
            status = "NO_OBJECTIVE"
        else:
            status = "OK"
            if expected is not None:
                try:
                    exp_f = float(expected)
                    ok = abs(objective - exp_f) <= tolerance
                except Exception:
                    status = "BAD_EXPECTED_FORMAT"

        short_err = None
        if os.path.exists(optim_path):
            with open(optim_path, "r", encoding="utf-8") as f:
                content = f.read()
            if "\n\n[stderr]\n" in content:
                stderr_part = content.split("\n\n[stderr]\n", 1)[1]
                short_err = _extract_error_line(stderr_part)

        results.append(
            {
                "id": pid,
                "folder": problem_folder,
                "source_dir": obj.get("source_dir"),
                "expected": expected,
                "objective": objective,
                "ok": ok,
                "status": status,
            }
        )

        if verbosity > 0:
            if status == "OK" and ok is True:
                head = "✅ PASSED"
            elif status == "OK" and ok is False:
                head = "❌ FAILED"
            elif status == "OK" and ok is None:
                head = "⚠️ OK (no expected)"
            else:
                head = f"⚠️ {status}"

            source_dir = obj.get("source_dir")
            src_label = f"  ({source_dir})" if source_dir else ""
            print(f"\n{head} — {problem_folder}{src_label}")
            print(f"   ├─ Expected:  {expected}")
            print(f"   ├─ Objective: {objective}")
            if short_err:
                print(f"   └─ Error:     {short_err}")
            else:
                print(f"   └─ Output:    {optim_path}")

            print()

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "folder", "source_dir", "expected", "objective", "ok", "status"])
        w.writeheader()
        w.writerows(results)

    passed = sum(1 for r in results if r.get("ok") is True)
    comparable = sum(1 for r in results if r.get("ok") is not None)
    timeouts = sum(1 for r in results if r.get("status") == "TIMEOUT")

    print(_hr())
    print(f"✅ Report saved: {report_path}")
    print(f"📊 Comparable: {comparable} | Passed: {passed} | Timeouts: {timeouts} | Total: {len(results)}")


def build_api_doc():
    import inspect
    from data import DataLoader

    def document_public_api(cls):
        docs = ["# API documentation for DataLoader class\n"]
        for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith("_"):
                sig = inspect.signature(member)
                doc = inspect.getdoc(member)
                docs.append(f"def {name}{sig}\n\n\"\"\"{doc}\"\"\"\n")
        return "\n".join(docs)

    return document_public_api(DataLoader)


def run_solution(problem_path, code, timeout: int = SOLUTION_TIMEOUT):
    """
    Write solution.py, copy dependencies, and execute it.

    The subprocess is given `timeout` seconds.  If it exceeds that limit
    (e.g. an INFEASIBLE/unbounded solver spinning forever), it is killed and
    a non-zero returncode is returned so the caller propagates RUN_FAILED.
    """
    output_file_path = os.path.join(problem_path, "solution.py")
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(code)
    print()

    shutil.copy("optimization_utils.py", os.path.join(problem_path, "optimization_utils.py"))
    shutil.copy("utils.py", os.path.join(problem_path, "utils.py"))
    shutil.copy("log_utils.py", os.path.join(problem_path, "log_utils.py"))

    if os.path.exists("data.py"):
        shutil.move("data.py", os.path.join(problem_path, "data.py"))

    try:
        result = subprocess.run(
            [sys.executable, "solution.py"],
            cwd=problem_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout,
        )
        return result
    except subprocess.TimeoutExpired:
        # Return a synthetic CompletedProcess so callers don't need to change
        msg = (
            f"TIMEOUT: solution.py exceeded {timeout}s and was killed. "
            f"The solver may have been stuck on an INFEASIBLE or unbounded model."
        )
        print(f"\n⏱  {msg}", file=sys.stderr)
        return subprocess.CompletedProcess(
            args=[sys.executable, "solution.py"],
            returncode=1,
            stdout="",
            stderr=msg,
        )


def run_baseline(problem_path, high_level_description):
    sys_prompt_path = os.path.join(PROMPT_DIR, "system_prompt_baseline.txt")
    code = llm_utils.ask_baseline(sys_prompt_path, high_level_description)
    output_file_path = os.path.join(problem_path, "baseline.py")
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(code)
    results = subprocess.run(["python", "baseline.py"], cwd=problem_path)
    print(results)
    return


def get_high_level_description(problem_path):
    if not os.path.exists(problem_path):
        raise FileNotFoundError(f"Problem directory {problem_path} does not exist.")

    problem_description_path = os.path.join(problem_path, "user_input.md")
    high_level_description = "# PROBLEM DESCRIPTION\n\n"
    with open(problem_description_path, "r", encoding="utf-8") as f:
        high_level_description += f.read()
    return high_level_description


def main(args):
    problem_dir = args.fname
    problems_root = args.problems_root if getattr(args, "problems_root", None) else PROBLEM_BASE_DIR
    problem_path = os.path.join(problems_root, problem_dir)

    if args.verbosity > 0:
        show_logo()

    high_level_description = get_high_level_description(problem_path)

    if args.baseline:
        run_baseline(problem_path, high_level_description)

    # PROBLEM REFINEMENT
    with SpinnerManager("Analyzing the user inputs ...", active=args.verbosity > 0):
        csv_files_summary, has_csv_file = io_utils.get_csv_files_summary(problem_path)
        refinement_questions = None
        if args.interactive:
            refinement_questions = io_utils.refine_problem_description(
                os.path.join(PROMPT_DIR, "system_prompt_problem_framing.txt"),
                high_level_description + "\n\n" + csv_files_summary,
            )

    if args.interactive and refinement_questions is not None:
        refinement = "# ADDITIONAL DETAILS\n\n"
        for q in refinement_questions["questions"]:
            answer = input(f"🤖 {q['question']} : ")
            refinement += f"Q: {q['question']}\nA: {answer}\n"
    else:
        refinement = ""

    # PROBLEM FORMALIZATION
    with SpinnerManager("Refining and formalizing the problem ...", active=args.verbosity > 0):
        sys_prompt_path = os.path.join(PROMPT_DIR, "system_prompt_problem_summary.txt")
        complete_description = llm_utils.summarize_problem_description(
            sys_prompt_path,
            high_level_description + "\n\n" + csv_files_summary + "\n\n" + refinement,
        )

    # DATA EXTRACTION
    api_doc = ""
    input_files_description = ""

    # Ensure data.py exists even when there is no CSV
    if not has_csv_file:
        with open("data.py", "w", encoding="utf-8") as f:
            f.write("class DataLoader:\n    pass\n")

    if has_csv_file:
        with SpinnerManager("Now I need to extract and prepare the data...", active=args.verbosity > 0):
            sys_prompt_path = os.path.join(PROMPT_DIR, "system_prompt_dataloader.txt")
            input_files_description, has_csv_file = io_utils.convert_file_to_json(
                problem_path, complete_description
            )
            context = (
                complete_description
                + "\n\n" + csv_files_summary
                + "\n\n" + input_files_description
            )

            code_data = llm_utils.data_processing(sys_prompt_path, context)

            # Sanitize data.py too (LLM sometimes outputs fences/explanations)
            code_data, data_fix_log = code_utils.sanitize_python(code_data)
            if args.verbosity > 1 and data_fix_log:
                print("\n🛠️  Applied code repairs to data.py:")
                for fx in data_fix_log:
                    print(f"   - {fx}")
                print()

            with open("data.py", "w", encoding="utf-8") as f:
                f.write(code_data)

            api_doc = build_api_doc()

    # MODEL IMPLEMENTATION
    code_base = code_utils.define_imports()

    with SpinnerManager("Ok time to code the model !", active=args.verbosity > 0):
        sys_prompt_path = os.path.join(PROMPT_DIR, "system_prompt_code_.txt")
        context = complete_description
        if has_csv_file:
            context += "\n\n" + csv_files_summary + "\n\n" + input_files_description

        # -------------------------------------------------------------------
        # Catch LLMPipelineError (network timeout mid-pipeline) so the batch
        # runner marks this problem as RUN_FAILED instead of crashing.
        # -------------------------------------------------------------------
        try:
            code_optimization = llm_utils.implement_optimization(
                sys_prompt_path, context, code_base, api_doc
            )
        except llm_utils.LLMPipelineError as e:
            # Write a minimal optim_summary so the batch layer can read stderr
            optim_summary_path = os.path.join(problem_path, "optim_summary.txt")
            with open(optim_summary_path, "w", encoding="utf-8") as f:
                f.write(f"\n\n[stderr]\nLLMPipelineError at step '{e.step}': {e.cause}\n")
            print(f"\n⚠️  LLM pipeline failed at step '{e.step}': {e.cause}", file=sys.stderr)
            sys.exit(1)

    # SOLUTION RENDERING
    with SpinnerManager("Almost there ! Just missing the final touch now ...", active=args.verbosity > 0):
        sys_prompt_path = os.path.join(PROMPT_DIR, "system_prompt_sol_print.txt")
        code_summary = code_utils.add_print_summary()
        solution_code = code_optimization + code_summary

        # print_solution is non-critical: if it times out, we skip the pretty
        # printer but keep the rest of the solution intact.
        try:
            code_print = llm_utils.print_solution(
                sys_prompt_path,
                complete_description + "\n\n" + csv_files_summary,
                solution_code,
                api_doc,
            )
        except RuntimeError as e:
            print(f"\n⚠️  print_solution LLM call failed ({e}), skipping pretty printer.",
                  file=sys.stderr)
            code_print = "pass  # print_solution skipped due to LLM error\n"

        # Wrap print_solution to avoid crashing after Solve() / objective print
        wrapped_print = (
            "\n\n# == Print detailed solution (guarded) ==\n"
            "try:\n"
            + textwrap.indent(code_print, "    ")
            + "\nexcept Exception as _e:\n"
            "    print('[print_solution_error]', type(_e).__name__, _e)\n"
        )
        solution_code += wrapped_print

    # Sanitize & auto-repair entire solution before execution
    solution_code, fix_log = code_utils.sanitize_python(solution_code)
    if args.verbosity > 1 and fix_log:
        print("\n🛠️  Applied code repairs before execution:")
        for fx in fix_log:
            print(f"   - {fx}")
        print()

    # Run solution with timeout (catches hung solvers)
    solution_timeout = getattr(args, "solution_timeout", SOLUTION_TIMEOUT)
    optim_summary = run_solution(problem_path, solution_code, timeout=solution_timeout)

    optim_summary_path = os.path.join(problem_path, "optim_summary.txt")
    with open(optim_summary_path, "w", encoding="utf-8") as f:
        f.write(optim_summary.stdout)
        if optim_summary.stderr:
            f.write("\n\n[stderr]\n")
            f.write(optim_summary.stderr)

    if args.verbosity > 1:
        print(optim_summary)

    # IMPORTANT: propagate solution.py failure to batch layer
    if optim_summary.returncode != 0:
        sys.exit(optim_summary.returncode)

    if args.verbosity > 1:
        sys_prompt_path = os.path.join(PROMPT_DIR, "system_prompt_write_report.txt")
        report = llm_utils.write_report(
            sys_prompt_path, complete_description, optim_summary.stdout
        )
        report_path = os.path.join(problem_path, "report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print("\n🤖 Lets see what we got : \n\n")
        print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an optimization problem (single or batch).")
    sub = parser.add_subparsers(dest="cmd")

    # --- Single (compat) ---
    parser.add_argument("-f", "--fname", type=str, required=False)
    parser.add_argument("-v", "--verbosity", type=int, default=1)
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("-b", "--baseline", action="store_true")
    parser.add_argument(
        "--problems-root",
        default=PROBLEM_BASE_DIR,
        help="Root directory containing problem folders for single runs.",
    )
    parser.add_argument(
        "--solution-timeout",
        type=int,
        default=SOLUTION_TIMEOUT,
        help=f"Max seconds for solution.py execution (default: {SOLUTION_TIMEOUT}s).",
    )

    # --- Batch ---
    p_batch = sub.add_parser("batch", help="Run a batch from a dataset JSONL.")
    p_batch.add_argument("--dataset", required=True)
    p_batch.add_argument("--data-root", default="datasets")
    p_batch.add_argument("--problems-root", default=PROBLEM_BASE_DIR)

    p_batch.add_argument("--all", action="store_true", help="Run all problems in dataset")
    p_batch.add_argument("--id", type=int, default=None, help="Run a single problem id")
    p_batch.add_argument("--ids", type=str, default=None, help="Run a list of ids: 3,7,9")
    p_batch.add_argument("--range", nargs=2, type=int, default=None, help="Run an interval: start end")
    p_batch.add_argument("--start", type=int, default=None, help="Start id (used with --limit)")
    p_batch.add_argument("--limit", type=int, default=None, help="Max number of problems to run")

    p_batch.add_argument("--overwrite", action="store_true")
    p_batch.add_argument("--dry-run", action="store_true")
    p_batch.add_argument("--tolerance", type=float, default=1e-6)
    p_batch.add_argument("--report", default=None, help="Output CSV path (default: batch_results_<dataset>_<date>.csv)")
    p_batch.add_argument(
        "--problem-timeout",
        type=int,
        default=BATCH_PROBLEM_TIMEOUT,
        help=f"Max seconds for the full per-problem pipeline (default: {BATCH_PROBLEM_TIMEOUT}s).",
    )

    args = parser.parse_args()

    if args.cmd == "batch":
        report_path = args.report
        if report_path is None:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"batch_results_{args.dataset}_{date_str}.csv"

        batch_run(
            dataset_name=args.dataset,
            data_root=args.data_root,
            problems_root=args.problems_root,
            all_flag=args.all,
            single_id=args.id,
            ids_csv=args.ids,
            range_pair=args.range,
            start=args.start,
            limit=args.limit,
            overwrite=args.overwrite,
            tolerance=args.tolerance,
            verbosity=args.verbosity,
            dry_run=args.dry_run,
            report_path=report_path,
            problem_timeout=args.problem_timeout,
        )
    else:
        if not args.fname:
            parser.error("the following arguments are required: -f/--fname (or use 'batch')")
        main(args)