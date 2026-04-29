# LLoCO

LLoCO (**LL**M-guided **o**perations and **C**ombinatorial **O**ptimization) is a project that aims to bridge the gap between Operations Research and a non expert user. The system requires a high-level and possibly ambiguous description of the problem as input and returns a report detailing actionable insights based on the solution found. Currently, LLoCO is able to solve Linear Programs, as well as Mixed-Integer Linear Programs. It also supports CSV files as data input.

![LLoCO system architecture](/assets/images/LLoCO_arch.png)

## Install

```bash
pip3 install -r requirements.txt
```

LLoCO uses OpenAI models (GPT-5). Your API key should be saved inside `api_key.txt` (or `.api_key.txt`) at the root of the LLoCO directory, or set as the `OPENAI_API_KEY` environment variable.

---

## Project structure

```text
LLoCO/
├── main.py                  # Entry point (single + batch + bench modes)
├── bench.py                 # Variability benchmark (N runs per temperature)
├── llm_utils.py             # LLM pipeline (gating, probing, retries)
├── code_utils.py            # Code extraction, sanitization, undefined-name detection
├── optimization_utils.py    # OR-Tools wrapper (solver, variables, objective, constraints)
├── log_utils.py             # Solution printing and status interpretation
├── io_utils.py              # CSV/JSON handling and problem refinement
├── utils.py                 # Helpers (type annotations, nested loops)
├── UI/utils.py              # Console formatting (boxes, batch display)
├── prompts/                 # System prompts for each pipeline stage
├── problems/                # Generated problem folders (one per problem)
│   └── <dataset>_<id>/
│       ├── user_input.md    # Problem description
│       ├── solution.py      # Generated solver code
│       └── optim_summary.txt# Solver output + objective value
├── benchmarks/              # Bench mode outputs (one dir per benchmark)
└── datasets/                # Evaluation datasets
    ├── IndustryOR/          # 100 problems (JSONL format)
    ├── ComplexOR/           # 18 problem types (subdirectory format)
    ├── LPWP/                # 288 problems (subdirectory format)
    └── ComplexOR_raw/       # Raw seed data
```

---

## Usage

### Single problem (classic mode)

Create a problem folder with a `user_input.md` file:

```
problems/<problem_name>/user_input.md
```

Run:

```bash
python3 main.py -f problem_name
```

Outputs (inside `problems/<problem_name>/`):

| File | Description |
| ---- | ----------- |
| `optim_summary.txt` | Raw optimization results + objective value |
| `solution.py` | Generated solver code |
| `report.txt` | Human-readable report (verbosity >= 2) |

Additional flags:

```bash
python3 main.py -f problem_name -v 2              # verbose output + report
python3 main.py -f problem_name -i                 # interactive mode (Q&A refinement)
python3 main.py -f problem_name --solution-timeout 60  # solver timeout (default: 120s)
```

---

## Batch mode

Batch mode automatically runs **many problems from a dataset**: generates problem folders, runs LLoCO for each, extracts objective values, compares with ground truth, and produces a CSV report.

### Supported datasets

| Dataset | Format | Problems | Description |
| ------- | ------ | -------- | ----------- |
| **IndustryOR** | Single JSONL file | 100 | Industrial OR problems (LP/MIP) |
| **ComplexOR** | Subdirectories (`description.txt` + `sample.json`) | 18 types | Complex OR problems (diet, knapsack, assignment...) |
| **LPWP** | Subdirectories (`description.txt` + `sample.json`) | 288 | Linear Programming Word Problems |

The dataset format is **detected automatically** -- no extra flags needed.

### Dataset formats

#### Format A: JSONL (IndustryOR)

```
datasets/IndustryOR/IndustryOR.json
```

Each line is a JSON object:

```json
{"id": 1, "en_question": "A factory produces...", "en_answer": "219816.0", "difficulty": "Medium"}
```

#### Format B: Subdirectories (ComplexOR, LPWP)

```
datasets/ComplexOR/
├── diet_problem/
│   ├── description.txt     # Problem description (becomes en_question)
│   ├── sample.json         # {"input": {...}, "output": [value]}
│   └── code_example.py     # Function signature + parameter docs (optional)
├── knapsack_optimization/
└── ...
```

For subdirectory datasets:
- `description.txt` + `sample.json["input"]` are combined into `en_question`
- `sample.json["output"]` becomes `en_answer`
- `code_example.py` (if present) is injected as parameter/return type hints
- The `output` value is **never** included in `user_input.md`

### Batch pipeline

```
datasets/<dataset>/
        │
        ▼
  Detect format (JSONL or subdirs)
        │
        ▼
  Load problems → (id, en_question, en_answer)
        │
        ▼
  Create problems/<dataset>_<id>/user_input.md
        │
        ▼
  python main.py -f <dataset>_<id>
        │
        ├── LLM: formalize → variables → objective → constraints
        ├── Direction check (maximize vs minimize)
        ├── INFEASIBLE retry (up to 2x)
        └── Runtime error retry (up to 2x)
        │
        ▼
  Extract "Optimization objective value:" from optim_summary.txt
        │
        ▼
  Compare with en_answer → batch_results.csv
```

### Batch commands

```bash
# Run ALL problems
python3 main.py batch --dataset IndustryOR --all

# Run ONE problem
python3 main.py batch --dataset IndustryOR --id 12

# Run a RANGE (10 → 25)
python3 main.py batch --dataset IndustryOR --range 10 25

# Run first N
python3 main.py batch --dataset IndustryOR --limit 20

# Run from start + limit
python3 main.py batch --dataset IndustryOR --start 50 --limit 20

# Run specific IDs
python3 main.py batch --dataset IndustryOR --ids 3,7,9,15

# ComplexOR and LPWP work the same way
python3 main.py batch --dataset ComplexOR --all
python3 main.py batch --dataset LPWP --id 0
python3 main.py batch --dataset LPWP --range 0 50
```

### Batch flags

| Flag | Description | Default |
| ---- | ----------- | ------- |
| `--overwrite` | Recreate problem folders from scratch | off |
| `--dry-run` | Only generate folders, don't solve | off |
| `--tolerance` | Float comparison tolerance for pass/fail | 1e-6 |
| `--report` | Output CSV path | `batch_results_<dataset>_<date>.csv` |
| `--problem-timeout` | Max seconds per problem (full pipeline) | 600 |
| `--solution-timeout` | Max seconds for `solution.py` execution | 120 |
| `--model` | LLM model: `gpt-5`, `gpt-4`, `gpt-4o`, `gpt-4o-mini` | `gpt-5` |
| `--temp` | Sampling temperature [0.0–2.0] (only for `gpt-4*` models) | None |
| `-v` | Verbosity (0=silent, 1=normal, 2=debug) | 1 |

### Model & temperature

LLoCO supports several models on the Akkodis OpenAI endpoint:

| Model | Reasoning | Supports `temperature` |
| ----- | --------- | ---------------------- |
| `gpt-5` (default) | ✅ Best for OR | ❌ Locked at default 1.0 |
| `gpt-4` | ❌ | ✅ Any value in [0, 2] |
| `gpt-4o` | ❌ | ✅ Any value in [0, 2] |
| `gpt-4o-mini` | ❌ | ✅ Any value in [0, 2] |

**Models that accept `temperature` — recommendations:**

| Model | Status | Recommendation |
| ----- | ------ | -------------- |
| `gpt-4` | ✅ Available, accepts `temperature` | Most stable and well-known model |
| `gpt-4o` | ✅ Available, accepts `temperature` | Faster and cheaper than `gpt-4` |
| `gpt-4o-mini` | ✅ Available, accepts `temperature` | The fastest / cheapest |
| `gpt-4-turbo`, `gpt-4.1` | ❌ Not deployed | Unavailable on the Akkodis endpoint |

**Use `--temp` to tune determinism / variability:**

```bash
# Reproducible runs (same output every time)
python3 main.py batch --dataset IndustryOR --all --model gpt-4o --temp 0

# Default behaviour (deterministic-ish)
python3 main.py batch --dataset IndustryOR --all --model gpt-4o --temp 0.2

# Creative / variable runs (useful for ensembling / pass@k testing)
python3 main.py batch --dataset IndustryOR --all --model gpt-4o --temp 0.7
```

**Important:** `gpt-5` only accepts the default temperature (1.0). Passing `--temp` with `gpt-5` will trigger an API error.

You can also set defaults via environment variables:

```bash
export LLOCO_MODEL=gpt-4o
export LLOCO_TEMPERATURE=0.5
python3 main.py batch --dataset IndustryOR --id 1
```

### Batch output

```
batch_results.csv
```

| id | folder | source_dir | expected | objective | ok | status |
| -- | ------ | ---------- | -------- | --------- | -- | ------ |
| 1 | IndustryOR_1 | | 219816.0 | 219816.0 | True | OK |
| 7 | ComplexOR_7 | diet_problem | 10.333 | 10.333 | True | OK |
| 0 | LPWP_0 | prob_0 | 1300 | 1300.0 | True | OK |

`source_dir` traces the original subfolder name for subdirectory datasets.

Console summary:

```
Comparable: 98 | Passed: 72 | Timeouts: 2 | Total: 100
```

Possible status values:

| Status | Meaning |
| ------ | ------- |
| `OK` | Pipeline completed, objective extracted |
| `RUN_FAILED` | Pipeline or solver crashed |
| `NO_OBJECTIVE` | Ran but no objective value found |
| `TIMEOUT` | Exceeded `--problem-timeout` |
| `ID_NOT_FOUND` | Requested ID not in dataset |
| `DRY_RUN` | `--dry-run` mode, not executed |

---

## Variability benchmarks (`bench`)

The `bench` subcommand runs the same dataset **N times per temperature** to measure **pipeline stability** and **compare configurations side-by-side** (e.g., `temp=0` vs `temp=0.7` on `gpt-4o`).

It is the right tool to answer questions like:
- "Is the pipeline reproducible at `temp=0`?"
- "How much variability does `temp=0.7` introduce on ComplexOR?"
- "Which configuration produces the highest pass rate?"

### Bench commands

```bash
# Default: 5 runs at temp=0 and 5 runs at temp=0.7 with gpt-4o
python3 main.py bench --dataset ComplexOR --all

# Custom: 10 runs at temp=0.5 only
python3 main.py bench --dataset IndustryOR --range 1 20 --runs 10 --temps 0.5

# Sweep: 3 runs each at three temperatures
python3 main.py bench --dataset LPWP --id 0 --runs 3 --temps 0,0.3,0.7

# Different model
python3 main.py bench --dataset ComplexOR --all --runs 5 --model gpt-4o-mini --temps 0,0.7
```

### Bench flags

| Flag | Description | Default |
| ---- | ----------- | ------- |
| `--dataset` | Dataset name (required) | — |
| `--runs` | Number of independent runs per temperature | 5 |
| `--model` | LLM model | `gpt-4o` |
| `--temps` | Comma-separated list of temperatures (e.g. `0,0.7`). Use `default` for the model's API default. | `0,0.7` |
| `--all` / `--id` / `--ids` / `--range` / `--start` / `--limit` | Problem selection (one mode required, same as batch) | — |
| `--problem-timeout` | Max seconds per problem | 600 |
| `--solution-timeout` | Max seconds for `solution.py` | 120 |
| `--tolerance` | Float tolerance for pass/fail | 1e-6 |
| `--output-root` | Root directory for benchmark results | `benchmarks/` |

### Output structure

Each benchmark creates an isolated, timestamped directory:

```
benchmarks/<dataset>_<model>_<YYYYMMDD_HHMMSS>/
├── config.json                    # full bench config (reproducibility)
├── temp_0.0/
│   ├── run_1.csv                  # raw batch_results from each run
│   ├── ...
│   ├── run_5.csv
│   └── per_problem.csv            # per-problem stats for this temperature
├── temp_0.7/
│   ├── ...
├── overall.csv                    # one row per (model, temperature)
└── summary.txt                    # human-readable report
```

### Per-problem stats (`per_problem.csv`)

For each problem, across the N runs:

| Column | Meaning |
| ------ | ------- |
| `id`, `source_dir`, `expected` | Problem identifiers |
| `n_runs` / `n_ok` / `n_passed` | Total runs / pipeline-completed / objective matches `expected` |
| `pass_rate` | `n_passed / n_runs` |
| `n_unique_objectives` | Distinct objective values produced (1 = fully stable) |
| `mean_objective` / `std_objective` | Mean and stddev of the objective |
| `min_objective` / `max_objective` | Range of objectives |
| `statuses` | `\|`-joined status of each run (e.g. `OK\|OK\|TIMEOUT`) |

### Overall stats (`overall.csv`)

One row per `(model, temperature)`:

| Column | Meaning |
| ------ | ------- |
| `n_problems`, `n_runs`, `total_evaluations` | Grid size |
| `avg_pass_rate` | Mean pass rate across all problems |
| `fully_passing` | Problems passing on **all** runs |
| `always_failing` | Problems failing on **all** runs |
| `fully_stable` | Problems with the **same objective** on every run |
| `avg_unique_objectives` | Mean diversity of objectives per problem |
| `avg_std_objective` | Mean stddev of objective values across problems |

### Human-readable report (`summary.txt`)

The summary contains three tables:

1. **OVERALL** — pass rate / stability metrics per temperature
2. **PER-PROBLEM** — detailed stats for each problem at each temperature
3. **COMPARISON** — side-by-side pass rate per problem when multiple temperatures are tested

Example:

```
==============================================================================
OVERALL — one row per (model, temperature)
==============================================================================
    temp | problems |  runs |  avg pass |  fully ✅ |  fully ❌ |  stable |   avg σ
------------------------------------------------------------------------------------
     0.0 |       18 |     5 |     88.9% |       16 |        2 |      18 |     0.0
     0.7 |       18 |     5 |     72.2% |       11 |        2 |       9 |  3.21
```

`temp=0` shows perfect stability (all problems produce the same objective on every run); `temp=0.7` introduces variability (only 9/18 problems are fully stable) but in this example actually has a lower pass rate.

---

## Pipeline quality features

The LLM pipeline includes several safeguards to improve model quality:

| Feature | Description |
| ------- | ----------- |
| **Gated code generation** | Each LLM step is validated (compile + undefined names check) before acceptance |
| **Narrative filtering** | Leaked instructions ("Add this code after:") are auto-removed |
| **INFEASIBLE retry** | If solver returns INFEASIBLE, constraints are regenerated with feedback (up to 2x) |
| **Runtime error retry** | ValueError/data dependency errors trigger objective regeneration (up to 2x) |
| **Direction validation** | Post-objective LLM probe verifies maximize vs minimize matches the problem |
| **Type annotations** | `define_variables` calls get numpy shape/type comments for downstream steps |
| **Worked example** | System prompt includes a complete LP example showing the correct API pattern |
| **Timeout protection** | Both full pipeline and solver execution have configurable timeouts |

---

## Tips

### Re-run only failed problems

```bash
# Extract failed IDs from CSV, then re-run
python3 main.py batch --dataset IndustryOR --ids 3,7,11 --overwrite
```

### Test generation only (no LLM calls)

```bash
python3 main.py batch --dataset ComplexOR --all --dry-run
```

### Faster runs with reduced timeout

```bash
python3 main.py batch --dataset LPWP --all --problem-timeout 300
```

### Test variability — use the `bench` subcommand

For systematic variability testing, prefer the `bench` subcommand (see the
"Variability benchmarks" section above). It handles N runs per temperature,
aggregation, and side-by-side comparison automatically:

```bash
python3 main.py bench --dataset IndustryOR --id 7 --runs 5 --temps 0,0.7
```

The manual loop below is only useful when you want full control over each run:

```bash
for i in 1 2 3 4 5; do
  python3 main.py batch --dataset IndustryOR --id 7 \
      --model gpt-4o --temp 0.7 --overwrite \
      --report batch_run_${i}.csv
done
```

### Compare models side-by-side

```bash
python3 main.py batch --dataset IndustryOR --all --model gpt-5 --report results_gpt5.csv
python3 main.py batch --dataset IndustryOR --all --model gpt-4o --temp 0 --report results_gpt4o.csv
```

---

## Notes

* Classic mode (`-f`) is unchanged and fully compatible
* Batch mode is fully CLI-based
* Dataset format is auto-detected (no configuration needed)
* All results are logged in `batch_results.csv` with full traceability
* Detailed technical changelog: see `STABILISATION_LLoCO.md`
