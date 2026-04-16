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
├── main.py                  # Entry point (single + batch modes)
├── llm_utils.py             # LLM pipeline (gating, probing, retries)
├── code_utils.py            # Code extraction, sanitization, undefined-name detection
├── optimization_utils.py    # OR-Tools wrapper (solver, variables, objective, constraints)
├── log_utils.py             # Solution printing and status interpretation
├── io_utils.py              # CSV/JSON handling and problem refinement
├── utils.py                 # Helpers (type annotations, nested loops)
├── prompts/                 # System prompts for each pipeline stage
├── problems/                # Generated problem folders (one per problem)
│   └── <dataset>_<id>/
│       ├── user_input.md    # Problem description
│       ├── solution.py      # Generated solver code
│       └── optim_summary.txt# Solver output + objective value
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
| `--report` | Output CSV path | `batch_results.csv` |
| `--problem-timeout` | Max seconds per problem (full pipeline) | 600 |
| `-v` | Verbosity (0=silent, 1=normal, 2=debug) | 1 |

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

---

## Notes

* Classic mode (`-f`) is unchanged and fully compatible
* Batch mode is fully CLI-based
* Dataset format is auto-detected (no configuration needed)
* All results are logged in `batch_results.csv` with full traceability
* Detailed technical changelog: see `STABILISATION_LLoCO.md`
