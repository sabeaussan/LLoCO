# LLoCO

LLoCO (**LL**M-guided **o**perations and **C**ombinatorial **O**ptimization) is a project that aims to bridge the gap between Operations Research and a non expert user. The system requires a high-level and possibly ambiguous description of the problem as input and returns a report detailing actionnable insights based on the solution found. Currently, LLoCO is able to solve Linear Programs, as well as Mixed-Integer Linear Programs. It also only supports CSV files as data input.  An architecure overview of the system is available ![LLoCO system architecture](/assets/images/LLoCO_arch.png)

## How to use
The projects is structured as follows:
```text
.
└── LLoCo/
    ├── main.py
    ├── prompts/
    ├── ...
    ├── problems/
    │   └── problem_name/
    │       ├── user_input.md
    │       ├── data.csv
    │       └── some_more_data.csv
    └── datasets/
        └── eval_datasets
```

A `user_input.md` file should be written inside the corresponding `problem_name` folder under `problems`. To run LLoCO:
`python3 main.py -f problem_name`
More arguments can be found using:
`python3 main.py -h`
A file summarizing raw optimization results can be found under `problem_name` as `optim_summary.txt`. In the same folder, the full report is written inside `report.txt`.

## Install
To use LLoCO, you first need to install the required libraries:
`pip3 install -r requirements.txt`

LLoCO uses openai models to run : o3 and GPT-5. Your api key should be saved inside `api_key.txt` at the root the LLoCO directory to be found.

# Usage

## Single problem (classic mode)

Create:

```
problems/<problem_name>/user_input.md
```

Run:

```bash
python3 main.py -f problem_name
```

Outputs:

* `optim_summary.txt`
* `report.txt`

---

# Batch Mode (NEW)

Batch mode automatically runs **many problems from a dataset**.

It:

* generates problem folders automatically
* runs LLoCO for each problem
* extracts objective values
* compares with ground truth
* produces a CSV evaluation report

---

# Dataset format

Inside:

```
datasets/<dataset_name>/
```

You must have **one `.json` file (JSONL format)**.

Each line:

```json
{
  "id": 1,
  "en_question": "Problem description ...",
  "en_answer": 123.45
}
```

Fields:

| Field       | Description                            |
| ----------- | -------------------------------------- |
| en_question | problem description                    |
| en_answer   | expected optimal objective value       |
| id          | optional (line number used if missing) |

---

# Batch pipeline (diagram)

```
datasets/IndustryOR/problems.json
            │
            ▼
   Read each JSON line
            │
            ▼
Create problems/IndustryOR_<id>/
            │
            ├── user_input.md
            ▼
python main.py -f IndustryOR_<id>
            │
            ▼
Generate optim_summary.txt
            │
            ▼
Extract:
"Optimization objective value:"
            │
            ▼
Compare with en_answer
            │
            ▼
Write batch_results.csv
```

---

# Batch commands

## Run ALL problems

```bash
python3 main.py batch --dataset IndustryOR --all
```

## Run ONE problem

```bash
python3 main.py batch --dataset IndustryOR --id 12
```

## Run a RANGE (x → y)

```bash
python3 main.py batch --dataset IndustryOR --range 10 25
```

## Run first N

```bash
python3 main.py batch --dataset IndustryOR --limit 20
```

## Run from start + limit

```bash
python3 main.py batch --dataset IndustryOR --start 50 --limit 20
```

## Run specific IDs

```bash
python3 main.py batch --dataset IndustryOR --ids 3,7,9,15
```

---

# Optional flags

| Flag        | Description                      |
| ----------- | -------------------------------- |
| --overwrite | recreate problem folders         |
| --dry-run   | only generate folders (no solve) |
| --tolerance | float comparison tolerance       |
| --report    | output CSV path                  |
| -v          | verbosity                        |

---

# Output

After batch:

```
batch_results.csv
```

Example:

| id | folder       | expected | objective | ok    | status |
| -- | ------------ | -------- | --------- | ----- | ------ |
| 1  | IndustryOR_1 | 120.0    | 120.0     | True  | OK     |
| 2  | IndustryOR_2 | 85.0     | 90.0      | False | OK     |

Console summary:

```
Comparable: 50 | Passed: 47 | Total: 50
```

---

# Tips

### Re-run only failed problems

```bash
python3 main.py batch --dataset IndustryOR --ids 3,7,11
```

### Test generation only

```bash
python3 main.py batch --dataset IndustryOR --all --dry-run
```

---

# Notes

* Classic mode (`-f`) unchanged
* Batch mode is fully CLI based
* No extra dependencies required
* Designed for automated evaluation / benchmarking
