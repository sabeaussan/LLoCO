import code_utils
import os
import requests
import utils
import time
import ast
import re
import subprocess
import sys
import tempfile
import shutil
from typing import Optional

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _defined_names_in_code(code: str) -> list[str]:
    """
    Retourne une liste de noms "définis" au niveau module (imports, assign, defs),
    utile pour contraindre le LLM à ne pas inventer d'identifiants.
    """
    try:
        tree = ast.parse(code)
    except Exception:
        return []

    defined = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                defined.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name != "*":
                    defined.add(a.asname or a.name)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    defined.add(t.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)
    return sorted(defined)


# ---------------------------------------------------------------------------
# Safe block concatenation
# ---------------------------------------------------------------------------

def _safe_join(base: str, snippet: str) -> str:
    """
    Concatenate two Python source blocks with a guaranteed blank-line separator.

    This prevents the class of SyntaxError where a snippet starts immediately
    after the last character of the previous block with no newline, producing
    invalid constructs like:
        solver = define_solver("SCIP")num_products = 3
    """
    base = base.rstrip()
    snippet = snippet.strip()
    if not snippet:
        return base
    return base + "\n\n" + snippet + "\n"


# ---------------------------------------------------------------------------
# Narrative-line filter
# ---------------------------------------------------------------------------
_NARRATIVE_PATTERNS = re.compile(
    r"""
    ^\s*(
        add\s+this\s+code
      | code\s+to\s+insert
      | insert\s+the\s+following
      | place\s+this\s+(after|before|here)
      | here\s+is\s+the\s+code
      | replace\s+the\s+line
      | step\s+\d+\s*[:\-]
      | note\s*:
      | explanation\s*:
      | output\s*:
      | usage\s*:
      | example\s*:
      | instructions?\s*:
      | updated?\s+code\s*:
      | new\s+code\s*:
      | solution\s*:
    )\b
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _strip_narrative_lines(code: str) -> str:
    """
    Remove / comment-out lines that are clearly natural-language instructions
    leaked by the LLM (e.g. "Add this code right after:", "Code to insert:").
    """
    _PY_STARTERS = (
        "def ", "class ", "import ", "from ", "return ", "if ", "elif ",
        "else", "for ", "while ", "try", "except", "with ", "raise ",
        "pass", "break", "continue", "yield", "async ", "await ", "@",
        "lambda ",
    )

    cleaned = []
    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            cleaned.append(line)
            continue
        if (
            any(stripped.startswith(kw) for kw in _PY_STARTERS)
            or "=" in stripped
            or "(" in stripped
            or stripped[0].isdigit()
        ):
            cleaned.append(line)
            continue
        if _NARRATIVE_PATTERNS.match(stripped):
            cleaned.append("# [narrative removed] " + stripped)
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Non-integer value sanitizer  (fix for Codex P2)
# ---------------------------------------------------------------------------

# Matches any use of dtype=np.int64 / dtype=np.int32 / dtype=int in generated code
_INT_DTYPE_PATTERN = re.compile(
    r"dtype\s*=\s*(np\.int(?:64|32|16|8)?|int)\b",
)

# Matches .astype(int) or .astype(np.int64) etc.
_ASTYPE_INT_PATTERN = re.compile(
    r"\.astype\s*\(\s*(int|np\.int(?:64|32|16|8)?)\s*\)",
)

# Matches to_numpy(dtype=np.int64) etc.
_TO_NUMPY_INT_PATTERN = re.compile(
    r"to_numpy\s*\(\s*dtype\s*=\s*(np\.int(?:64|32|16|8)?|int)\s*\)",
)


def _patch_int_casts(code: str) -> str:
    """
    Replace unsafe integer casts in LLM-generated data loading code with
    float64 equivalents, then add a strict integer validation guard.

    Rationale (Codex P2):
    - CSV data may contain decimal values (e.g. 2.9).
    - Casting directly to int silently truncates (2.9 → 2).
    - This corrupts optimization coefficients and produces wrong results.

    Strategy:
    1. Replace dtype=np.int64 / .astype(int) / to_numpy(dtype=int)
       with their float64 equivalent.
    2. Inject a validation helper at the top of the snippet that raises
       ValueError if any column claimed to be integer actually contains
       non-integer float values.

    The validation helper is injected only once (idempotent).
    """
    # Step 1: replace int dtypes with float64
    code = _INT_DTYPE_PATTERN.sub("dtype=np.float64", code)
    code = _ASTYPE_INT_PATTERN.sub(".astype(np.float64)", code)
    code = _TO_NUMPY_INT_PATTERN.sub("to_numpy(dtype=np.float64)", code)

    # Step 2: inject validation helper if not already present
    guard_marker = "# __lloco_int_guard__"
    if guard_marker not in code:
        guard = (
            f"{guard_marker}\n"
            "import numpy as _np\n"
            "def _validate_integer_column(arr, name='column'):\n"
            "    \"\"\"Raise ValueError if arr contains non-integer float values.\"\"\"\n"
            "    arr = _np.asarray(arr, dtype=float)\n"
            "    if not _np.all(arr == _np.floor(arr)):\n"
            "        bad = arr[arr != _np.floor(arr)]\n"
            "        raise ValueError(\n"
            "            f'Column {name!r} contains non-integer values '\n"
            "            f'(e.g. {bad[:3].tolist()}). '\n"
            "            f'Use float variables or clean the data first.'\n"
            "        )\n"
            "    return arr.astype(int)\n\n"
        )
        code = guard + code

    return code


# ---------------------------------------------------------------------------
# INFEASIBLE detection helpers
# ---------------------------------------------------------------------------

_INFEASIBLE_PATTERNS = re.compile(
    r"(INFEASIBLE|infeasible|No\s+solution\s+exists|MPSOLVER_INFEASIBLE"
    r"|model\s+is\s+infeasible|problem\s+is\s+infeasible)",
    re.IGNORECASE,
)

_ZERO_OBJ_PATTERN = re.compile(
    r"Optimization objective value:\s*0\.0",
    re.IGNORECASE,
)


def _is_infeasible(stdout: str, stderr: str) -> bool:
    """Return True if the solver output indicates an INFEASIBLE model."""
    combined = (stdout or "") + (stderr or "")
    return bool(_INFEASIBLE_PATTERNS.search(combined))


# Patterns that indicate a runtime ValueError / data dependency error in
# generated code (e.g. "must be defined before", "not defined", "missing").
_RUNTIME_VALUE_ERROR_PATTERNS = re.compile(
    r"(ValueError|must\s+be\s+defined\s+before|not\s+defined\s+before"
    r"|has\s+not\s+been\s+initialized|missing\s+required\s+data"
    r"|cannot\s+be\s+used\s+before)",
    re.IGNORECASE,
)


def _is_runtime_value_error(stdout: str, stderr: str) -> bool:
    """
    Return True if the probe output indicates a runtime ValueError caused by
    missing data / wrong ordering (e.g. objective referencing D before it is
    defined).  These errors come from the LLM-generated code itself, not the
    solver, so they require regenerating the offending step.
    """
    combined = (stdout or "") + (stderr or "")
    return bool(_RUNTIME_VALUE_ERROR_PATTERNS.search(combined))


def _probe_solution(code: str, timeout: int = 60) -> tuple[str, str, int]:
    """
    Run `code` in an isolated temp directory and return (stdout, stderr, returncode).
    Dependencies (optimization_utils.py, utils.py, log_utils.py, data.py) are
    copied from the current working directory if they exist.
    """
    tmpdir = tempfile.mkdtemp(prefix="lloco_probe_")
    try:
        for dep in ["optimization_utils.py", "utils.py", "log_utils.py", "data.py"]:
            src = os.path.join(os.getcwd(), dep)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(tmpdir, dep))

        script = os.path.join(tmpdir, "probe_solution.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            result = subprocess.run(
                [sys.executable, "probe_solution.py"],
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                timeout=timeout,
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "PROBE_TIMEOUT", 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Core gating loop
# ---------------------------------------------------------------------------

def _llm_step_with_gating(
    messages,
    code_prefix: str,
    max_tries: int = 4,
    allow_undefined: bool = False,
) -> str:
    """
    Calls the LLM and validates the returned snippet before accepting it.

    Validation layers (applied in order):
      1. _strip_narrative_lines  – remove leaked instruction text
      2. sanitize_python_code    – strip fences, fix unicode, auto-repair
      3. _safe_join              – guarantee \\n\\n separator before compile
      4. compile()               – catches SyntaxError / invalid syntax
      5. find_undefined_names    – catches NameError-class hallucinations
                                   (skipped when allow_undefined=True)
    """
    last_snippet = ""

    for attempt in range(max_tries):
        try:
            raw = openai_ask_requests(messages)
        except RuntimeError:
            raise

        snippet = code_utils.outer_code_parse(raw)
        snippet = _strip_narrative_lines(snippet)
        snippet, _ = code_utils.sanitize_python_code(snippet)
        snippet = utils.add_type_comments(snippet)

        last_snippet = snippet

        combined = _safe_join(code_prefix, snippet)

        try:
            compile(combined, "solution.py", "exec")
        except SyntaxError as e:
            feedback = (
                f"Your previous output has a SyntaxError when combined with the "
                f"existing code: {e}. "
                f"Regenerate ONLY valid Python code. "
                f"Do NOT include markdown fences, natural-language instructions, "
                f"prose, or any text that is not valid Python. "
                f"If you need to explain something, use Python comments (#). "
                f"Make sure your code starts on a new line and is complete."
            )
            messages = messages + [{"role": "system", "content": feedback}]
            continue

        if not allow_undefined:
            undef = code_utils.find_undefined_names(combined)
            if undef:
                feedback = (
                    "Your previous output introduced undefined names: "
                    + ", ".join(undef)
                    + ". Regenerate ONLY the requested Python code. "
                      "Use ONLY already-defined identifiers, or define them "
                      "BEFORE first use. "
                      "Do NOT invent helper functions or variables that are "
                      "not yet defined."
                )
                messages = messages + [{"role": "system", "content": feedback}]
                continue

        return snippet  # ✅ passed all checks

    return last_snippet


# ---------------------------------------------------------------------------
# OpenAI / Akkodis API wrapper
# ---------------------------------------------------------------------------

def openai_ask_requests(
    messages,
    model="gpt-5",
    response_format=None,
    max_tokens=10000,
    timeout=90,
    max_retries=4,
):
    """
    Robust wrapper:
    - support .api_key.txt / api_key.txt / OPENAI_API_KEY
    - avoids JSONDecodeError
    - readable error messages (HTTP + preview)
    - automatic retry (timeout / 429 / 5xx)
    """

    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        for fname in [".api_key.txt", "api_key.txt"]:
            path = os.path.join(os.getcwd(), fname)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    api_key = f.read().strip()
                break

    if not api_key:
        raise RuntimeError(
            "API key not found (.api_key.txt / api_key.txt / OPENAI_API_KEY)."
        )

    url = (
        f"https://cld.akkodis.com/api/openai/deployments/models-{model}"
        f"/chat/completions?api-version=2024-12-01-preview"
    )

    headers = {
        "Content-Type": "application/json",
        "Cache-Control": "no-cache",
        "api-key": api_key,
    }

    data = {
        "max_tokens": max_tokens,
        "messages": messages,
    }

    if response_format is not None:
        data["response_format"] = response_format

    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=timeout)

            if resp.status_code != 200:
                preview = (resp.text or "").strip().replace("\n", " ")[:300]
                raise RuntimeError(f"HTTP {resp.status_code} | {preview}")

            try:
                payload = resp.json()
            except Exception:
                preview = (resp.text or "").strip().replace("\n", " ")[:300]
                raise RuntimeError(f"Non-JSON response | {preview}")

            return payload["choices"][0]["message"]["content"]

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            wait = 2.0 * attempt
            time.sleep(wait)
            continue
        except RuntimeError as e:
            last_err = e
            if "429" in str(e) or "HTTP 5" in str(e):
                time.sleep(2 * attempt)
                continue
            raise

    raise RuntimeError(
        f"OpenAI request failed after {max_retries} retries: {last_err}"
    )


# ---------------------------------------------------------------------------
# Public LLM pipeline functions
# ---------------------------------------------------------------------------

def ask_baseline(prompt_path, hl_desc):
    with open(prompt_path, "r") as f:
        sys_prompt = f.read()
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": hl_desc},
    ]
    raw_response = openai_ask_requests(messages)
    source_code = code_utils.outer_code_parse(raw_response)
    return source_code


def summarize_problem_description(prompt_path, context):
    with open(prompt_path, "r") as f:
        prompt = f.read()
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": context},
    ]
    return openai_ask_requests(messages, model="gpt-5", timeout=180)


def formalize_problem_description(prompt_path, hl_desc):
    with open(prompt_path, "r") as f:
        prompt = f.read()
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"# High-level problem description:\n{hl_desc}"},
    ]
    return openai_ask_requests(messages)


def _define_solver(prompt, ctx):
    code = 'solver = define_solver("SCIP")'
    return "\n\n" + code + "\n"


def print_solution(sys_prompt, context, code, api_doc):
    """
    Asks the LLM to generate the solution visualization block.
    Uses allow_undefined=True: visualization references runtime vars (solver,
    status…) — static undefined-names check would produce false positives.
    compile() check still runs, catching narrative-text SyntaxErrors.
    """
    func_code = code_utils.get_function_code("log_utils.py", ["get_solution_values"])
    code_hint = f"""The user has already implemented the optimization model. \
The code so far is as follows:

```python
{code}
```

You also have access to an API documentation for the DataLoader class which \
loads and processes the input data. Use it when relevant:

```python
{api_doc}
```

Your task is only to implement the solution visualization. \
To do so, you **MUST** use the function provided below:

```python
{func_code}
```

**Strict output rules:**
- Output ONLY valid Python code.
- Do NOT include markdown fences (```), prose, or natural-language instructions.
- Do NOT write lines like "Add this code right after:", "Code to insert:", etc.
- The code will be appended directly after the existing solution; write it so \
it executes correctly in that context without any wrapping.
- If you need to comment something, use Python comments (#).
"""
    messages = [
        {"role": "system", "content": sys_prompt + code_hint},
        {"role": "user", "content": context},
    ]
    source_code = _llm_step_with_gating(
        messages,
        code_prefix=code,
        allow_undefined=True,
    )
    return source_code


# ---------------------------------------------------------------------------
# Gated model-building steps
# ---------------------------------------------------------------------------

def _api_doc_guard(api_doc: str) -> str:
    """Return an IMPORTANT block to inject when api_doc is empty."""
    if not (api_doc or "").strip():
        return (
            "\nIMPORTANT:\n"
            "- DataLoader has NO usable API for this problem (api_doc is empty).\n"
            "- Do NOT use DataLoader at all.\n"
            "- Extract all required numeric data directly from the problem text "
            "and encode it as Python lists/dicts.\n"
        )
    return ""


# Injected into every data-loading prompt to prevent silent int truncation.
_FLOAT_SAFETY_RULE = (
    "\nDATA TYPE SAFETY (CRITICAL):\n"
    "- NEVER cast data to int or np.int64 without first verifying all values\n"
    "  are whole numbers. Silent truncation (e.g. 2.9 → 2) corrupts coefficients.\n"
    "- Use np.float64 as the default dtype for ALL numeric arrays loaded from CSV.\n"
    "- Only use integer dtype when the problem explicitly guarantees integer-only\n"
    "  values AND you have verified this in the data.\n"
    "- If integer values are required by the model (e.g. counts), call\n"
    "  _validate_integer_column(arr, 'column_name') before casting.\n"
)


def _define_variables(sys_prompt, context, code, api_doc):
    func_code = code_utils.get_function_code(
        "optimization_utils.py", ["define_variables"]
    )
    code_hint = f"""The user has already implemented part of the optimization \
model. The code so far is as follows:

```python
{code}
```

You also have access to an API documentation for the DataLoader class which \
loads and processes the input data. Use it when relevant:

```python
{api_doc}
```

Your task is only to implement the decision variable definitions. \
To do so, you **MUST** use the functions provided below:

```python
{func_code}
```

Choose the most appropriate parameters based on the nature of the problem \
(e.g., binary decisions, integer allocations, indexed variables, etc.).

**Strict output rules:**
* Only provide the Python code necessary to define the decision variables.
* Follow the conventions and structure used in the existing implementation.
* Do **not** include objective functions, constraints, or any other parts.
* Output ONLY valid Python code — no markdown fences, no prose, no instructions.
* Your response must start on a new line and be syntactically self-contained.
"""
    code_hint += _api_doc_guard(api_doc)

    messages = [
        {"role": "system", "content": sys_prompt + code_hint},
        {"role": "user", "content": context},
    ]
    source_code = _llm_step_with_gating(messages, code_prefix=code)
    source_code = utils.add_type_comments(source_code)
    return source_code


def _define_objective(sys_prompt, context, code, api_doc):
    func_code = code_utils.get_function_code(
        "optimization_utils.py", ["define_linear_expr", "add_objective"]
    )
    code_hint = f"""The user has already implemented part of the optimization \
model. The code so far is as follows:

```python
{code}
```

You also have access to an API documentation for the DataLoader class which \
loads and processes the input data. Use it when relevant:

```python
{api_doc}
```

Your task is only to implement the objective function definitions. \
To do so, you **MUST** use the functions provided below:

```python
{func_code}
```

**CRITICAL — Optimization direction (maximize vs minimize):**
- Read the problem statement **very carefully** to determine whether to \
MAXIMIZE or MINIMIZE.
- Profit, revenue, production output, number of items → `maximize=True`
- Cost, time, distance, waste, penalty, expense → `maximize=False`
- If the problem says "minimize the total cost", you MUST use `maximize=False`.
- If the problem says "maximize the profit", you MUST use `maximize=True`.
- Do NOT guess — find the explicit objective direction in the problem text.

**Strict output rules:**
- Only provide the Python code necessary to define the objective function.
- Follow the conventions and structure used in the existing implementation.
- Do **not** include constraints, or any other parts of the solution.
- Output ONLY valid Python code — no markdown fences, no prose, no instructions.
- Your response must start on a new line and be syntactically self-contained.
"""
    code_hint += _api_doc_guard(api_doc)

    messages = [
        {"role": "system", "content": sys_prompt + code_hint},
        {"role": "user", "content": context},
    ]
    source_code = _llm_step_with_gating(messages, code_prefix=code)
    return source_code


def _define_constraints(sys_prompt, context, code, api_doc):
    func_code = code_utils.get_function_code(
        "optimization_utils.py", ["define_linear_expr", "add_constraint"]
    )
    code_hint = f"""The user has already implemented part of the optimization \
model. The code so far is as follows:

```python
{code}
```

You also have access to an API documentation for the DataLoader class which \
loads and processes the input data. Use it when relevant:

```python
{api_doc}
```

Your task is only to implement the constraints definitions. \
To do so, you **MUST** use the functions provided below:

```python
{func_code}
```

**CRITICAL — Constraint completeness:**
- Re-read the mathematical formulation carefully and list ALL constraints \
before writing code.
- Include: capacity constraints, demand constraints, variable bounds, \
logical constraints, and any other constraint mentioned in the problem.
- Do NOT skip constraints that seem "obvious" — the solver needs them all.
- Do NOT add constraints that are not in the problem statement.

**Strict vs non-strict inequalities (eps_relax):**
- For strict inequalities (< or >), you MUST set `eps_relax` to a small \
positive value in `add_constraint`. Otherwise the strict inequality is \
silently treated as non-strict (≤ or ≥).
- For integer variables: `eps_relax=1` (e.g., x < y becomes x <= y - 1).
- For continuous variables: `eps_relax=0.0001` or appropriate to the scale.
- For non-strict inequalities (≤, ≥) or equalities (=): leave `eps_relax=0.0`.

**Constraint direction:**
- `operator.le` means ≤ (less than or equal).
- `operator.ge` means ≥ (greater than or equal).
- `operator.eq` means = (equality).
- Double-check the direction of EACH constraint against the problem text.

**Strict output rules:**
- Only provide the Python code necessary to define the constraints.
- Follow the conventions and structure used in the existing implementation.
- Do **not** include any other parts of the solution.
- Output ONLY valid Python code — no markdown fences, no prose, no instructions.
- Your response must start on a new line and be syntactically self-contained.
"""
    code_hint += _api_doc_guard(api_doc)

    messages = [
        {"role": "system", "content": sys_prompt + code_hint},
        {"role": "user", "content": context},
    ]
    source_code = _llm_step_with_gating(messages, code_prefix=code)
    return source_code


def _regenerate_constraints_infeasible(
    sys_prompt: str,
    context: str,
    code_before_constraints: str,
    api_doc: str,
    infeasible_stderr: str,
    attempt: int,
) -> str:
    """
    Ask the LLM to regenerate constraints after an INFEASIBLE result.

    Provides the solver error output as explicit feedback so the LLM
    understands what went wrong and can correct over-constrained formulations.
    """
    func_code = code_utils.get_function_code(
        "optimization_utils.py", ["define_linear_expr", "add_constraint"]
    )

    feedback_block = (
        f"\n\n# ⚠️  INFEASIBLE FEEDBACK (attempt {attempt})\n"
        f"# The previous constraint implementation made the model INFEASIBLE.\n"
        f"# Solver output:\n"
        + "\n".join(f"# {line}" for line in (infeasible_stderr or "").splitlines()[:10])
    )

    code_hint = f"""The user has already implemented part of the optimization \
model. The code so far is as follows:

```python
{code_before_constraints}
```

{feedback_block}

You also have access to an API documentation for the DataLoader class which \
loads and processes the input data. Use it when relevant:

```python
{api_doc}
```

**CRITICAL: The previous constraint implementation caused an INFEASIBLE model.**
This means the constraints were too restrictive or contradictory.

Please regenerate ONLY the constraints with the following corrections:
- Re-read the problem description carefully to identify which constraints \
may be wrong (wrong direction ≤ vs ≥, wrong RHS value, or missing slack).
- Make sure all equality constraints (=) are truly required; prefer \
inequalities (≤ or ≥) where the problem allows.
- Do NOT add constraints that are not explicitly required by the problem.
- Verify sign conventions: a minimization model with cost ≥ 0 constraints \
is more likely feasible than strict equalities.

To implement constraints, you **MUST** use the functions provided below:

```python
{func_code}
```

**Strict output rules:**
- Only provide the Python code necessary to define the constraints.
- Output ONLY valid Python code — no markdown fences, no prose, no instructions.
- Your response must start on a new line and be syntactically self-contained.
"""
    code_hint += _api_doc_guard(api_doc)

    messages = [
        {"role": "system", "content": sys_prompt + code_hint},
        {"role": "user", "content": context},
    ]
    source_code = _llm_step_with_gating(messages, code_prefix=code_before_constraints)
    return source_code


def _regenerate_objective_runtime_error(
    sys_prompt: str,
    context: str,
    code_before_objective: str,
    api_doc: str,
    runtime_stderr: str,
    attempt: int,
) -> str:
    """
    Ask the LLM to regenerate the objective function after a runtime ValueError.

    This handles cases where the LLM generates objective code that references
    data (e.g. a distance matrix D) that has not yet been defined at that point
    in the accumulated code.  The feedback tells the LLM to define all required
    data inline before using it.
    """
    func_code = code_utils.get_function_code(
        "optimization_utils.py", ["define_linear_expr", "add_objective"]
    )

    feedback_block = (
        f"\n\n# ⚠️  RUNTIME ERROR FEEDBACK (attempt {attempt})\n"
        f"# The previous objective implementation raised a RuntimeError/ValueError.\n"
        f"# This usually means the code referenced data or variables that were not\n"
        f"# yet defined at the point where the objective was built.\n"
        f"# Error output:\n"
        + "\n".join(f"# {line}" for line in (runtime_stderr or "").splitlines()[:10])
    )

    code_hint = f"""The user has already implemented part of the optimization \
model. The code so far is as follows:

```python
{code_before_objective}
```

{feedback_block}

You also have access to an API documentation for the DataLoader class which \
loads and processes the input data. Use it when relevant:

```python
{api_doc}
```

**CRITICAL: The previous objective implementation caused a runtime error.**
The code referenced data (e.g. a matrix, a list, a coefficient) that was not \
yet defined.

Please regenerate ONLY the objective function with the following corrections:
- Define ALL required data (matrices, cost vectors, etc.) inline at the top of \
your snippet, BEFORE using them in the objective expression.
- Do NOT assume any variable exists unless it is visible in the "code so far" block.
- If data comes from DataLoader, load it explicitly using the DataLoader API.
- Use ONLY already-defined identifiers, or define them BEFORE first use.

To implement the objective, you **MUST** use the functions provided below:

```python
{func_code}
```

**Strict output rules:**
- Only provide the Python code necessary to define the objective function.
- Output ONLY valid Python code — no markdown fences, no prose, no instructions.
- Your response must start on a new line and be syntactically self-contained.
"""
    code_hint += _api_doc_guard(api_doc)

    messages = [
        {"role": "system", "content": sys_prompt + code_hint},
        {"role": "user", "content": context},
    ]
    source_code = _llm_step_with_gating(messages, code_prefix=code_before_objective)
    return source_code


# ---------------------------------------------------------------------------
# Direction validation (maximize vs minimize)
# ---------------------------------------------------------------------------

_MAXIMIZE_RE = re.compile(r"add_objective\s*\([^)]*maximize\s*=\s*(True|False)", re.IGNORECASE)

_DIRECTION_CHECK_PROMPT = """You are an Operations Research expert. Your ONLY task is to verify whether the optimization direction is correct.

Problem description:
{context}

The generated code uses: `add_objective(..., maximize={current_direction})`

Question: Based on the problem description, should the objective be MAXIMIZED or MINIMIZED?

Answer with EXACTLY one word: MAXIMIZE or MINIMIZE. Nothing else."""


def _check_objective_direction(context: str, obj_snippet: str) -> Optional[str]:
    """
    Ask the LLM to verify the objective direction.
    Returns "MAXIMIZE", "MINIMIZE", or None if unable to determine.
    """
    match = _MAXIMIZE_RE.search(obj_snippet)
    if not match:
        return None  # cannot determine current direction from code

    current_direction = match.group(1)  # "True" or "False"

    messages = [
        {
            "role": "user",
            "content": _DIRECTION_CHECK_PROMPT.format(
                context=context[:3000],  # truncate to save tokens
                current_direction=current_direction,
            ),
        }
    ]

    try:
        answer = openai_ask_requests(messages, max_tokens=10, timeout=30)
        answer = answer.strip().upper()
        if "MAXIMIZE" in answer:
            return "MAXIMIZE"
        if "MINIMIZE" in answer:
            return "MINIMIZE"
        return None
    except Exception:
        return None


def _flip_objective_direction(obj_snippet: str) -> str:
    """Flip maximize=True to maximize=False and vice versa."""
    def _flip(m):
        val = m.group(1)
        new_val = "False" if val == "True" else "True"
        return m.group(0).replace(f"maximize={val}", f"maximize={new_val}")
    return _MAXIMIZE_RE.sub(_flip, obj_snippet)


def implement_optimization(
    prompt_path: str,
    context: str,
    code_base: str,
    api_doc: str,
    max_infeasible_retries: int = 2,
    max_runtime_retries: int = 2,
    probe_timeout: int = 60,
) -> str:
    """
    Builds the full optimization code incrementally.

    Pipeline:
      1. Solver definition  (deterministic)
      2. Decision variables (gated LLM)
      3. Objective function (gated LLM + runtime error retry loop)
      4. Constraints        (gated LLM + INFEASIBLE retry loop)

    After step 3, the accumulated code is probed in a sandbox to catch
    runtime ValueErrors caused by data dependencies (e.g. objective
    referencing a matrix that was not yet defined).

    After step 4, the probe detects INFEASIBLE models and retries constraints.

    Raises LLMPipelineError if a step cannot be completed (e.g. network failure).
    """
    with open(prompt_path, "r") as f:
        sys_prompt = f.read()

    # Step 1 — solver (deterministic)
    solver_snippet = _define_solver("", None)
    code_base = _safe_join(code_base, solver_snippet)

    # Step 2 — decision variables
    try:
        vars_snippet = _define_variables(sys_prompt, context, code_base, api_doc)
    except RuntimeError as e:
        raise LLMPipelineError("variables", e) from e
    code_base = _safe_join(code_base, vars_snippet)

    # Step 3 — objective function (with runtime error retry loop)
    # Snapshot before objective so we can roll back on runtime error.
    code_before_objective = code_base
    last_runtime_stderr = ""

    for runtime_attempt in range(max_runtime_retries + 1):
        if runtime_attempt == 0:
            try:
                obj_snippet = _define_objective(
                    sys_prompt, context, code_before_objective, api_doc
                )
            except RuntimeError as e:
                raise LLMPipelineError("objective", e) from e
        else:
            print(
                f"\n🔁  Runtime error in objective — regenerating "
                f"(attempt {runtime_attempt}/{max_runtime_retries}) ...",
                file=sys.stderr,
            )
            try:
                obj_snippet = _regenerate_objective_runtime_error(
                    sys_prompt=sys_prompt,
                    context=context,
                    code_before_objective=code_before_objective,
                    api_doc=api_doc,
                    runtime_stderr=last_runtime_stderr,
                    attempt=runtime_attempt,
                )
            except RuntimeError as e:
                raise LLMPipelineError("objective_runtime_retry", e) from e

        obj_candidate = _safe_join(code_before_objective, obj_snippet)

        # Probe: run up to and including the objective (no Solve yet).
        # We only need to check that the code doesn't raise at build time.
        probe_code = obj_candidate + "\n# probe: objective defined OK\n"
        stdout, stderr, rc = _probe_solution(probe_code, timeout=probe_timeout)

        if _is_runtime_value_error(stdout, stderr):
            last_runtime_stderr = stderr or stdout
            continue  # retry objective with feedback

        # No runtime error — accept objective and move on
        code_base = obj_candidate

        # Step 3b — direction validation (maximize vs minimize)
        match = _MAXIMIZE_RE.search(obj_snippet)
        if match:
            current_is_max = match.group(1) == "True"
            verdict = _check_objective_direction(context, obj_snippet)
            if verdict == "MAXIMIZE" and not current_is_max:
                print(
                    "\n🔄  Direction mismatch: code says minimize but problem says maximize. Flipping.",
                    file=sys.stderr,
                )
                obj_snippet = _flip_objective_direction(obj_snippet)
                code_base = _safe_join(code_before_objective, obj_snippet)
            elif verdict == "MINIMIZE" and current_is_max:
                print(
                    "\n🔄  Direction mismatch: code says maximize but problem says minimize. Flipping.",
                    file=sys.stderr,
                )
                obj_snippet = _flip_objective_direction(obj_snippet)
                code_base = _safe_join(code_before_objective, obj_snippet)

        break
    else:
        # Retries exhausted — use last generated objective anyway
        print(
            f"\n⚠️  Runtime error in objective persists after {max_runtime_retries} retries. "
            f"Using last objective; final execution may still fail.",
            file=sys.stderr,
        )
        code_base = _safe_join(code_before_objective, obj_snippet)

    # Snapshot before constraints so we can roll back on INFEASIBLE
    code_before_constraints = code_base

    # Step 4 — constraints (with INFEASIBLE retry loop)
    last_infeasible_stderr = ""

    for infeasible_attempt in range(max_infeasible_retries + 1):
        if infeasible_attempt == 0:
            try:
                cst_snippet = _define_constraints(
                    sys_prompt, context, code_before_constraints, api_doc
                )
            except RuntimeError as e:
                raise LLMPipelineError("constraints", e) from e
        else:
            print(
                f"\n🔁  INFEASIBLE detected — regenerating constraints "
                f"(attempt {infeasible_attempt}/{max_infeasible_retries}) ...",
                file=sys.stderr,
            )
            try:
                cst_snippet = _regenerate_constraints_infeasible(
                    sys_prompt=sys_prompt,
                    context=context,
                    code_before_constraints=code_before_constraints,
                    api_doc=api_doc,
                    infeasible_stderr=last_infeasible_stderr,
                    attempt=infeasible_attempt,
                )
            except RuntimeError as e:
                raise LLMPipelineError("constraints_infeasible_retry", e) from e

        candidate = _safe_join(code_before_constraints, cst_snippet)

        probe_code = candidate + "\nstatus = solver.Solve()\nprint('STATUS:', status)\n"
        stdout, stderr, rc = _probe_solution(probe_code, timeout=probe_timeout)

        if _is_infeasible(stdout, stderr):
            last_infeasible_stderr = stderr or stdout
            continue

        code_base = candidate
        break
    else:
        print(
            f"\n⚠️  INFEASIBLE persists after {max_infeasible_retries} retries. "
            f"Using last constraints; final execution may still fail.",
            file=sys.stderr,
        )
        code_base = _safe_join(code_before_constraints, cst_snippet)

    return code_base


class LLMPipelineError(RuntimeError):
    """Raised when a gated LLM step fails due to a network / API error."""

    def __init__(self, step: str, cause: Exception):
        self.step = step
        self.cause = cause
        super().__init__(f"LLM pipeline failed at step '{step}': {cause}")


def data_processing(prompt_path, context):
    """
    Ask the LLM to generate a DataLoader class from CSV files.

    After code generation, applies _patch_int_casts() to replace any
    unsafe integer dtype casts with float64 + validation guards (Codex P2).
    """
    with open(prompt_path, "r") as f:
        sys_prompt = f.read()

    # Inject float safety rule into the system prompt for data loading
    sys_prompt_patched = sys_prompt + _FLOAT_SAFETY_RULE

    messages = [
        {"role": "system", "content": sys_prompt_patched},
        {"role": "user", "content": context},
    ]
    raw_response = openai_ask_requests(messages)
    source_code = code_utils.outer_code_parse(raw_response)

    # Post-process: replace int casts with float64 + inject validation guard
    source_code = _patch_int_casts(source_code)

    return source_code


def write_report(prompt_path, context, summary):
    with open(prompt_path, "r") as f:
        sys_prompt = f.read()
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": context + summary},
    ]
    raw_response = openai_ask_requests(messages)
    return raw_response