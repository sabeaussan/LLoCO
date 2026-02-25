import ast
import builtins
import os
import re
from typing import List, Optional, Tuple, Set


# ==========================================================
# Extract code blocks
# ==========================================================

def outer_code_parse(contents: str) -> str:
    """
    Robustly extract python code from an LLM response.

    Supports:
    - ```python ... ```
    - ``` ... ```
    - raw text without fences (fallback)
    """
    if not contents:
        return ""

    # Prefer fenced python blocks
    match = re.search(r"```python\s*(.*?)```", contents, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback to any fenced block
    match = re.search(r"```\s*(.*?)```", contents, re.DOTALL)
    if match:
        return match.group(1).strip()

    return contents.strip()


# ==========================================================
# Imports / boilerplate generation
# ==========================================================

def define_imports() -> str:
    # IMPORTANT: keep DataLoader name correct
    return """import sys
import os
from optimization_utils import (
    define_linear_expr,
    add_objective,
    define_variables,
    define_solver,
    add_constraint
)
from log_utils import (
    print_objective_solution_value,
    interpret_status,
    get_solution_values
)
from data import DataLoader
import pandas as pd
import operator
from ortools.linear_solver import pywraplp
import numpy as np
"""


def load_csv(directory: str) -> str:
    files = os.listdir(directory)
    source_code = "\n\n# Load input files\n"
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(directory, file)
            fname = file_path.split("/")[-1]
            var_name = fname.split(".")[0]
            source_code += f'{var_name}_df = pd.read_csv("{fname}")\n'
    return source_code


def add_print_summary() -> str:
    return """# Solve the optimization problem
status = solver.Solve()
# == Print summary ==
interpret_status(status)
print_objective_solution_value(solver)
"""


def get_function_code(target_file: str, function_names: List[str]) -> str:
    with open(target_file, "r", encoding="utf-8") as file:
        source_code = file.read()
    tree = ast.parse(source_code)

    result = ""
    lines = source_code.split("\n")
    for name in function_names:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith(name):
                start_line = node.lineno
                end_line = node.end_lineno
                result += "\n".join(lines[start_line - 1 : end_line])
                result += "\n"
    return result


# ==========================================================
# Sanitation: unicode punct + markdown fences + compile-repair
# ==========================================================

_UNICODE_PUNCT_TRANSLATION = str.maketrans({
    # Hyphens / dashes that break Python or copy/paste
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen  (THIS ONE caused your SyntaxError)
    "\u2012": "-",  # figure dash
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2212": "-",  # minus sign

    # Quotes
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',

    # Spaces
    "\u00a0": " ",  # nbsp
})


def sanitize_python_code(code: str, max_fixes: int = 8) -> Tuple[str, List[str]]:
    """
    Returns (sanitized_code, fixes_applied)

    - Extract fenced code if present
    - Normalize problematic unicode punctuation
    - Try compile(); if SyntaxError/IndentationError, drop offending line (best effort)
    """
    fixes: List[str] = []
    if not code:
        return "", fixes

    parsed = outer_code_parse(code)
    if parsed != code:
        fixes.append("extracted_code_block")
    code = parsed

    normalized = code.translate(_UNICODE_PUNCT_TRANSLATION)
    if normalized != code:
        fixes.append("normalized_unicode_punct")
    code = normalized

    lines = code.splitlines()

    for _ in range(max_fixes):
        try:
            compile("\n".join(lines) + "\n", "solution.py", "exec")
            final_code = "\n".join(lines).rstrip()
            return (final_code + "\n") if final_code else "", fixes
        except (SyntaxError, IndentationError) as e:
            lineno = getattr(e, "lineno", None)
            msg = getattr(e, "msg", str(e))

            if lineno and 1 <= lineno <= len(lines):
                bad = lines[lineno - 1]
                fixes.append(f"dropped_line:{lineno}:{msg}:{bad[:80]}")
                lines.pop(lineno - 1)
                continue

            # Fallback: drop last non-empty line
            dropped = False
            for j in range(len(lines) - 1, -1, -1):
                if lines[j].strip():
                    fixes.append(f"dropped_line_end:{j+1}:{msg}:{lines[j][:80]}")
                    lines.pop(j)
                    dropped = True
                    break
            if not dropped:
                break

    final_code = "\n".join(lines).rstrip()
    return (final_code + "\n") if final_code else "", fixes


# ==========================================================
# Undefined-name detection (for gating)
# ==========================================================

class _DefUseVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.defined: Set[str] = set()
        self.used: Set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            self.defined.add(a.asname or a.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for a in node.names:
            if a.name != "*":
                self.defined.add(a.asname or a.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.defined.add(node.name)
        # args are defined inside function scope; we don't need them at module scope
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.defined.add(node.name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for t in node.targets:
            self._collect_targets(t)
        self.generic_visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._collect_targets(node.target)
        if node.value:
            self.generic_visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        self._collect_targets(node.target)
        self.generic_visit(node.iter)
        for b in node.body:
            self.visit(b)
        for b in node.orelse:
            self.visit(b)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars:
                self._collect_targets(item.optional_vars)
            self.visit(item.context_expr)
        for b in node.body:
            self.visit(b)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.defined.add(node.name)
        for b in node.body:
            self.visit(b)

    def visit_Comprehension(self, node: ast.comprehension) -> None:
        # Python internal; handled via ListComp/SetComp/DictComp/GenExp traversal
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        for gen in node.generators:
            self._collect_targets(gen.target)
            self.visit(gen.iter)
            for if_ in gen.ifs:
                self.visit(if_)
        self.visit(node.elt)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        for gen in node.generators:
            self._collect_targets(gen.target)
            self.visit(gen.iter)
            for if_ in gen.ifs:
                self.visit(if_)
        self.visit(node.elt)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        for gen in node.generators:
            self._collect_targets(gen.target)
            self.visit(gen.iter)
            for if_ in gen.ifs:
                self.visit(if_)
        self.visit(node.key)
        self.visit(node.value)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        for gen in node.generators:
            self._collect_targets(gen.target)
            self.visit(gen.iter)
            for if_ in gen.ifs:
                self.visit(if_)
        self.visit(node.elt)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.used.add(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.defined.add(node.id)

    def _collect_targets(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self.defined.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._collect_targets(elt)
        # ignore attributes/subscripts (x[i] = ..., obj.a = ...)


def find_undefined_names(code: str, extra_allowed: Optional[List[str]] = None) -> List[str]:
    """
    Returns list of names that are used (Load) but not defined at module scope.
    Used to gate LLM outputs (avoid NameError).
    """
    extra_allowed = extra_allowed or []
    try:
        tree = ast.parse(code)
    except Exception:
        # If it doesn't parse, we can't do name analysis reliably
        return []

    v = _DefUseVisitor()
    v.visit(tree)

    builtin_names = set(dir(builtins))
    allowed = v.defined | builtin_names | set(extra_allowed)

    # common "globals" we never want to flag
    allowed |= {
        "__name__", "__file__", "__package__", "__doc__", "__builtins__",
    }

    undef = sorted(n for n in v.used if n not in allowed)
    return undef

def sanitize_python(code: str) -> Tuple[str, str]:
    """
    Nettoie un code généré par LLM pour éviter les erreurs de parsing/exécution :
    - enlève markdown fences ```...```
    - supprime les lignes "explicatives" non-python (Task 1:, Explanation:, etc.)
    - remplace certains caractères unicode (hyphens / minus) qui causent SyntaxError
    - garde un log des corrections appliquées
    - tente compile() en fin de sanitize (sans lever, mais loggue si échec)
    """
    log: List[str] = []
    if not code:
        return "", "sanitize_python: empty input"

    raw = code

    # --- 1) Remove markdown fences if present (keep inner) ---
    fence_py = re.search(r"```python\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    fence_any = re.search(r"```\s*(.*?)```", raw, flags=re.DOTALL)
    if fence_py:
        raw = fence_py.group(1)
        log.append("removed ```python fences")
    elif fence_any:
        raw = fence_any.group(1)
        log.append("removed ``` fences")

    # --- 2) Normalize common unicode chars that break python ---
    # U+2010..U+2015 hyphens, U+2212 minus, U+00AD soft hyphen, U+2011 non-breaking hyphen
    repl_map = {
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-",
        "\u2212": "-", "\u00ad": "",
    }
    for bad, good in repl_map.items():
        if bad in raw:
            raw = raw.replace(bad, good)
            log.append(f"replaced unicode {hex(ord(bad))}")

    # --- 3) Drop leading explanatory junk before first plausible python line ---
    lines = raw.splitlines()
    kept: List[str] = []

    def _looks_like_python(ln: str) -> bool:
        s = ln.strip()
        if not s:
            return False
        if s.startswith("#"):
            return True
        starters = (
            "import ", "from ", "def ", "class ", "solver", "data", "for ", "while ", "if ",
            "try:", "with ", "print(", "status", "objective", "model", "x_", "y_", "z_",
        )
        return s.startswith(starters)

    started = False
    for ln in lines:
        if not started:
            if _looks_like_python(ln):
                started = True
                kept.append(ln)
            else:
                # ignore junk line
                if ln.strip():
                    log.append(f"dropped leading junk: {ln.strip()[:60]}")
                continue
        else:
            kept.append(ln)

    # if never started, just keep everything (best effort)
    if not started:
        kept = lines
        log.append("could not find python start; kept all lines")

    # --- 4) Remove common mid-code junk lines (non-comment) ---
    junk_prefixes = (
        "Task ", "Explanation", "Here is", "Below is", "Step ", "Note:", "IMPORTANT:",
        "Final answer", "Output:", "Solution:", "We will",
    )
    cleaned: List[str] = []
    for ln in kept:
        s = ln.strip()
        if s and (not s.startswith("#")) and any(s.startswith(p) for p in junk_prefixes):
            log.append(f"removed junk line: {s[:60]}")
            continue
        cleaned.append(ln)

    out = "\n".join(cleaned).strip() + "\n"

    # --- 5) Validate compile (log only) ---
    try:
        compile(out, "<sanitized>", "exec")
        log.append("compile() OK")
    except SyntaxError as e:
        log.append(f"compile() failed: {e.__class__.__name__}: {e}")

    return out, "\n".join(log)