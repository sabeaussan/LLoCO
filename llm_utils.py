import code_utils
import os
import requests
import utils
import time
import ast

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


def _llm_step_with_gating(messages, code_prefix: str, max_tries: int = 3) -> str:
    last_raw = None

    for _ in range(max_tries):
        last_raw = openai_ask_requests(messages)
        snippet = code_utils.outer_code_parse(last_raw)

        # 1) sanitize unicode + fences + basic compile repair on the SNIPPET alone
        snippet, _ = code_utils.sanitize_python_code(snippet)

        # 2) add your type comments
        snippet = utils.add_type_comments(snippet)

        combined = code_prefix.rstrip() + "\n\n" + snippet.strip() + "\n"

        # 3) compile check on combined (catches many runtime-ish syntax leftovers)
        try:
            compile(combined, "solution.py", "exec")
        except Exception as e:
            messages = messages + [{
                "role": "system",
                "content": f"Your previous output does not compile when combined with existing code: {e}. "
                           f"Regenerate ONLY valid Python code."
            }]
            continue

        # 4) undefined-names check
        undef = code_utils.find_undefined_names(combined)
        if not undef:
            return snippet

        messages = messages + [{
            "role": "system",
            "content": (
                "Your previous output introduced undefined names: "
                + ", ".join(undef)
                + ". Regenerate ONLY the requested Python code. "
                  "Use ONLY already-defined identifiers, or define them BEFORE first use."
            )
        }]

    # best effort
    return snippet if last_raw else ""


def openai_ask_requests(
    messages,
    model="gpt-5",
    response_format=None,
    max_tokens=10000,
    timeout=90,
    max_retries=4,
):
    """
    Version robuste:
    - support .api_key.txt / api_key.txt / OPENAI_API_KEY
    - évite JSONDecodeError
    - messages d'erreur lisibles (HTTP + preview)
    - retry automatique (timeout / 429 / 5xx)
    """

    # -------- API KEY (fix .api_key.txt) --------
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        for fname in [".api_key.txt", "api_key.txt"]:
            path = os.path.join(os.getcwd(), fname)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    api_key = f.read().strip()
                break

    if not api_key:
        raise RuntimeError("API key not found (.api_key.txt / api_key.txt / OPENAI_API_KEY).")

    # -------- URL --------
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

            # HTTP error
            if resp.status_code != 200:
                preview = (resp.text or "").strip().replace("\n", " ")[:300]
                raise RuntimeError(f"HTTP {resp.status_code} | {preview}")

            # Safe JSON
            try:
                payload = resp.json()
            except Exception:
                preview = (resp.text or "").strip().replace("\n", " ")[:300]
                raise RuntimeError(f"Non-JSON response | {preview}")

            return payload["choices"][0]["message"]["content"]

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            time.sleep(1.5 * attempt)
            continue
        except RuntimeError as e:
            last_err = e
            if "429" in str(e) or "HTTP 5" in str(e):
                time.sleep(2 * attempt)
                continue
            raise

    raise RuntimeError(f"OpenAI request failed after retries: {last_err}")

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
        {"role": "user", "content": context}
    ]
	# Query the LLM
	return openai_ask_requests(messages, model="o4-mini")

def formalize_problem_description(prompt_path, hl_desc):
	with open(prompt_path, "r") as f:
		prompt = f.read()
	messages = [
        {"role": "system", "content": prompt}, 
        {"role": "user", "content": f"# High-level problem description:\n{hl_desc}"}
    ]
	# Query the LLM
	return openai_ask_requests(messages)


def _define_solver(prompt, ctx):
	solver_type = "SCIP"  # TODO: ask the LLM to set the solver type based on the problem description
	code = """solver = define_solver("SCIP")"""
	return "\n\n" + code

def print_solution(sys_prompt, context, code, api_doc):
	func_code = code_utils.get_function_code("log_utils.py", ["get_solution_values"])
	code_hint = f"""The user has already implemented the optimization model. The code so far is as follows:

```python
{code}.
```

You also have acces to an API documentation for the DataLoader class which loads and processes the input data. Use it when relevant:

```python
{api_doc}
```

Your task is only to implement the solution visualization. To do so, you **MUST** use the function provided below:

```python
{func_code}
```
""" 
	messages = [
		{"role": "system", "content": sys_prompt+code_hint}, 
		{"role": "user", "content": context},
	]
	raw_response = openai_ask_requests(messages)
	source_code = code_utils.outer_code_parse(raw_response)
	return source_code

def _define_variables(sys_prompt, context, code, api_doc):
	func_code = code_utils.get_function_code("optimization_utils.py", ["define_variables"])
	code_hint = f"""The user has already implemented part of the optimization model. The code so far is as follows:

```python
{code}.
```

You also have acces to an API documentation for the DataLoader class which loads and processes the input data. Use it when relevant:

```python
{api_doc}
```

Your task is only to implement the decision variable definitions. To do so, you **MUST** use the functions provided below:

```python
{func_code}
```

Choose the most appropriate parameters based on the nature of the problem (e.g., binary decisions, integer allocations, indexed variables, etc.).

**Your task:**

* Only provide the Python code necessary to define the decision variables.
* Follow the conventions and structure used in the existing implementation.
* Do **not** include objective functions, constraints, or any other parts of the solution in this step.
"""

	# ✅ IMPORTANT: if api_doc is empty, forbid DataLoader usage (force manual extraction from text)

	if not (api_doc or "").strip():
		code_hint += (
		"\nIMPORTANT:\n"
		"- DataLoader has NO usable API for this problem (api_doc is empty).\n"
		"- Do NOT use DataLoader at all.\n"
		"- Extract all required numeric data directly from the problem text and encode it as Python lists/dicts.\n"
		)

	messages = [
	{"role": "system", "content": sys_prompt + code_hint},
	{"role": "user", "content": context},
	]
	raw_response = openai_ask_requests(messages)
	source_code = code_utils.outer_code_parse(raw_response)
	source_code = utils.add_type_comments(source_code)
	return source_code

def _define_objective(sys_prompt, context, code, api_doc):
	func_code = code_utils.get_function_code("optimization_utils.py", ["define_linear_expr", "add_objective"])

	code_hint = f"""The user has already implemented part of the optimization model. The code so far is as follows:

```python
{code}.
```

You also have acces to an API documentation for the DataLoader class which loads and processes the input data. Use it when relevant:

```python
{api_doc}
```

Your task is only to implement the objective function definitions. To do so, you **MUST** use the functions provided below:

```python
{func_code}
```

**Your task:**  
- Only provide the Python code necessary to define the objective function.  
- Follow the conventions and structure used in the existing implementation.
- Do **not** include constraints, or any other parts of the solution in this step.
""" 
	# ✅ IMPORTANT: if api_doc is empty, forbid DataLoader usage (force manual extraction from text)

	if not (api_doc or "").strip():
		code_hint += (
		"\nIMPORTANT:\n"
		"- DataLoader has NO usable API for this problem (api_doc is empty).\n"
		"- Do NOT use DataLoader at all.\n"
		"- Extract all required numeric data directly from the problem text and encode it as Python lists/dicts.\n"
		)
	messages = [
		{"role": "system", "content": sys_prompt+code_hint}, 
		{"role": "user", "content": context},
	]
	raw_response = openai_ask_requests(messages)
	source_code = code_utils.outer_code_parse(raw_response)
	return source_code

def _define_constraints(sys_prompt, context, code, api_doc):
	func_code = code_utils.get_function_code("optimization_utils.py", ["define_linear_expr", "add_constraint"])

	code_hint = f"""The user has already implemented part of the optimization model. The code so far is as follows:

```python
{code}.
```

You also have acces to an API documentation for the DataLoader class which loads and processes the input data. Use it when relevant:

```python
{api_doc}
```

Your task is only to implement the constraints definitions. To do so, you **MUST** use the functions provided below:

```python
{func_code}
```

**Your task:**  
- Only provide the Python code necessary to define the constraints.  
- Follow the conventions and structure used in the existing implementation.
- Do **not** include any other parts of the solution in this step.
"""
	# ✅ IMPORTANT: if api_doc is empty, forbid DataLoader usage (force manual extraction from text)

	if not (api_doc or "").strip():
		code_hint += (
		"\nIMPORTANT:\n"
		"- DataLoader has NO usable API for this problem (api_doc is empty).\n"
		"- Do NOT use DataLoader at all.\n"
		"- Extract all required numeric data directly from the problem text and encode it as Python lists/dicts.\n"
		)

	messages = [
		{"role": "system", "content": sys_prompt+code_hint}, 
		{"role": "user", "content": context},
	]
	raw_response = openai_ask_requests(messages)
	source_code = code_utils.outer_code_parse(raw_response)
	return source_code

def implement_optimization(prompt_path, context, code_base, api_doc):
	with open(prompt_path, "r") as f:
		sys_prompt = f.read()

	# Add the solver to the context
	code_base += _define_solver("", None)

	# Add variables to the context
	code_base += _define_variables(sys_prompt, context, code_base, api_doc)
	
	# Add objective to the context
	code_base += _define_objective(sys_prompt, context, code_base, api_doc)

	# Add constraints to the context
	code_base += _define_constraints(sys_prompt, context, code_base, api_doc)

	return code_base
	
def data_processing(prompt_path, context):
	with open(prompt_path, "r") as f:
		sys_prompt = f.read()
	messages = [
		{"role": "system", "content": sys_prompt}, 
		{"role": "user", "content": context},
	]

	raw_response = openai_ask_requests(messages)
	source_code = code_utils.outer_code_parse(raw_response)
	return source_code

def write_report(prompt_path, context, summary):
	with open(prompt_path, "r") as f:
		sys_prompt = f.read()
	messages = [
		{"role": "system", "content": sys_prompt}, 
		{"role": "user", "content": context+summary},
	]
	
	raw_response = openai_ask_requests(messages)
	return raw_response