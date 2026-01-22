import code_utils
import os
import requests
import utils

def openai_ask_requests(messages, model="gpt-5", response_format=None):

	url = f"https://cld.akkodis.com/api/openai/deployments/models-{model}/chat/completions?api-version=2024-12-01-preview"
	headers = {
		"Content-Type": "application/json",
		"Cache-Control": "no-cache",
		"api-key": open(os.path.join(os.getcwd(), ".api_key.txt"), 'r').read().strip()
	}

	data = {
		"max_tokens": 10000,
		"messages": messages
	}

	if response_format is not None:
		data["response_format"] = response_format
	response = requests.post(url, headers=headers, json=data).json()
	return response['choices'][0]['message']['content']

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
	print(raw_response)
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
- Only provide the Python code necessary to define the decision variables.  
- Follow the conventions and structure used in the existing implementation.
- Do **not** include objective functions, constraints, or any other parts of the solution in this step.
""" 
	messages = [
		{"role": "system", "content": sys_prompt+code_hint}, 
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
