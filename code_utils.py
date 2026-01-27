import ast
import os

def get_function_code(target_file, function_names):
	with open(target_file, "r") as file:
		source_code = file.read()
	tree = ast.parse(source_code)
	result = ""
	for name in function_names:
		for node in ast.walk(tree):
			if isinstance(node, ast.FunctionDef) and node.name.startswith(name):
				start_line = node.lineno
				end_line = node.end_lineno
				result += "\n".join(source_code.split('\n')[start_line - 1:end_line])
				result += "\n"
	return result

def load_csv(directory):
	files = os.listdir(directory)
	source_code = "\n\n# Load input files\n"
	for file in files:
		if file.endswith(".csv"):
			file_path = os.path.join(directory, file)
			fname = file_path.split("/")[-1]
			var_name = fname.split(".")[0]
			source_code += f'{var_name}_df = pd.read_csv("{fname}")\n'
	return source_code


def define_imports():
	source_code = """import sys
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
import numpy as np"""
	return source_code

def add_print_summary():
	source_code = """# Solve the optimization problem
status = solver.Solve()
# == Print summary ==
interpret_status(status)
print_objective_solution_value(solver)
"""
	return source_code

def outer_code_parse(contents):
	code = contents.split('```python')[1]
	code = "'''".join(code.split('```')[:-1])
	return code
