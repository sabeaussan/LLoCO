from ortools.linear_solver import pywraplp
import utils
import numpy as np

def _extract_constraint_value(solver, constraint):
	"""
	Computes the evaluated value of a constraint based on the current solution.

	Parameters
	----------
	solver : pywraplp.Solver
		The solver instance containing the decision variables and solution.
	constraint : pywraplp.Constraint
		The constraint whose value is to be evaluated.

	Returns
	-------
	float
		The computed left-hand side value of the constraint expression using the current solution values of the decision variables.
	"""

	val = 0
	for var in solver.variables():
		val += (constraint.GetCoefficient(var) * var.solution_value())
	return val

def print_solution_summary(solver):
	"""
	Prints a summary of all decision variables and their assigned values in the current solution.

	Parameters
	----------
	solver : pywraplp.Solver
		The solver instance containing the solved decision variables.

	Returns
	-------
	None
		Outputs the variable names and values to standard output.
	"""
	print("=== Solution summary ===")
	for var in solver.variables():
		name = var.name()
		val = var.solution_value()
		if val > 0.01:
			print(f"Variable {name} : value {val}")
	print()

def print_objective_solution_value(solver):
	"""
	Prints the objective function value from the current solution, prefixed with a custom description.

	Parameters
	----------
	solver : pywraplp.Solver
		The solver instance that has solved the optimization problem.

	Returns
	-------
	None
		Outputs the objective value to standard output.
	"""

	print("=== Optimization objective value ===")
	obj_value = solver.Objective().Value()
	print("Optimization objective value" + f": {obj_value}")
	print()

def print_constraints_summary(solver):
	"""
	Prints a summary of all constraints, showing their bounds and the computed values based on the current solution.

	Parameters
	----------
	solver : pywraplp.Solver
		The solver instance containing the constraints and solution.

	Returns
	-------
	None
		Print constraint names and their satisfaction status.
	"""

	print("=== Constraints satisfaction summary ===")
	for constraint in solver.constraints():
		name = constraint.name()
		val = _extract_constraint_value(solver, constraint)
		print(f"constraint {name} : {constraint.Lb()} <= {val} <= {constraint.Ub()}")
	print()

def interpret_status(status):
	"""
	Interprets and returns a human-readable message corresponding to a solver status code.

	Parameters
	----------
	status : int
		Solver status code (e.g., pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE).

	Returns
	-------
	str
		A human-readable explanation of the solver status.
	"""
	print("\nOptimization Results:")
	print("---------------------")
	summary = None
	if status == pywraplp.Solver.OPTIMAL:
		summary = "Optimal solution found."
	elif status == pywraplp.Solver.FEASIBLE:
		summary = "A feasible solution was found, but it may not be optimal."
	elif status == pywraplp.Solver.INFEASIBLE:
		summary = "No feasible solution exists."
	elif status == pywraplp.Solver.UNBOUNDED:
		summary = "The problem is unbounded."
	else:
		summary = "Solver status unknown."
	print(f"Solver Status: {summary}\n")

def get_solution_values(vars, print_threshold=0.01):
	"""
    Extracts numeric solution values from a NumPy array of OR-Tools decision variables.

    The function iterates through all elements of a NumPy array containing OR-Tools variables,
    calls `.solution_value()` on each, and returns a numeric array of the same shape.
    Only variable values greater than the given threshold are retained, which helps filter
    out near-zero numerical artifacts in solver outputs.

    Parameters
    ----------
    vars : np.ndarray
        NumPy array of OR-Tools variable objects (e.g., IntVar, NumVar), of arbitrary shape.
    print_threshold : float, optional
        Minimum value threshold for recording a variable’s solution. Variables with values
        less than or equal to this threshold are set to zero. Default is 0.01.

    Returns
    -------
    np.ndarray
        NumPy array of floats with the same shape as `vars`, containing the numerical
        solution values extracted from the solver.
    """
	solution = np.zeros_like(vars, dtype=float)

	# List of iterables
	variables_indices = []

	# Add iterables 
	for i in range(len(vars.shape)):
		variables_indices.append(range(vars.shape[i])) 

	# Recursive looping through indices
	for idxs in utils.nested_loops(variables_indices):
		val = vars[idxs].solution_value()
		if val > print_threshold:
			solution[idxs]=val
	return solution
