import numpy as np
import utils
from ortools.linear_solver import pywraplp
import operator

# TODO : perhaps better to define linear expression directly with solver.Sum

def _get_value(value, neg=False):
	"""
		Check if the value is None. If it is, return np.inf, otherwise return the value as a float.
	"""
	if value is None:
		if neg:
			return -np.inf
		else:
			return np.inf
	else:
		return float(value)

def define_linear_expr(x, weights):
	"""
        Define a linear expression used for constraints definition and/or objective function definition. It is a linear combination of the decision variables x and the weights. The decision variables x and weights are both numpy arrays that MUST have only one dimension.

		Parameters
    	----------
            x : ndarray
				1-D Numpy array of decision variables. The dtype of the array is object to allow for or-tools variable objects.
			weights : ndarray.
				1-D Numpy array of weights, with the same shape as x. The dtype of the array is float.
		Return : ndarray
		-------
			Numpy array of linear terms, i.e a linear combination of the decision variables x and the weights.

		Raises
		------
		ValueError
			If x and weights are not 1-D numpy arrays

		Examples
    	--------
			# Combine the decision variables into a single array
			decision_variables = np.concatenate((x, y))
			# decision_variables and weights are both 1-D numpy arrays of the same shape
			expr = define_linear_expr(decision_variables, weights)
    """
	# Remove potential extra axes of length one
	if x.ndim != 1:
		x = x.squeeze()

	# Remove potential extra axes of length one
	if weights.ndim != 1:
		weights = weights.squeeze()

	if x.ndim != 1 or weights.ndim != 1:
		raise ValueError("x and weights must be 1-D numpy arrays !")
	return x * weights

def define_solver(id):
    """
        Define a solver based on provided id.

		Parameters
    	----------
        id : string
			Solver type (for now only "SCIP" is available. TODO: add more types when required)

		Returns
    	-------
		solver : pywraplp.Solver
			The solver instance which will contain the decision variables and solution.
	
		Raises
		------
		ValueError
			If the solver could not be created (e.g., unsupported id).
    """
    solver = pywraplp.Solver.CreateSolver(id)

    if not solver:
        raise ValueError("Solver missing !")

    return solver

def add_objective(solver, expr, maximize=True):
	"""
        Add the optimization objective to the provided solver in-place.

        Parameters
    	----------
        solver : pywraplp.Solver
			The solver instance which will contain the decision variables and solution.
		expr : 1-D numpy array
			1-D numpy array of linear term composing the objective function.
		direction : bool
			Whether we should we maximize or minimize the objective.
		
		Returns
    	-------
		None. The constraint is added to the solver in-place.

		Raises
		------
		ValueError
			If expr is not a 1-D numpy array.

		Examples
    	--------
			objective_expr = define_linear_expr(decision_variables, weights)
			add_objective(solver, objective_expr, maximize=False)
	"""
	# Remove potential extra axes of length one
	if expr.ndim != 1:
		expr = expr.squeeze()

	if expr.ndim != 1:
		raise ValueError("expr must be 1-D numpy arrays !")
	
	o_expr = solver.Sum(expr)
	if maximize:
		solver.Maximize(o_expr)
	else:
		solver.Minimize(o_expr)
	
def define_variables(solver, shape, lbs, ubs, integer, suffix):
	"""
		Define the decision variables. If the integer boolean parameter is set to True,
		the variables will be integer variables, otherwise they will be continuous variables.

		Parameters
    	----------
		solver : pywraplp.Solver
			The solver instance which will contain the decision variables and solution.
		shape : tuple of ints
			The shape for the Numpy array of decision variables. Empty shape is not a valid input. For single variable, use shape=(1,).
		lb : array_like
			Lower bounds for the decision variables. Same shape as shape. If set to None, no lower bounds (-inf).
		ub : array_like
			Upper bounds for the decision variables. Same shape as shape. If set to None, no upper bounds (+inf).
		integer : bool
			Boolean indicating whether the decision variables are integer or continuous.
		suffix : str
			Name suffix for the decision variables. If None, no suffix is added.

		Returns
    	-------
		x : ndarray
			Numpy array of decision variables, with the specified shape, bounds and type.
			The dtype of the array is object to allow for or-tools variable objects.

		Examples
    	--------
			x = define_variables(solver, shape=(2,), lbs=[0,0], ubs=[None, None], integer=True, suffix="amount")

		Raises
		------
		ValueError
			If shape is empty.
	"""

	# Make sure the bounds are numpy arrays
	# to allow for tuple indexing
	if isinstance(lbs, list):
		lbs = np.array(lbs)
	
	if isinstance(ubs, list):
		ubs = np.array(ubs)

	if isinstance(shape, int):
		# If shape is an integer, convert it to a tuple
		shape = (shape,)

	if shape == ():
		raise ValueError("shape cannot be empty ! Please provide a valid shape for the decision variables.")

    # Pre-allocate Numpy array of variables object
	x = np.empty(shape, dtype=object)

	# List of iterables
	variables_indices = []

	# Add iterables 
	for i in range(len(shape)):
		variables_indices.append(range(shape[i])) 

	if suffix is not None:
		suffix = "_"+ suffix
	else:
		suffix = ""
	
	# Recursive looping through indices to assign Decision Variables
	for i in utils.nested_loops(variables_indices):
		if isinstance(ubs, (list, np.ndarray)):
			ub = _get_value(ubs[i])
		else:
			ub = _get_value(ubs)
		if isinstance(lbs, (list, np.ndarray)):
			lb = _get_value(lbs[i], neg=True)
		else:
			lb = _get_value(lbs, neg=True)
		x[i] = solver.Var(ub=ub, lb=lb, integer=integer, name="x"+suffix+f"_{i}")
	return x


def add_constraint(solver, expr, c_val, c_operator, c_name, eps_relax=0.0):
	# TODO : specify direction of the operator comparison
	"""
        Add constraint to the provided solver in-place.

		Parameters
    	----------
        solver : pywraplp.Solver
			The solver instance which will contain the decision variables and solution.
		expr : 1-D numpy array
			1-D numpy array of linear term composing the left hand side of the constraint.
		c_val : float
			Value of the constraint (right hand side).
		c_operator : python operator
			Constraint comparison operator. Can be either : operator.le, operator.ge or operator.eq. If set to operator.lt or operator.gt (i.e strit inequality), the eps_relax value needs to be set accordingly;
		c_name : str
			Name of the constraint. Used for debugging and logging purposes.
		eps_relax : positive float
			If set to a non-zero float value, will apply epsilon relaxation to a strict constraint inequality to approximate it and be compatible with the MILP solver. If not None, eps_relax must be positive. The sign is handled according the inequality direction in the code. For e.g, if a and b are two decision variables a < b <=> a-b < 0 which is (approximately) a-b <= -eps_relax. For integer values, eps_relax=1. For continuous values, eps_relax must be set according to the scale and units of the decision variables. For time in seconds, 0.0001 might be safe. For distances in kilometers, that same value might be too small.
		
		Returns
    	-------
		None. The constraint is added to the solver in-place.

		Raises
		------
		ValueError
			If expr is not a 1-D numpy array.

		ValueError
			If c_operator is either operator.gt or operator.lt.
		
		Examples
    	--------
			budget_expr = define_linear_expr(decision_variables, weights)
			# Add the budget constraint to the solver
			add_constraint(solver, budget_expr, c_val=2000, c_operator=operator.lt, c_name="budget_constraint", eps_relax=0.01)
    """
	# Remove potential extra axes of length one
	if expr.ndim != 1:
		expr = expr.squeeze()

	if expr.ndim != 1:
		raise ValueError("expr must be 1-D numpy arrays !")
	
	
	# Define constraint lhs expression
	c_expr = solver.Sum(expr)

	# Add epsilon relaxation for strict inequalities
	if c_operator == operator.gt:
		c_val += eps_relax
		c_operator = operator.ge
	if c_operator == operator.lt:
		c_val -= eps_relax
		c_operator = operator.le
	
	# Define the constraint (in)-equality 
	constraint = c_operator(c_expr,c_val)
	solver.Add(constraint, name=c_name)