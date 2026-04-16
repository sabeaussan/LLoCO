import re

# Regex: captures  var_name = define_variables(solver, shape=..., ...)
_DEFVAR_RE = re.compile(
    r"^(\s*)"                             # leading indent
    r"(\w+)"                              # variable name
    r"\s*=\s*define_variables\s*\("       # = define_variables(
    r"[^)]*?"                             # args before shape (non-greedy)
    r"shape\s*=\s*(\([^)]*\)|\w+)"       # shape=(tuple) or shape=name_or_int
    r"[^)]*\)",                           # rest of args + closing )
    re.MULTILINE,
)

def nested_loops(loops_iterables):
    """
        Define nested loops recursively to avoid if/else statements based on len(num_terms)
            -  loops_iterables: List of iterables to loop through.

            Ex: nested_loops([range(10),range(20)]) is equivalent to
                for i in range(10):
                    for j in range(20): 
                        yield i,j

        Yield indices
    """
    return _nested_loops(loops_iterables=loops_iterables, depth=len(loops_iterables), indices=[])

def _nested_loops(loops_iterables, depth, indices):
    if depth == 0:
        return
    for i in loops_iterables[0]:
        yield from _nested_loops(loops_iterables=loops_iterables[1:], indices=indices+[i], depth=depth-1)
        if depth-1 == 0:
            yield tuple(indices + [i])

def add_type_comments(code):
    """
    Annotate each `define_variables(...)` call with a comment reminding the LLM
    that the return value is a numpy array of OR-Tools decision variables.
    """
    def _insert_comment(match):
        indent = match.group(1)
        var_name = match.group(2)
        shape_raw = match.group(3).strip()
        comment = (
            f"{indent}# IMPORTANT: {var_name} is a numpy array (dtype=object) "
            f"of OR-Tools decision variables with shape={shape_raw}.\n"
            f"{indent}# Index it like a numpy array. "
            f"Use .flatten() to get a 1-D array for define_linear_expr.\n"
        )
        return comment + match.group(0)

    return _DEFVAR_RE.sub(_insert_comment, code)




