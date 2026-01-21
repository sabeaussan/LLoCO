import libcst as cst

class TypeCommentInserter(cst.CSTTransformer):

    def _build_comment_type(self, assign_node, call_node):
        comment = "# "+assign_node.targets[0].target.value
        comment += " is a numpy array of decision variables. Its shape is defined by "
        shape_arg = call_node.args[1]
        if isinstance(shape_arg.value, cst.Integer):
            comment += f"({shape_arg.value.value},)"
        elif isinstance(shape_arg.value, cst.Tuple):
            comment += "("
            for item in shape_arg.value.elements:
                if isinstance(item, cst.Integer):
                    comment += f"{item.value.value},"
            comment = comment[:-1] + ")"
        elif isinstance(shape_arg.value, cst.Name):
            comment += f"the {shape_arg.value.value} variable."
        else:
            raise NotImplementedError("Unsupported shape type in define_variables call")
                
        return comment

    def leave_Module(self, original_node, updated_node):
        new_body = []
        for stmt in updated_node.body:
            new_stmt = None
            if isinstance(stmt, cst.SimpleStatementLine):
                # Check if it's an assignment statement
                assign = stmt.body[0]
                if isinstance(assign, cst.Assign):
                    value = assign.value
                    if isinstance(value, cst.Call):
                        func = value.func
                        if isinstance(func, cst.Name) and func.value == "define_variables":
                            str_comment_warning = "# **IMPORTANT** : The define_variables method always returns a numpy array of decision variables."
                            comment_warning = cst.Comment(value=str_comment_warning)
                            str_comment_type = self._build_comment_type(assign, value)
                            comment_type = cst.Comment(value=str_comment_type)
                            lines = list(stmt.leading_lines) + [cst.EmptyLine(comment=comment_warning), cst.EmptyLine(comment=comment_type)]
                            new_stmt = stmt.with_changes(
                                leading_lines=tuple(lines)
                            )
            if new_stmt is not None:
                new_body.append(new_stmt)
            else:
                new_body.append(stmt)
        return updated_node.with_changes(body=new_body)

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
    #tree = cst.parse_module(code)
    #updated_tree = tree.visit(TypeCommentInserter())
    #return updated_tree.code
    return code




