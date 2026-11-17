import ast

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import Function
from petsc4py import PETSc


class ExpressionHandler:
    def __init__(self):
        self.math_env = {
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "exp": np.exp,
            "log": np.log,
            "sqrt": np.sqrt,
            "pi": np.pi,
            "np": np,
            "abs": np.abs,
            "min": np.minimum,
            "max": np.maximum,
        }
        self.symbolic_env = {
            "sin": ufl.sin,
            "cos": ufl.cos,
            "tan": ufl.tan,
            "exp": ufl.exp,
            "log": ufl.ln,
            "sqrt": ufl.sqrt,
            "pi": ufl.pi,
            "abs": abs,
            "min": ufl.min_value,
            "max": ufl.max_value,
        }
        self.allowed_names = set(self.math_env) | {"x", "t"}
        self.allowed_call_names = {
            name
            for name, value in {**self.math_env, **self.symbolic_env}.items()
            if callable(value)
        }

    def create_function(self, space, value_config, form="constant"):
        """
        Create a Function and, for time-dependent inputs, return an updater closure.

        Returns:
            func: dolfinx.fem.Function
            updater: function(t) -> None, or None for constant data
        """
        func = Function(space)
        component_count = self._component_count(func)
        is_vector = component_count > 1
        self._validate_value_shape(value_config, is_vector, component_count)
        
        if form == "constant":
            self._set_constant(func, value_config, is_vector)
            return func, None

        if form == "expression":
            updater = self._make_updater(func, value_config, is_vector)
            updater(0.0)
            return func, updater

        raise ValueError(f"Unknown form: {form}")

    def create_exact_expression(self, mesh, value_config, form="constant", time_symbol=None):
        if form == "constant":
            if isinstance(value_config, (list, tuple)):
                return ufl.as_vector(tuple(PETSc.ScalarType(value) for value in value_config))
            return PETSc.ScalarType(value_config)

        if form != "expression":
            raise ValueError(f"Unknown form: {form}")

        local_env = self.symbolic_env.copy()
        local_env["x"] = ufl.SpatialCoordinate(mesh)
        local_env["t"] = PETSc.ScalarType(0.0) if time_symbol is None else time_symbol

        if isinstance(value_config, (list, tuple)):
            components = []
            for expression in value_config:
                compiled = self._compile_expression(expression)
                components.append(eval(compiled, {"__builtins__": None}, local_env))
            return ufl.as_vector(tuple(components))

        compiled = self._compile_expression(value_config)
        return eval(compiled, {"__builtins__": None}, local_env)

    def create_exact_function(self, space, value_config, form="constant", time_symbol=None):
        func = Function(space)
        component_count = self._component_count(func)
        is_vector = component_count > 1
        self._validate_value_shape(value_config, is_vector, component_count)

        if form == "constant":
            self._set_constant(func, value_config, is_vector)
            return func, None

        if form != "expression":
            raise ValueError(f"Unknown form: {form}")

        exact_expr = self.create_exact_expression(
            space.mesh,
            value_config,
            form=form,
            time_symbol=time_symbol,
        )
        interpolation_points = space.element.interpolation_points
        if callable(interpolation_points):
            interpolation_points = interpolation_points()
        fem_expr = fem.Expression(exact_expr, interpolation_points)

        def updater(_t):
            func.interpolate(fem_expr)
            func.x.scatter_forward()

        updater(0.0)
        return func, updater

    def _compile_expression(self, expression):
        expr_text = str(expression)
        tree = ast.parse(expr_text, mode="eval")
        self._validate_expression(tree)
        return compile(tree, "<expression>", "eval")

    def _validate_expression(self, tree):
        allowed_nodes = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Call,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.Subscript,
            ast.Compare,
            ast.BoolOp,
            ast.IfExp,
            ast.Tuple,
            ast.List,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.Mod,
            ast.UAdd,
            ast.USub,
            ast.And,
            ast.Or,
            ast.Eq,
            ast.NotEq,
            ast.Lt,
            ast.LtE,
            ast.Gt,
            ast.GtE,
        )

        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                raise ValueError(f"Unsupported expression syntax: {ast.dump(node, include_attributes=False)}")

            if isinstance(node, ast.Name) and node.id not in self.allowed_names:
                raise ValueError(f"Unsupported symbol in expression: {node.id}")

            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in self.allowed_call_names:
                    raise ValueError("Only whitelisted scalar functions can be called in expressions.")
                if node.keywords:
                    raise ValueError("Keyword arguments are not supported in expressions.")

            if isinstance(node, ast.Subscript):
                if not isinstance(node.value, ast.Name) or node.value.id != "x":
                    raise ValueError("Only coordinate access of the form x[i] is supported in expressions.")
                if not isinstance(node.slice, ast.Constant) or not isinstance(node.slice.value, int):
                    raise ValueError("Coordinate access must use an integer index, for example x[0].")
                if node.slice.value < 0:
                    raise ValueError("Coordinate indices must be non-negative.")

    @staticmethod
    def _component_count(func):
        if func.function_space.num_sub_spaces > 0:
            return func.function_space.num_sub_spaces
        return func.function_space.dofmap.index_map_bs

    @staticmethod
    def _validate_value_shape(value, is_vector, component_count):
        if is_vector:
            if not isinstance(value, (list, tuple)):
                raise ValueError(
                    f"Vector data must have {component_count} components."
                )
            if len(value) != component_count:
                raise ValueError(
                    f"Vector data must have {component_count} components; got {len(value)}."
                )
            return

        if isinstance(value, (list, tuple)):
            raise ValueError("Scalar data must be a single value, not a list.")

    def _set_constant(self, func, value, is_vector):
        if is_vector:
            val = np.array(value, dtype=PETSc.ScalarType)
            func.interpolate(
                lambda x: np.stack(
                    [
                        np.full(x.shape[1], val[i], dtype=PETSc.ScalarType)
                        for i in range(len(val))
                    ]
                )
            )
        else:
            val = PETSc.ScalarType(value)
            func.x.array[:] = val
        func.x.scatter_forward()

    def _make_updater(self, func, expr_list, is_vector):
        math_env = self.math_env
        compiled = None
        if is_vector:
            compiled = [self._compile_expression(expr) for expr in expr_list]
        else:
            compiled = self._compile_expression(expr_list)

        def updater(t):
            local_env = math_env.copy()
            local_env["t"] = t
            
            if is_vector:
                def interp_wrapper(x):
                    local_env["x"] = x
                    comps = []
                    for compiled_expr, e_str in zip(compiled, expr_list):
                        try:
                            res = eval(compiled_expr, {"__builtins__": None}, local_env)
                            if np.isscalar(res) or res.shape == ():
                                res = np.full(x.shape[1], res, dtype=PETSc.ScalarType)
                            comps.append(res)
                        except Exception as e:
                            raise RuntimeError(f"Expr eval failed: {e_str}, Error: {e}")
                    return np.asarray(np.stack(comps), dtype=PETSc.ScalarType)
            else:
                def interp_wrapper(x):
                    local_env["x"] = x
                    try:
                        res = eval(compiled, {"__builtins__": None}, local_env)
                        if np.isscalar(res):
                            res = np.full(x.shape[1], res, dtype=PETSc.ScalarType)
                        return np.asarray(res, dtype=PETSc.ScalarType)
                    except Exception as e:
                        raise RuntimeError(f"Expr eval failed: {expr_list}, Error: {e}")

            func.interpolate(interp_wrapper)
            func.x.scatter_forward()
        
        return updater
