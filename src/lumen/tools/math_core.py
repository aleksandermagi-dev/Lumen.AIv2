from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from typing import Any


class MathExpressionError(ValueError):
    pass


Polynomial = dict[tuple[tuple[str, int], ...], float]

_ALLOWED_FUNCTIONS = {"sin", "cos", "tan", "sqrt", "log", "exp"}
_CONSTANTS = {"pi": math.pi, "e": math.e}
_UNICODE_MATH_TRANSLATION = str.maketrans(
    {
        "²": "^2",
        "³": "^3",
        "⁰": "^0",
        "¹": "^1",
        "⁴": "^4",
        "⁵": "^5",
        "⁶": "^6",
        "⁷": "^7",
        "⁸": "^8",
        "⁹": "^9",
        "−": "-",
        "–": "-",
        "—": "-",
    }
)


def _tokenize(expression: str) -> list[str]:
    expression = expression.translate(_UNICODE_MATH_TRANSLATION)
    tokens: list[str] = []
    i = 0
    while i < len(expression):
        char = expression[i]
        if char.isspace():
            i += 1
            continue
        if char.isdigit() or char == ".":
            j = i + 1
            while j < len(expression) and (expression[j].isdigit() or expression[j] == "."):
                j += 1
            tokens.append(expression[i:j])
            i = j
            continue
        if char.isalpha() or char == "_":
            j = i + 1
            while j < len(expression) and (expression[j].isalpha() or expression[j] == "_"):
                j += 1
            word = expression[i:j]
            if word in _ALLOWED_FUNCTIONS or word in _CONSTANTS:
                tokens.append(word)
            else:
                tokens.extend(list(word))
            i = j
            continue
        if char in "+-*/^=(),":
            tokens.append("**" if char == "^" else char)
            i += 1
            continue
        raise MathExpressionError(f"Unsupported character '{char}' in expression")
    return tokens


def normalize_expression(expression: str) -> str:
    tokens = _tokenize(expression)
    if not tokens:
        raise MathExpressionError("Expression is empty")
    result: list[str] = []
    for index, token in enumerate(tokens):
        if index > 0 and _needs_implicit_multiplication(tokens[index - 1], token):
            result.append("*")
        result.append(token)
    return "".join(result)


def _needs_implicit_multiplication(left: str, right: str) -> bool:
    if left in {"+", "-", "*", "/", "**", "=", ",", "("}:
        return False
    if right in {"+", "-", "*", "/", "**", "=", ",", ")"}:
        return False
    if left in _ALLOWED_FUNCTIONS and right == "(":
        return False
    return True


def parse_expression(expression: str) -> ast.AST:
    normalized = normalize_expression(expression)
    try:
        return ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise MathExpressionError(f"Invalid expression '{expression}'") from exc


def evaluate_expression(expression: str, variables: dict[str, float] | None = None) -> float:
    node = parse_expression(expression)
    return _eval_node(node.body, variables or {})


def _eval_node(node: ast.AST, variables: dict[str, float]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in variables:
            return float(variables[node.id])
        if node.id in _CONSTANTS:
            return float(_CONSTANTS[node.id])
        raise MathExpressionError(f"Unknown variable '{node.id}'")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
    if isinstance(node, ast.UnaryOp):
        value = _eval_node(node.operand, variables)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCTIONS:
        if len(node.args) != 1:
            raise MathExpressionError(f"Function '{node.func.id}' expects one argument")
        value = _eval_node(node.args[0], variables)
        return float(getattr(math, node.func.id)(value))
    raise MathExpressionError("Expression uses unsupported syntax")


def expression_variables(expression: str) -> set[str]:
    node = parse_expression(expression)
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id not in _ALLOWED_FUNCTIONS and child.id not in _CONSTANTS:
            names.add(child.id)
    return names


def polynomial_from_expression(expression: str) -> Polynomial:
    node = parse_expression(expression)
    return _poly_from_node(node.body)


def _poly_from_node(node: ast.AST) -> Polynomial:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return {(): float(node.value)}
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return {(): float(_CONSTANTS[node.id])}
        return {((node.id, 1),): 1.0}
    if isinstance(node, ast.UnaryOp):
        poly = _poly_from_node(node.operand)
        if isinstance(node.op, ast.USub):
            return {monomial: -coefficient for monomial, coefficient in poly.items()}
        if isinstance(node.op, ast.UAdd):
            return poly
    if isinstance(node, ast.BinOp):
        left = _poly_from_node(node.left)
        right = _poly_from_node(node.right)
        if isinstance(node.op, ast.Add):
            return _add_poly(left, right)
        if isinstance(node.op, ast.Sub):
            return _add_poly(left, {monomial: -value for monomial, value in right.items()})
        if isinstance(node.op, ast.Mult):
            return _mul_poly(left, right)
        if isinstance(node.op, ast.Div):
            divisor = _constant_value(right)
            if divisor is None or divisor == 0:
                raise MathExpressionError("Only constant division is supported")
            return {monomial: value / divisor for monomial, value in left.items()}
        if isinstance(node.op, ast.Pow):
            exponent = _constant_value(right)
            if exponent is None or int(exponent) != exponent or exponent < 0 or exponent > 6:
                raise MathExpressionError("Only bounded non-negative integer powers are supported")
            result: Polynomial = {(): 1.0}
            for _ in range(int(exponent)):
                result = _mul_poly(result, left)
            return result
    raise MathExpressionError("Expression cannot be represented as a bounded polynomial")


def _constant_value(poly: Polynomial) -> float | None:
    if set(poly.keys()) == {()}:
        return poly[()]
    return None


def _add_poly(left: Polynomial, right: Polynomial) -> Polynomial:
    result = dict(left)
    for monomial, value in right.items():
        result[monomial] = result.get(monomial, 0.0) + value
        if abs(result[monomial]) < 1e-12:
            result.pop(monomial)
    return result or {(): 0.0}


def _mul_poly(left: Polynomial, right: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for left_monomial, left_value in left.items():
        for right_monomial, right_value in right.items():
            monomial = _merge_monomials(left_monomial, right_monomial)
            result[monomial] = result.get(monomial, 0.0) + (left_value * right_value)
            if abs(result[monomial]) < 1e-12:
                result.pop(monomial)
    return result or {(): 0.0}


def _merge_monomials(
    left: tuple[tuple[str, int], ...],
    right: tuple[tuple[str, int], ...],
) -> tuple[tuple[str, int], ...]:
    powers: dict[str, int] = {}
    for name, exponent in left + right:
        powers[name] = powers.get(name, 0) + exponent
    return tuple(sorted((name, exponent) for name, exponent in powers.items() if exponent))


def polynomial_to_expression(poly: Polynomial) -> str:
    ordered = sorted(
        poly.items(),
        key=lambda item: (sum(exponent for _, exponent in item[0]), item[0]),
        reverse=True,
    )
    if not ordered:
        return "0"
    pieces: list[str] = []
    for monomial, coefficient in ordered:
        if abs(coefficient) < 1e-12:
            continue
        term = _term_to_expression(monomial, coefficient, leading=not pieces)
        pieces.append(term)
    return " ".join(pieces) if pieces else "0"


def _term_to_expression(
    monomial: tuple[tuple[str, int], ...],
    coefficient: float,
    *,
    leading: bool,
) -> str:
    sign = "-" if coefficient < 0 else "+"
    magnitude = abs(coefficient)
    variable_parts: list[str] = []
    for name, exponent in monomial:
        if exponent == 1:
            variable_parts.append(name)
        else:
            variable_parts.append(f"{name}^{exponent}")
    variable_expr = " * ".join(variable_parts)
    if variable_expr:
        if abs(magnitude - 1.0) < 1e-12:
            core = variable_expr
        else:
            core = f"{_format_number(magnitude)} * {variable_expr}"
    else:
        core = _format_number(magnitude)
    if leading:
        return core if sign == "+" else f"-{core}"
    return f"{sign} {core}"


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(round(value)))
    return f"{value:.6g}"


@dataclass(slots=True)
class EquationSolveResult:
    solution: list[str]
    steps: list[str]
    assumptions: list[str]
    confidence: str


def solve_equation(equation: str, variable: str) -> EquationSolveResult:
    if "=" not in equation:
        raise MathExpressionError("Equation must contain '='")
    left_text, right_text = equation.split("=", maxsplit=1)
    try:
        left_poly = polynomial_from_expression(left_text)
        right_poly = polynomial_from_expression(right_text)
    except MathExpressionError as exc:
        if _should_try_symbolic_solver(str(exc)):
            return _solve_equation_symbolically(equation, variable, fallback_reason=str(exc))
        raise
    combined = _add_poly(left_poly, {monomial: -value for monomial, value in right_poly.items()})
    degree = max((dict(monomial).get(variable, 0) for monomial in combined), default=0)
    if degree == 0:
        raise MathExpressionError(f"Variable '{variable}' is not present in the equation")
    if degree > 2:
        return _solve_equation_symbolically(
            equation,
            variable,
            fallback_reason="Polynomial degree is above the fast quadratic solver.",
        )

    if degree == 1:
        coefficient_poly: Polynomial = {}
        constant_poly: Polynomial = {}
        for monomial, value in combined.items():
            powers = dict(monomial)
            exponent = powers.pop(variable, 0)
            reduced = tuple(sorted((name, exp) for name, exp in powers.items() if exp))
            if exponent == 1:
                coefficient_poly[reduced] = coefficient_poly.get(reduced, 0.0) + value
            else:
                constant_poly[monomial] = constant_poly.get(monomial, 0.0) + value
        coefficient_expr = polynomial_to_expression(coefficient_poly or {(): 0.0})
        constant_expr = polynomial_to_expression(constant_poly or {(): 0.0})
        simplified_solution = _linear_solution_expression(coefficient_expr, constant_expr)
        return EquationSolveResult(
            solution=[simplified_solution],
            steps=[
                f"Move all terms to one side: {polynomial_to_expression(combined)} = 0",
                f"Collect terms in {variable}: ({coefficient_expr}) * {variable} + ({constant_expr}) = 0",
                f"Solve for {variable}: {variable} = {simplified_solution}",
            ],
            assumptions=["All non-target symbols are treated as constants."],
            confidence="high",
        )

    numeric_variables = (expression_variables(left_text) | expression_variables(right_text)) - {variable}
    if numeric_variables:
        raise MathExpressionError("Quadratic solving with additional symbolic variables is not supported")
    a = b = c = 0.0
    for monomial, value in combined.items():
        exponent = dict(monomial).get(variable, 0)
        if exponent == 2:
            a += value
        elif exponent == 1:
            b += value
        elif exponent == 0:
            c += value
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        raise MathExpressionError("Quadratic equation has no real roots")
    root = math.sqrt(discriminant)
    solutions = sorted({(-b + root) / (2 * a), (-b - root) / (2 * a)})
    return EquationSolveResult(
        solution=[_format_number(item) for item in solutions],
        steps=[
            f"Move all terms to one side: {polynomial_to_expression(combined)} = 0",
            f"Identify coefficients: a={_format_number(a)}, b={_format_number(b)}, c={_format_number(c)}",
            "Apply the quadratic formula.",
        ],
        assumptions=["Real-valued roots only."],
        confidence="high",
    )


def _should_try_symbolic_solver(reason: str) -> bool:
    normalized = reason.lower()
    return any(
        phrase in normalized
        for phrase in (
            "only constant division is supported",
            "only bounded non-negative integer powers are supported",
            "cannot be represented as a bounded polynomial",
        )
    )


def _solve_equation_symbolically(
    equation: str,
    variable: str,
    *,
    fallback_reason: str,
) -> EquationSolveResult:
    try:
        import sympy as sp
    except Exception as exc:  # pragma: no cover - depends on packaged optional dependency state.
        raise MathExpressionError("Symbolic equation solving is unavailable in this runtime") from exc

    if "=" not in equation:
        raise MathExpressionError("Equation must contain '='")

    left_text, right_text = equation.split("=", maxsplit=1)
    names = expression_variables(left_text) | expression_variables(right_text)
    if variable not in names:
        raise MathExpressionError(f"Variable '{variable}' is not present in the equation")
    extra_variables = sorted(name for name in names if name != variable)
    if extra_variables:
        raise MathExpressionError(
            "Symbolic equation solving currently supports one target variable at a time"
        )

    symbol = sp.Symbol(variable, real=True)
    locals_map: dict[str, Any] = {
        variable: symbol,
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "log": sp.log,
        "exp": sp.exp,
        "pi": sp.pi,
        "e": sp.E,
    }
    try:
        left_expr = sp.sympify(normalize_expression(left_text), locals=locals_map)
        right_expr = sp.sympify(normalize_expression(right_text), locals=locals_map)
    except Exception as exc:
        raise MathExpressionError(f"Invalid symbolic equation '{equation}'") from exc

    expression = left_expr - right_expr
    raw_solutions = (
        _numeric_symbolic_candidates(sp, expression, symbol)
        if _prefer_numeric_symbolic_path(sp, expression)
        else _sympy_solve_candidates(sp, expression, symbol)
    )
    verified = _verified_real_solutions(sp, expression, symbol, raw_solutions)
    if not verified:
        raise MathExpressionError("Equation has no verified real solution in the local symbolic solver")

    formatted = [_format_symbolic_solution(sp, solution) for solution in verified]
    return EquationSolveResult(
        solution=formatted,
        steps=[
            "Moved all terms to one side for symbolic solving.",
            f"Used the local symbolic solver because {fallback_reason.rstrip('.')}.",
            "Verified candidate roots in the original equation and rejected invalid or extraneous roots.",
        ],
        assumptions=["Real-valued solutions only.", "Single target variable only."],
        confidence="high",
    )


def _sympy_solve_candidates(sp: Any, expression: Any, symbol: Any) -> list[Any]:
    candidates: list[Any] = []
    try:
        solved = sp.solve(expression, symbol)
        if isinstance(solved, dict):
            solved = [solved.get(symbol)]
        candidates.extend(item for item in solved if item is not None)
    except Exception:
        pass
    if not candidates:
        try:
            solved_set = sp.solveset(expression, symbol, domain=sp.S.Reals)
            if getattr(solved_set, "is_FiniteSet", False):
                candidates.extend(list(solved_set))
        except Exception:
            pass
    return list(dict.fromkeys(candidates))


def _prefer_numeric_symbolic_path(sp: Any, expression: Any) -> bool:
    for power in expression.atoms(sp.Pow):
        exponent = getattr(power, "exp", None)
        if exponent is not None and exponent.is_Rational and not exponent.is_integer:
            return True
    return False


def _numeric_symbolic_candidates(sp: Any, expression: Any, symbol: Any) -> list[Any]:
    try:
        evaluator = sp.lambdify(symbol, expression, "math")
    except Exception:
        return []

    samples: list[tuple[float, float]] = []
    for index in range(801):
        value = -100.0 + (index * 0.25)
        try:
            y_value = float(evaluator(value))
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            continue
        if math.isfinite(y_value):
            samples.append((value, y_value))

    guesses: set[float] = set()
    previous: tuple[float, float] | None = None
    for current in samples:
        x_value, y_value = current
        if abs(y_value) <= 1e-7:
            guesses.add(x_value)
        if previous is not None:
            prev_x, prev_y = previous
            if prev_y == 0 or y_value == 0 or (prev_y < 0 < y_value) or (prev_y > 0 > y_value):
                guesses.add((prev_x + x_value) / 2.0)
        previous = current

    candidates: list[Any] = []
    for guess in sorted(guesses):
        try:
            root = sp.nsolve(expression, symbol, guess, verify=False, tol=1e-14, maxsteps=50)
        except Exception:
            continue
        candidates.append(sp.nsimplify(root))
    return list(dict.fromkeys(candidates))


def _verified_real_solutions(sp: Any, expression: Any, symbol: Any, candidates: list[Any]) -> list[Any]:
    verified: list[Any] = []
    for candidate in candidates:
        if not _is_real_solution(sp, candidate):
            continue
        try:
            residual = sp.simplify(expression.subs(symbol, candidate))
            if residual in {sp.zoo, sp.oo, -sp.oo, sp.nan}:
                continue
            if residual == 0:
                verified.append(sp.simplify(candidate))
                continue
            numeric_residual = complex(sp.N(residual, 16))
            if abs(numeric_residual) <= 1e-8:
                verified.append(sp.simplify(candidate))
        except Exception:
            continue
    return sorted(dict.fromkeys(verified), key=lambda item: float(sp.N(item)))


def _is_real_solution(sp: Any, candidate: Any) -> bool:
    if candidate is None:
        return False
    if getattr(candidate, "is_real", None) is True:
        return True
    try:
        numeric = complex(sp.N(candidate, 16))
    except Exception:
        return False
    return abs(numeric.imag) <= 1e-9


def _format_symbolic_solution(sp: Any, value: Any) -> str:
    simplified = sp.simplify(value)
    try:
        numeric = float(sp.N(simplified, 14))
    except Exception:
        numeric = None
    text = str(sp.sstr(simplified))
    denominator = getattr(simplified, "q", None)
    if numeric is not None and (len(text) > 56 or "RootOf" in text or (denominator and denominator > 10_000)):
        return _format_number(numeric)
    return text.replace("**", "^")


def _linear_solution_expression(coefficient_expr: str, constant_expr: str) -> str:
    if constant_expr == "0":
        return "0"
    if constant_expr.startswith("-"):
        numerator = constant_expr[1:].strip()
        if numerator.startswith("(") and numerator.endswith(")"):
            numerator = numerator[1:-1].strip()
    else:
        numerator = f"-({constant_expr})"
    if not any(ch.isalpha() for ch in f"{numerator}{coefficient_expr}"):
        value = evaluate_expression(f"({numerator})/({coefficient_expr})")
        return _format_number(value)
    if coefficient_expr == "1":
        return numerator
    return f"{numerator} / ({coefficient_expr})"


def trapezoid_integrate(expression: str, variable: str, lower: float, upper: float, steps: int) -> float:
    width = (upper - lower) / steps
    total = 0.5 * (
        evaluate_expression(expression, {variable: lower})
        + evaluate_expression(expression, {variable: upper})
    )
    for index in range(1, steps):
        total += evaluate_expression(expression, {variable: lower + (index * width)})
    return total * width


def simpson_integrate(expression: str, variable: str, lower: float, upper: float, steps: int) -> float:
    if steps % 2 != 0:
        raise MathExpressionError("Simpson integration requires an even number of steps")
    width = (upper - lower) / steps
    total = evaluate_expression(expression, {variable: lower}) + evaluate_expression(
        expression, {variable: upper}
    )
    for index in range(1, steps):
        factor = 4 if index % 2 else 2
        total += factor * evaluate_expression(expression, {variable: lower + (index * width)})
    return total * width / 3.0


def sampled_optimize(
    expression: str,
    variable: str,
    lower: float,
    upper: float,
    *,
    objective: str,
    samples: int,
) -> tuple[float, float]:
    if samples < 2:
        raise MathExpressionError("Optimization requires at least two samples")
    width = (upper - lower) / (samples - 1)
    best_x = lower
    best_y = evaluate_expression(expression, {variable: lower})
    for index in range(1, samples):
        current_x = lower + (index * width)
        current_y = evaluate_expression(expression, {variable: current_x})
        if objective == "maximize":
            if current_y > best_y:
                best_x, best_y = current_x, current_y
        else:
            if current_y < best_y:
                best_x, best_y = current_x, current_y
    return best_x, best_y
