import pytest

from lumen.tools.math_core import MathExpressionError, solve_equation


def test_solve_equation_handles_biquadratic_polynomial() -> None:
    result = solve_equation("x^4 - 10x^2 + 9 = 0", "x")

    assert result.solution == ["-3", "-1", "1", "3"]
    assert "local symbolic solver" in " ".join(result.steps).lower()


def test_solve_equation_handles_rational_equation_and_rejects_invalid_domains() -> None:
    result = solve_equation("(x+3)/(x-2) + (x-2)/(x+3) = 25/6", "x")

    joined = " ".join(result.solution)
    assert "sqrt(481)" in joined
    assert "2" not in result.solution
    assert "-3" not in result.solution


def test_solve_equation_handles_radical_equation_with_verified_real_root() -> None:
    result = solve_equation("sqrt(x+6) + sqrt(3x-2) = 2x", "x")

    assert result.solution == ["2.71584"]
    assert "extraneous roots" in " ".join(result.steps).lower()


def test_solve_equation_rejects_multivariable_symbolic_fallback() -> None:
    with pytest.raises(MathExpressionError, match="one target variable"):
        solve_equation("x^3 + y = 0", "x")
