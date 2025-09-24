"""Marimo notebook covering Higher Order Binary Optimization and quadratization techniques."""

import marimo

__generated_with = "0.16.0"
app = marimo.App(
    app_title="QUBO Higher Order Binary Optimization – QCobalt",
    html_head_file="",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Higher-Order Binary Optimisation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Binary optimisation problems can involve polynomial terms of varying degrees. When the objective function contains terms of degree higher than 2, we call it Higher-Order Binary Optimisation (<span style="font-variant-caps: small-caps">hobo</span>) or Polynomial Unconstrained Binary Optimisation (<span style="font-variant-caps: small-caps">pubo</span>).

    This lesson introduces quadratisation techniques that transform higher-order problems into <span style="font-variant-caps: small-caps">qubo</span> format, making them suitable for quantum annealing and other binary optimisation approaches.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Classifications

    Consider these example objective functions with binary variables:

    1. **Linear**: $f(x) = x_1 + x_2 - x_3$
    2. **Quadratic**: $f(x) = x_1 x_2 + x_2 x_3 - x_3 x_1 - 3$
    3. **Cubic**: $f(x) = x_1 x_2 x_3 + x_2 x_3 x_1 - x_3 x_1 x_2$

    Problems with polynomial terms of degree $k$ are called **$k$-local problems**:

    - <span style="font-variant-caps: small-caps">qubo</span> problems are **2-local**
    - <span style="font-variant-caps: small-caps">hobo</span> problems have terms of degree **3 or higher**

    Since quantum annealers and many optimisation devices are designed for quadratic problems, we need techniques to reduce higher-order terms to quadratic form through _quadratisation_ (also called _degree reduction_).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Quadratisation

    The Rosenberg method (1975) systematically reduces polynomial degree through variable substitution and penalty terms:

    1. **Identify** a product of two binary variables $x_i x_j$ from the highest-degree term
    2. **Replace** every instance of this product with a new binary variable $y_{ij}$
    3. **Add penalty term** to enforce $y_{ij} = x_i x_j$:
        <ul>
            <li>**Minimisation**: $C \cdot (x_i x_j - 2x_i y_{ij} - 2x_j y_{ij} + 3y_{ij})$</li>
            <li>**Maximisation**: Negate the penalty term</li>
        </ul>

    The penalty coefficient $C$ must be chosen large enough to ensure the constraint $y_{ij} = x_i x_j$ is satisfied in the optimal solution.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 1**

    Consider the following higher-order problem and reformulate it as a <span style="font-variant-caps: small-caps">qubo</span> problem using Rosenberg's method:

    $$f(x_1, x_2, x_3) = 5x_1 + 7x_1 x_2 - 3x_1 x_2 x_3$$

    _Hint: Start by identifying the highest-degree term and systematically apply the substitution process._
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 2**

    Consider a more complex higher-order problem and reformulate it as a <span style="font-variant-caps: small-caps">qubo</span> problem:

    $$f(x_1, x_2, x_3, x_4) = 5x_1 + 7x_1 x_2 + 2x_1x_2x_3 - 3x_1 x_2 x_3 x_4$$

    _Note: This problem contains both cubic and quartic terms, requiring multiple substitutions._
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## <span style="font-variant-caps: small-caps">max-3sat</span>

    <span style="font-variant-caps: small-caps">hobo</span> formulations naturally arise in combinatorial problems. <span style="font-variant-caps: small-caps">max-3sat</span> is a classic example where quadratisation is essential.

    ### Review

    A **Boolean formula** in Conjunctive Normal Form (<span style="font-variant-caps: small-caps">cnf</span>):
    $$\psi = C_1 \wedge C_2 \wedge \cdots \wedge C_n$$

    where each clause $C_i = y_1 \vee y_2 \vee \cdots \vee y_k$ contains literals $y_k \in \{x_k, \overline{x_k}\}$.

    **<span style="font-variant-caps: small-caps">3sat</span>**: Each clause contains at most 3 literals  
    **<span style="font-variant-caps: small-caps">max-3sat</span>**: Maximise the number of satisfied clauses
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### <span style="font-variant-caps: small-caps">max-3sat</span> <span style="font-variant-caps: small-caps">hobo</span> Formulation

    Each clause type yields a different objective function that equals 1 when satisfied:

    | Clause Type | Example | Objective Function |
    |-------------|---------|-------------------|
    | **Zero negations** | $(x_i \vee x_j \vee x_k)$ | $x_i + x_j + x_k - x_i x_j - x_i x_k - x_j x_k + x_i x_j x_k$ |
    | **One negation** | $(x_i \vee x_j \vee \overline{x_k})$ | $1 - x_k + x_i x_k + x_j x_k - x_i x_j x_k$ |
    | **Two negations** | $(x_i \vee \overline{x_j} \vee \overline{x_k})$ | $1 - x_j x_k + x_i x_j x_k$ |
    | **Three negations** | $(\overline{x_i} \vee \overline{x_j} \vee \overline{x_k})$ | $1 - x_i x_j x_k$ |

    **<span style="font-variant-caps: small-caps">max-3sat</span> <span style="font-variant-caps: small-caps">hobo</span>**: $$\max \sum_{i=1}^n g_i(x)$$ where $g_i$ is the objective for clause $i$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### <span style="font-variant-caps: small-caps">hobo</span> → <span style="font-variant-caps: small-caps">qubo</span>

    Rosenberg's quadratisation method transforms the <span style="font-variant-caps: small-caps">hobo</span> into <span style="font-variant-caps: small-caps">qubo</span> format:

    - **Introduce auxiliary variables** $Y$ to replace higher-degree terms 
    - **Add penalty constraints** for each substitution:
    $$\max \sum_{i=1}^n g_i(X, Y) - C \sum_{ij \in Y} (x_i x_j - 2x_i y_{ij} - 2x_j y_{ij} + 3y_{ij})$$

    **Key insight**: Each clause generates exactly one cubic term, so the augmented <span style="font-variant-caps: small-caps">qubo</span> has at most $n$ auxiliary variables for $n$ clauses.

    The optimal solution value corresponds to the number of satisfied clauses $(\leq n)$ in the original <span style="font-variant-caps: small-caps">max-3sat</span> problem.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 3**

    Given the following <span style="font-variant-caps: small-caps">max-3sat</span> formulation with 4 literals and 2 clauses:

    $$(x_1 \vee x_2 \vee x_3) \wedge (x_1 \vee x_4 \vee \overline{x_3})$$

    1. **Write the <span style="font-variant-caps: small-caps">hobo</span> formulation** using the clause objective functions above
    2. **Apply quadratisation** to transform it into a <span style="font-variant-caps: small-caps">qubo</span> problem

    > Each cubic term requires one auxiliary variable.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Interactive Example: Clause Evaluation

    Explore how different variable assignments affect clause satisfaction:
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, x1_switch, x2_switch, x3_switch, x4_switch):
    _x1, _x2, _x3, _x4 = (
        x1_switch.value,
        x2_switch.value,
        x3_switch.value,
        x4_switch.value,
    )

    # Clause 1: (x1 ∨ x2 ∨ x3) - zero negations
    _clause1_satisfied = _x1 or _x2 or _x3
    _clause1_obj = (
        _x1 + _x2 + _x3 - _x1 * _x2 - _x1 * _x3 - _x2 * _x3 + _x1 * _x2 * _x3
    )

    # Clause 2: (x1 ∨ x4 ∨ ¬x3) - one negation
    _clause2_satisfied = _x1 or _x4 or (not _x3)
    _clause2_obj = 1 - _x3 + _x1 * _x3 + _x4 * _x3 - _x1 * _x4 * _x3

    _total_satisfied = int(_clause1_satisfied) + int(_clause2_satisfied)
    _total_objective = _clause1_obj + _clause2_obj

    mo.vstack(
        [
            mo.hstack([x1_switch, x2_switch, x3_switch, x4_switch]),
            mo.md(
                f"**Assignment**: $(x_1, x_2, x_3, x_4) = ({int(_x1)}, {int(_x2)}, {int(_x3)}, {int(_x4)})$"
            ),
            mo.md(
                f"**Clause 1**: $(x_1 \\vee x_2 \\vee x_3)$ → {'✓ Satisfied' if _clause1_satisfied else '✗ Not satisfied'} (objective = {_clause1_obj})"
            ),
            mo.md(
                f"**Clause 2**: $(x_1 \\vee x_4 \\vee \\overline{{x_3}})$ → {'✓ Satisfied' if _clause2_satisfied else '✗ Not satisfied'} (objective = {_clause2_obj})"
            ),
            mo.md(f"**Total satisfied clauses**: {_total_satisfied}/2"),
            mo.md(
                f'**<span style="font-variant-caps: small-caps">hobo</span> objective value**: {_total_objective}'
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    penalty_coefficient = mo.ui.slider(
        start=1,
        stop=10,
        step=1,
        value=5,
        label="Penalty coefficient C",
        show_value=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    x1_switch = mo.ui.switch(value=False, label="$x_1$")
    x2_switch = mo.ui.switch(value=True, label="$x_2$")
    x3_switch = mo.ui.switch(value=False, label="$x_3$")
    x4_switch = mo.ui.switch(value=True, label="$x_4$")
    return x1_switch, x2_switch, x3_switch, x4_switch


if __name__ == "__main__":
    app.run()
