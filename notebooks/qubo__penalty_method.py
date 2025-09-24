"""Marimo notebook introducing the penalty method for QUBO formulations."""

import marimo

__generated_with = "0.16.0"
app = marimo.App(app_title="QUBO Penalty Method – QCobalt", html_head_file="")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""# <span style="font-variant-caps: small-caps">qubo</span> Penalty Method"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Quadratic _unconstrained_ binary optimisation (<span style="font-variant-caps: small-caps">qubo</span>) problems do not include explicit constraints. Many real decision models, however, impose requirements that feasible solutions must satisfy. This notebook shows how to absorb those constraints into a <span style="font-variant-caps: small-caps">qubo</span> objective with the **penalty method**.
    We begin with a small constrained objective, then catalogue common penalty terms and learn how to derive new ones for equality and inequality constraints involving integer variables.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Constrained Optimisation
    A constrained optimisation problem pairs an objective function with one or more restrictions that admissible solutions must satisfy. We denote the $n$ decision variables collectively by $\mathbf{x}$, the objective function by $f(\mathbf{x})$, and each constraint by $c_i(\mathbf{x}) = 0$ or $c_i(\mathbf{x}) \leq 0$.
    In a <span style="font-variant-caps: small-caps">qubo</span> problem, every component of $\mathbf{x}$ is binary, so feasible solutions sit inside $\{0, 1\}^n$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 1**

    Consider the objective function $f(x_1, x_2) = 5 x_1 + 7 x_1 x_2 - 3 x_2$ subject to the constraint $x_1 - x_2 = 0$. Determine the minimising assignment while respecting the constraint. Use the switches below to explore every combination.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, task1_objective, task1_residual, task1_x1, task1_x2):
    mo.vstack(
        [
            task1_x1,
            task1_x2,
            mo.md(
                f"$f({int(task1_x1.value)}, {int(task1_x2.value)}) = {task1_objective(task1_x1.value, task1_x2.value)}$"
            ),
            mo.md(
                f"$x_1 - x_2 = {task1_residual(task1_x1.value, task1_x2.value)}$"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Penalty Method

    The penalty method converts a constrained optimisation problem into an unconstrained one by adding a penalty term for each constraint. Given an objective function, $f(\mathbf{x})$, and constraint functions $c_i(\mathbf{x})$ that vanish on feasible solutions, we consider
    $$f(\mathbf{x}) + \sum_i P_i c_i(\mathbf{x}).$$
    Each coefficient $P_i > 0$ controls how severely constraint violations are punished. Choose $c_i$ so that it equals zero whenever the constraint holds and is positive otherwise.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Templates

    For binary variables, the following quadratic penalties enforce frequently encountered constraints.

    | Classical constraint | Quadratic penalty |
    | :-- | :-- |
    | $x + y \leq 1$ | $xy$ |
    | $x + y \geq 1$ | $1 - x - y + xy$ |
    | $x + y = 1$ | $1 - x - y + 2xy$ |
    | $x \leq y$ | $x - xy$ |
    | $x_1 + x_2 + x_3 \leq 1$ | $x_1 x_2 + x_1 x_3 + x_2 x_3$ |
    | $x = y$ | $x + y - 2xy$ |

    > All variables in the table are binary. Every penalty evaluates to zero if and only if the associated constraint is satisfied.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 2**
    Consider an objective function $f(x_1, x_2)$ subject to the constraint $x_1 + x_2 \leq 1$. Express the corresponding <span style="font-variant-caps: small-caps">qubo</span> using the penalty method. You may leave the penalty coefficient as a symbolic $P$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 3**
    Consider an objective function $f(x_1, x_2)$ subject to the constraint $x_1 + x_2 = 1$. Express the resulting <span style="font-variant-caps: small-caps">qubo</span>, keeping the penalty coefficient symbolic as $P$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 4**
    Rewrite the constrained problem from Task&nbsp;1 as a <span style="font-variant-caps: small-caps">qubo</span>. Leave the penalty coefficient as $P$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Linear Equality Constraints

    > To keep the discussion general, we temporarily allow integer-valued decision variables. We will show later how to convert integer variables to binary variables.

    Suppose we minimise $f(x_1, \dots, x_n)$ subject to a linear equality
    $$\sum_{i=1}^{n} a_i x_i = b, \qquad a_i, b \in \mathbb{R}.$$
    The penalty method replaces the constraint with
    $$f(x_1, \dots, x_n) + P \left(\sum_{i=1}^{n} a_i x_i - b\right)^2,$$
    producing an unconstrained objective that discourages violations.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 5**
    Let $f(x_1, x_2, x_3) = -5 x_1 - 2 x_1 x_2 - 3 x_3$ with the constraint $x_1 + x_2 + x_3 = 1$. Use the penalty method to express the problem as a <span style="font-variant-caps: small-caps">qubo</span>, keeping $P$ symbolic.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Choosing Penalty Coefficients

    Selecting an appropriate penalty coefficient requires balancing feasibility and numerical stability. If $P$ is too small, the optimiser may prefer an infeasible point with a lower objective value. If $P$ is too large, numerical routines may struggle to distinguish between feasible solutions. In practice we choose $P$ just large enough to enforce the constraint.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 6**
    Using your formulation from Task&nbsp;5, build the associated $Q$ matrix and explore how different choices of $P$ influence the optimum. Identify the smallest value of $P$ that yields a feasible solution. Remember to add the constant term after evaluating $\mathbf{x}^T Q \mathbf{x}$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Linear Inequality Constraints

    A linear inequality of the form $$\sum_{i=1}^{n} a_i x_i \leq b$$ can be converted into an equality by introducing a _slack variable,_ $\eta \geq 0$:
    $$\sum_{i=1}^{n} a_i x_i + \eta = b.$$
    Apply the penalty method to this equality and remember that the slack variable must eventually be encoded with binary variables as well.

    > When a constraint appears as $\sum_{i=1}^{n} a_i x_i \geq b$, it is usually convenient to multiply through by $-1$ so that it becomes $\leq$ before introducing the slack variable.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Integer Variables → Binary Variables

    Consider an integer variable $y_i$ with bounds $\underline{y_i} \leq y_i \leq \overline{y_i}$. Define $N = \lceil \log_2(\overline{y_i} - \underline{y_i} + 1) \rceil$. Then $y_i$ can be written in terms of binary variables $x_j^i$ as
    $$\underline{y_i} + \sum_{j=0}^{N-2} 2^j x_j^i + \bigl(\overline{y_i} - \underline{y_i} - \sum_{j=0}^{N-2} 2^j\bigr) x_{N-1}^i.$$
    Apply the same construction to any slack variable $\eta$, choosing bounds from the constraint via
    $$0 \leq \eta \leq b - \sum_{i=1}^{k} \min \{a_i \underline{y_i}, a_i \overline{y_i}\}.$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 7**
    Let $y_1$ and $y_2$ be integer variables with $0 \leq y_1 \leq 8$ and $0 \leq y_2 \leq 5$, subject to the constraint $y_1 + y_2 \geq 10$. Express the problem as a <span style="font-variant-caps: small-caps">qubo</span>, keeping the penalty coefficient symbolic as $P$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 8**
    Let $y_1$ and $y_2$ be integer variables with $2 \leq y_1 \leq 8$ and $3 \leq y_2 \leq 5$, subject to the constraint $y_1 + 2 y_2 \leq 10$. Express the problem as a <span style="font-variant-caps: small-caps">qubo</span>, again leaving the penalty coefficient as $P$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Integer Programming
    Integer linear programming (<span style="font-variant-caps: small-caps">ilp</span>) minimises a linear objective over integer variables subject to linear constraints:
    $$\text{minimise} \quad \sum_j c_j y_j, \qquad \text{subject to} \quad \sum_j A_{ij} y_j \leq b_i, \; y_j \geq 0, \; y_j \in \mathbb{Z}.$$
    When the objective is quadratic, the model becomes an integer quadratic programme (<span style="font-variant-caps: small-caps">iqp</span>). By combining the penalty method with binary encodings of integer variables, any <span style="font-variant-caps: small-caps">ilp</span> or <span style="font-variant-caps: small-caps">iqp</span> can be translated into an equivalent <span style="font-variant-caps: small-caps">qubo</span>.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    def task1_objective(x1: bool, x2: bool) -> float:
        x1_int = int(x1)
        x2_int = int(x2)
        return 5.0 * x1_int + 7.0 * x1_int * x2_int - 3.0 * x2_int


    def task1_residual(x1: bool, x2: bool) -> int:
        return int(x1) - int(x2)


    task1_x1 = mo.ui.switch(value=False, label="$x_1$")
    task1_x2 = mo.ui.switch(value=False, label="$x_2$")
    return task1_objective, task1_residual, task1_x1, task1_x2


if __name__ == "__main__":
    app.run()
