"""Marimo notebook introducing the mathematical formulation of QUBO."""

import marimo

__generated_with = "0.16.0"
app = marimo.App(app_title="QUBO Definition – QCobalt", html_head_file="")


@app.cell(hide_code=True)
def _(mo):
	mo.md(r"""# Quadratic Unconstrained Binary Optimisation""")
	return


@app.cell(hide_code=True)
def _(mo):
	mo.md(
		r"""
    Quadratic unconstrained binary optimisation (<span style="font-variant-caps: small-caps">qubo</span>) captures a wide range of combinatorial optimisation problems. Every decision variable is binary (either $0$ or $1$), and the objective function is quadratic function of those variables.

    In the previous notebook you met several problems that can
    be expressed in this form; here we focus on their mathematical structure.
    """
	)
	return


@app.cell(hide_code=True)
def _(mo):
	mo.md(
		r"""
    ## Definition

    A <span style="font-variant-caps: small-caps">qubo</span> instance is specified by an $n 	\times n$ matrix, $Q$, and a binary vector, $\mathbf{x} \in \{0, 1\}^n$. The matrix is usually taken to be symmetric or upper triangular; we will work with the upper-triangular form for convenience. The vector $\mathbf x$ is a collection of $n$ binary decision variables.

    The goal is to minimise the objective function
    $$f(\mathbf{x}) = \sum_i Q_{i,i} x_i + \sum_{i < j} Q_{i,j} x_i x_j,$$
    where

    * $x_i$ is the $i$<sup>th</sup> entry of $\mathbf x$,
    * the diagonal entries, $Q_{i,i}$, are the _linear_ coefficients of $f$, and
    * the off-diagonal entries, $Q_{i,j}$, are its _quadratic_ coefficients.

    More compactly we write
    $$\min_{\mathbf{x} \in \{0, 1\}^n} \mathbf{x}^T Q \mathbf{x}.$$

    Maximisation problems fit the same template: maximising $\mathbf{x}^T Q \mathbf{x}$ is equivalent to minimising $-\mathbf{x}^T Q \mathbf{x}$.
    """
	)
	return


@app.cell(hide_code=True)
def _(mo):
	mo.md(
		r"""
    ## **Task 1**

    Find the assignment of $x_1$ and $x_2$ that minimises
    $$f(x_1, x_2) = 5 x_1 + 7 x_1 x_2 - 3 x_2.$$
    Write down the value of the objective function for each binary assignment
    and identify the minimiser. The helper below evaluates the function for the
    arguments you provide.
    """
	)
	return


@app.cell(hide_code=True)
def _(mo, task1_f, task1_x1, task1_x2):
	mo.vstack(
		[
			task1_x1,
			task1_x2,
			mo.md(
				f"$f({int(task1_x1.value)}, {int(task1_x2.value)}) = {task1_f(task1_x1.value, task1_x2.value)}$"
			),
		]
	)
	return


@app.cell(hide_code=True)
def _(mo):
	mo.md(
		r"""
    ## **Task 2**

    Repeat the experiment with four decision variables. Determine the assignment of $x_1, x_2, x_3,$ and $x_4$ that minimises
    $$f(x_1, x_2, x_3, x_4) = -5 x_1 - 3 x_2 - 8 x_3 - 6 x_4 + 4 x_1 x_2 + 8 x_1 x_3 + 2 x_2 x_3 + 10 x_3 x_4.$$
    Again, evaluate the cost for each combination of binary choices and report the minimum.
    """
	)
	return


@app.cell(hide_code=True)
def _(mo, task2_f, task2_x1, task2_x2, task2_x3, task2_x4):
	mo.vstack(
		[
			mo.hstack([task2_x1, task2_x2], gap="1rem"),
			mo.hstack([task2_x3, task2_x4], gap="1rem"),
			mo.md(
				f"$f({int(task2_x1.value)}, {int(task2_x2.value)}, {int(task2_x3.value)}, {int(task2_x4.value)}) = {task2_f(task2_x1.value, task2_x2.value, task2_x3.value, task2_x4.value)}$"
			),
		],
		align="start",
		gap="0.5rem",
	)
	return


@app.cell(hide_code=True)
def _(mo):
	mo.md(
		r"""
    In the tasks above, toggling a switch between "on" and "off" mirrors a yes-or-no decision. Real optimisation problems often contain hundreds or thousands of these binary choices.

    Think back to the travelling salesperson problem from the previous notebook: should the tour visit city $A$ before city $B$? Perhaps it is better to head to city $C$ first?

    Task&nbsp;1 is manageable because just two variables yield four possible assignments. Task&nbsp;2, with four variables, already demands consideration of $2^4 = 16$ combinations. The number of possibilities grows exponentially with the number of binary decisions.

    We can partially overcome this scaling challenge by packaging the objective function into a matrix, $Q$.
    """
	)
	return


@app.cell(hide_code=True)
def _(mo):
	mo.md(
		r"""
    ## Matrix form

    We express a <span style="font-variant-caps: small-caps">qubo</span> as
    $$\min_{\mathbf{x} \in \{0, 1\}^n} \mathbf{x}^T Q \mathbf{x}.$$
    To keep the representation tidy, we store $Q$ in an upper-triangular form. For four variables, for example,
    $$Q = \begin{bmatrix}
        Q_{11} & Q_{12} & Q_{13} & Q_{14} \\
        0      & Q_{22} & Q_{23} & Q_{24} \\
        0      & 0      & Q_{33} & Q_{34} \\
        0      & 0      & 0      & Q_{44}
    \end{bmatrix}.$$

    Populate the matrix as follows:

    - place each linear coefficient on the diagonal; and
    - record the coefficient of $x_i x_j$ in the entry $Q_{ij}$.

    Because binary variables satisfy $x^2 \equiv  x$, moving linear terms to the diagonal leaves the objective unchanged. Expanding $\mathbf{x}^T Q \mathbf{x}$ shows that $Q_{11}$ becomes the coefficient of $x_1^2$, $Q_{12}$ the coefficient of $x_1 x_2$, and so on. Any entry not associated with a term in the objective can simply be set to $0$.
    """
	)
	return


@app.cell(hide_code=True)
def _(mo):
	mo.md(
		r"""
    ## **Task 3**

    Show that $x \in \{\, 0, 1 \,\} \implies x^2 = x$ and explain how it lets us place linear terms on the diagonal of $Q$.
    """
	)
	return


@app.cell(hide_code=True)
def _(mo):
	mo.md(
		r"""
    ## Worked example

    Revisit Task&nbsp;2 and encode its objective function in matrix form. Start by placing the linear coefficients on the diagonal of $Q$. Next, distribute each quadratic coefficient across the corresponding symmetric pair so that $Q$ remains upper triangular (or symmetric if you prefer). Finally, evaluate the objective with the helper below.
    """
	)
	return


@app.cell
def _():
	from foundation import optimise

	_Q: list[list[float]] = [
		[-5, 4, 8, 0],
		[0, -3, 2, 0],
		[0, 0, -8, 10],
		[0, 0, 0, -6],
	]
	(_min_value, _optimal_vector) = optimise(_Q)

	print(
		f"A minimum value of {_min_value} is achieved with the optimal vector {_optimal_vector}"
	)
	return


@app.cell(hide_code=True)
def _(mo):
	mo.md(
		r"""The `optimise` helper performs an exhaustive search over all binary vectors of length $n$, evaluates $\mathbf{x}^T Q \mathbf{x}$ for each candidate, and returns the best score together with its minimiser. It is good for our toy examples but it does not scale to large-scale industrial problems."""
	)
	return


@app.cell(hide_code=True)
def _(mo):
	mo.md(
		r"""
    ## **Task 4**

    Use the matrix formulation to revisit Task&nbsp;1. Build a $2 \times 2$ matrix representing the objective function and call `optimise` to verify your answer. (You won't need to import the function again, as it has already been imported in an earlier cell.)
    """
	)
	return


@app.cell(hide_code=True)
def _(mo):
	mo.md(
		r"""
    ## **Task 5**

    Formulate the maximisation problem
    $$f(x_1, x_2, x_3, x_4) = x_1 + x_2 + x_3 + x_4 - 6 x_1 x_3 - 6 x_1 x_4 - 6 x_2 x_3 - 6 x_2 x_4 - 6 x_3 x_4$$
    in matrix form. Remember that maximising $f$ is equivalent to minimising $-f$.
    """
	)
	return


@app.cell(hide_code=True)
def _(mo):
	mo.md(
		r"""
    ## **Task 6** <span style="font-variant-caps: small-caps">optional</span>

    Implement your own exhaustive <span style="font-variant-caps: small-caps">qubo</span> solver that mirrors the behaviour of `optimise`. Given a matrix $Q$, it should return `(min_value, optimal_vector)`.
    """
	)
	return


@app.cell(hide_code=True)
def _():
	import marimo as mo

	return (mo,)


@app.cell(hide_code=True)
def _(mo):
	def task1_f(x1: bool, x2: bool) -> float:
		return 5.0 * x1 + 7.0 * x1 * x2 - 3.0 * x2

	task1_x1 = mo.ui.switch(value=False, label="$x_1$")
	task1_x2 = mo.ui.switch(value=False, label="$x_2$")
	return task1_f, task1_x1, task1_x2


@app.cell(hide_code=True)
def _(mo):
	def task2_f(x1: bool, x2: bool, x3: bool, x4: bool) -> float:
		return (
			-5.0 * x1
			- 3.0 * x2
			- 8.0 * x3
			- 6.0 * x4
			+ 4.0 * x1 * x2
			+ 8.0 * x1 * x3
			+ 2.0 * x2 * x3
			+ 10.0 * x3 * x4
		)

	task2_x1 = mo.ui.switch(value=False, label="$x_1$")
	task2_x2 = mo.ui.switch(value=False, label="$x_2$")
	task2_x3 = mo.ui.switch(value=False, label="$x_3$")
	task2_x4 = mo.ui.switch(value=False, label="$x_4$")
	return task2_f, task2_x1, task2_x2, task2_x3, task2_x4


if __name__ == "__main__":
	app.run()
