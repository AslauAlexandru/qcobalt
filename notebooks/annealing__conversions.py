import marimo

__generated_with = "0.16.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Conversions

    In this notebook, we will look at the different ways of converting a problem between <span style="font-variant: small-caps;">qubo</span>, Ising and <span style="font-variant: small-caps;">bqm</span> formulations.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## <span style="font-variant: small-caps;">qubo → bqm</span> via $Q$ 

    It is possible to construct a Binary Quadratic Model from a matrix using the `BQM` constructor.

    Let's consider the following objective function:

    $$f(x_1, x_2, x_3, x_4) = - 5x_1 - 3x_2 - 8x_3 - 6x_4 + 4x_1 x_2 + 8x_1 x_3 + 2x_2 x_3 + 10x_3 x_4$$

    The <span style="font-variant: small-caps;">qubo</span> matrix $Q$ for the objective function is:

    $$
    Q = \begin{bmatrix}
    	-5  &  4   &  8   &  0  \\
    	0   &  -3  &  2   &  0  \\
    	0   &  0   &  -8  &  10  \\
    	0   &  0   &  0   &  -6  \\
    \end{bmatrix}
    $$
    """
    )
    return


@app.cell
def _():
    Q = [[-5, 4, 8, 0], [0, -3, 2, 0], [0, 0, -8, 10], [0, 0, 0, -6]]
    return (Q,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### <span style="font-variant: small-caps;">bqm</span> Constructor - Parameters

    - `Q` - The <span style="font-variant: small-caps;">qubo</span> as a matrix (NumPy array or list of lists)
    - `BINARY` - Variable type

    Now let's create a <span style="font-variant: small-caps;">bqm</span> from the above matrix. We have to pass `Q` as an argument to the `BQM` constructor.
    """
    )
    return


@app.cell
def _(BQM, Q):
    BQM(Q, "BINARY")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## <span style="font-variant: small-caps;">qubo → bqm</span> via `dict`

    We can also represent a <span style="font-variant: small-caps;">qubo</span> problem as a dictionary. What is the need for it you may ask? Dictionary representations can be very helpful for problems with a large number of variables.

    In the dictionary representation, only the non-zero terms of a <span style="font-variant: small-caps;">qubo</span> matrix are considered. This saves up space and improves the efficiency of the problem solving process.

    Let's consider a $3 \times 3$ matrix:

    $$
    Q = \begin{bmatrix}
    	Q_{11} & Q_{12} & Q_{13}  \\
    	0      & Q_{22} & Q_{23}  \\
    	0      & 0      & Q_{33}  \\
    \end{bmatrix}
    $$

    In the dictionary representation, the keys should be the binary variables and their values should be the coefficients associated with these binary variables. The variables can be represented either as a tuple of variable names or as a tuple of numbers. The key for the term $Q_{11}$ in the above matrix can be represented as:

    - `('x1', 'x1')` - Tuple of variable names
    - `(0, 0)` - Tuple of numbers that indicate the position of the term in the matrix

    > The advantage of dictionary representation becomes apparent when we consider a large <span style="font-variant: small-caps;">qubo</span> matrix.

    $$
    Q_L = \begin{bmatrix}
    	\mathbf{3} & 0 & 0 & 0 & 0 & \mathbf{4} & 0 & 0 \\
    	0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    	0 & 0 & 0 & 0 & 0 & 0 & \mathbf{9} & 0 \\
    	0 & 0 & 0 & \mathbf{1} & 0 & 0 & 0 & 0 \\
    	0 & 0 & 0 & 0 & \mathbf{4} & 0 & 0 & 0 \\
    	0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    	0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    	0 & 0 & 0 & 0 & 0 & 0 & 0 & \mathbf{8} \\
    \end{bmatrix}
    $$

    The above matrix $Q_L$ can be represented as a much more compact dictionary.
    """
    )
    return


@app.cell
def _(BQM):
    Q_Large = {
        ("x1", "x1"): 3,
        ("x4", "x4"): 1,
        ("x5", "x5"): 4,
        ("x8", "x8"): 8,
        ("x1", "x6"): 4,
        ("x3", "x7"): 9,
    }
    BQM.from_qubo(Q_Large)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 1**

    Create a <span style="font-variant: small-caps;">qubo</span> in matrix form for the following objective function and create a <span style="font-variant: small-caps;">bqm</span> from it.

    $$f(x_1, x_2, x_3, x_4) = 3x_1 - 7x_2 + 11x_3 - x_4 + 9x_1 x_2 + x_1 x_3 + 2x_2 x_3 + 8x_3 x_4$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 2**

    Create a <span style="font-variant: small-caps;">qubo</span> dictionary form for the following objective function and create a <span style="font-variant: small-caps;">bqm</span> from it.

    $$f(x_1, x_2, x_3, x_4) = 3x_1 - 7x_2 + 11x_3 - x_4 + 9x_1 x_2 + x_1 x_3 + 2x_2 x_3 + 8x_3 x_4$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## <span style="font-variant: small-caps;">bqm → qubo</span>

    ### Getting $Q$ matrix

    It is possible to reconstruct a <span style="font-variant: small-caps;">qubo</span> matrix from a Binary Quadratic Model using the `to_numpy_vectors`.

    #### `to_numpy_vectors` Parameters

    - `variable_order` - Variable order as a list (optional)
    - `sort_indices` - Sort indices for consistent ordering (defaults to False)
    - `dtype` - Data type for arrays (defaults to float)

    Let us consider the following <span style="font-variant: small-caps;">bqm</span> where the keys are variable names as strings:
    """
    )
    return


@app.cell
def _(BQM, np):
    bqm = BQM(
        {"x1": -5.0, "x2": -3.0, "x3": -8.0, "x4": -6.0},
        {("x1", "x2"): 4, ("x1", "x3"): 8, ("x2", "x3"): 2, ("x3", "x4"): 10},
        0,
        "BINARY",
    )

    _variable_order = ["x1", "x2", "x3", "x4"]
    _linear, (_row, _col, _quadratic), _offset = bqm.to_numpy_vectors(
        variable_order=_variable_order, sort_indices=True
    )

    # Reconstruct the Q matrix
    _n = len(_variable_order)
    _Q = np.zeros((_n, _n))
    np.fill_diagonal(_Q, _linear)  # Linear terms on diagonal
    _Q[_row, _col] = _quadratic  # Quadratic terms

    print(_Q)
    return (bqm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""And another one where the keys are numeric:""")
    return


@app.cell
def _(BQM, np):
    _bqm = BQM(
        {0: -5.0, 1: -3.0, 2: -8.0, 3: -6.0},
        {(0, 1): 4, (0, 2): 8, (1, 2): 2, (2, 3): 10},
        0.0,
        "BINARY",
    )

    # Using to_numpy_vectors with numeric indices (no variable_order needed)
    _linear, (_row, _col, _quadratic), _offset = _bqm.to_numpy_vectors(
        sort_indices=True
    )

    # Reconstruct the Q matrix
    _Q = np.zeros((_bqm.num_variables, _bqm.num_variables))
    np.fill_diagonal(_Q, _linear)
    _Q[_row, _col] = _quadratic

    print(_Q)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 3**

    Obtain the $Q$ matrix for the <span style="font-variant: small-caps;">bqm</span> formulation you created in Task 2.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Getting <span style="font-variant: small-caps;">qubo</span> dictionary

    The `to_qubo` method can be used to construct a <span style="font-variant: small-caps;">qubo</span> model from a <span style="font-variant: small-caps;">bqm</span>. If the `vartype` of the <span style="font-variant: small-caps;">bqm</span> is `'SPIN'`, it is converted to `'BINARY'`.

    This method returns a tuple of form `(biases, offset)` where `biases` is a dictionary of the linear and quadratic terms and `offset` is a number.

    Let's consider the same `bqm` used in the previous example. The <span style="font-variant: small-caps;">qubo</span> form of the <span style="font-variant: small-caps;">bqm</span> is:
    """
    )
    return


@app.cell
def _(bqm, mo):
    qubo = bqm.to_qubo()

    mo.md(
        f"""
    **Converting <span style="font-variant: small-caps;">bqm</span> to <span style="font-variant: small-caps;">qubo</span>:**
    ```python
    {qubo}
    ```

    In the above output:

    - The first term of the tuple corresponds to the linear and quadratic terms of the <span style="font-variant: small-caps;">qubo</span>:
    	```python
    	{qubo[0]}
    	```

    - The second term corresponds to the offset: `{qubo[1]}`
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 4**

    Obtain the <span style="font-variant: small-caps;">qubo</span> dictionary for the <span style="font-variant: small-caps;">bqm</span> formulation you created in Task 1.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Ising → <span style="font-variant: small-caps;">bqm</span> via `dict`

    As we have already discussed, it is good to know how to represent our problem as a dictionary. We can define the $h$ and $J$ coefficients as two separate dictionaries.

    The keys of the dictionary can either be variable names or numbers indicating the position of a particular term.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    h = {"s1": 3, "s2": 1, "s3": 4, "s4": 2}
    J = {("s1", "s2"): 4, ("s1", "s3"): 1, ("s1", "s4"): 6, ("s3", "s4"): 7}

    mo.md(
        f"""
    **Dictionary representation for $h$ (external fields):**
    ```python
    {h}
    ```

    **Dictionary representation for $J$ (coupling strengths):**
    ```python
    {J}
    ```

    A `BinaryQuadraticModel` can be constructed from an Ising Model using the `from_ising`.

    #### `from_ising` method - parameters

    - `h` - The linear terms should be passed as a dictionary or a list. If it is passed as a list, the indices would be the variable labels.
    - `J` - The quadratic terms should be passed as a dictionary.
    - `offset` (optional) - Constant offset
    """
    )
    return J, h


@app.cell
def _(BQM, J, h):
    BQM.from_ising(h, J)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 5**

    Create dictionaries $h$ and $J$ for the following Ising model and obtain <span style="font-variant: small-caps;">bqm</span> model using `from_ising` function.

    $$s_1 + s_2 + s_3 + s_4 - 6s_1 s_3 - 6s_1 s_4 - 6s_3 s_4 - 6s_1 s_2$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## <span style="font-variant: small-caps;">bqm</span> → Ising

    Similarly an Ising Model can be constructed from a Binary Quadratic Model using the `to_ising` method of the `BinaryQuadraticModel` class. If the `vartype` of the <span style="font-variant: small-caps;">bqm</span> is `'BINARY'`, it is converted to `'SPIN'`.

    ### Getting Ising dictionary

    `to_ising` method returns a tuple of form `(linear, quadratic, offset)` where `linear` and `quadratic` are dictionaries and `offset` is a number.

    Let's consider the following <span style="font-variant: small-caps;">bqm</span> instance:
    """
    )
    return


@app.cell
def _(BQM):
    bqm_example = BQM(
        {"s1": 3.0, "s2": 1.0, "s3": 4.0, "s4": 2.0},
        {("s1", "s2"): 4, ("s1", "s3"): 1, ("s1", "s4"): 6, ("s3", "s4"): 7},
        0,
        "SPIN",
    )
    ising = bqm_example.to_ising()
    return (ising,)


@app.cell(hide_code=True)
def _(ising, mo):
    mo.md(
        f"""```python
    {ising}
    ```

    In the above output:

    - The first term of the tuple corresponds to the linear terms ($h$):
    	```python
    	{ising[0]}
    	```

    - The second term of the tuple corresponds to the quadratic terms ($J$):
    	```python
    	{ising[1]}
    	```

    - The third term corresponds to the offset: `{ising[2]}`
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 6**

    Obtain the Ising model for the <span style="font-variant: small-caps;">bqm</span> formulation you created in Task 2.

    **Note:** <span style="font-variant: small-caps;">bqm</span> formulation you created in Task 2 has variable type `BINARY` and it will be converted into `SPIN` after you make the conversion.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## <span style="font-variant: small-caps;">qubo</span> ↔ Ising

    As you have seen in Task 6, it is possible to convert between Ising model and <span style="font-variant: small-caps;">qubo</span> formulation through the <span style="font-variant: small-caps;">bqm</span> class.

    This is the way to follow if you want to make a conversion between the two models: first obtain a <span style="font-variant: small-caps;">bqm</span> instance, then use the conversion functions.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 7**

    Convert the following <span style="font-variant: small-caps;">qubo</span> formulation into Ising formulation using the <span style="font-variant: small-caps;">bqm</span> class.

    $$5x_1 + 7x_1 x_2 - 3x_2$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 8**

    Convert the following Ising model into <span style="font-variant: small-caps;">qubo</span> formulation using the <span style="font-variant: small-caps;">bqm</span> class.

    $$s_1s_2 - s_1 + 3s_2$$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    from dimod import BQM
    import marimo as mo
    import numpy as np
    return BQM, mo, np


if __name__ == "__main__":
    app.run()
