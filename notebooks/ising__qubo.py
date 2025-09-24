import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Ising to QUBO Conversion 

    The Ising problems are expressed through spin variables $\{-1, 1\}$, whilst <span style="font-variant: small-caps;">qubo</span> formulations are expressed through binary variables $\{0, 1\}$. 

    The two formulations can be converted into each other by the following transformation:

    $$ x_j \mapsto \frac{1 + s_j }{2} $$

    where $x_j$ is the <span style="font-variant: small-caps;">qubo</span> variable and $s_j$ is the Ising variable. 

    Substituting the value of the variable will result in conversion from one model to the other.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 1**

    Convert the following <span style="font-variant: small-caps;">qubo</span> formulation into Ising formulation:

    $$f (x_1, x_2) = 5x_1 + 7x_1 x_2 - 3x_2$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 2**

    Convert the following Ising model into QUBO formulation:

    $$ s_1s_2 - s_1 + 3s_2 $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 3**

    Using the <span style="font-variant: small-caps;">qubo</span> formulation you obtained in Task 2, calculate the energy for different spin and binary variable assignments and compare the results.

    Write two functions:

    - A function that takes as input values for $x_1$ and $x_2$ and returns the value of the <span style="font-variant: small-caps;">qubo</span> 
    - A function that takes as input values for $s_1$ and $s_2$ and returns the energy of the Ising model.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Max-Cut Review

    Recall that we have given both a <span style="font-variant: small-caps;">qubo</span> formulation and an Ising model for the Max-Cut problem. 

    The <span style="font-variant: small-caps;">qubo</span> formulation was defined as:

    $$\min \sum_{(i,j) \in E} -x_i-x_j+2x_ix_j$$

    while the Ising model was defined as:

    $$ \min  \frac{1}{2} \sum_{(i, j) \in E} (s_is_j-1) $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 4**

    Convert the <span style="font-variant: small-caps;">qubo</span> formulation for the Max-Cut problem into Ising formulation through variable change.

    Compare your result with the Ising Model we have defined.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
