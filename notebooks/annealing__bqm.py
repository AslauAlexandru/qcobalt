import marimo

__generated_with = "0.16.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # `BinaryQuadraticModel`

    So far, we have learned about formulating combinatorial optimisation problems as <span style="font-variant: small-caps;">qubo</span> or Ising Model problems. We have also learned how to convert between these two formulations. 

    Our final goal is to take advantage of quantum annealing to solve these problems. To do so, we have to formulate our <span style="font-variant: small-caps;">qubo</span> or Ising Model problems in a way that they can be run on quantum annealing devices, currently provided by D-Wave.

    The [Ocean <span style="font-variant: small-caps;">sdk</span>](https://docs.dwavequantum.com/en/latest/ocean/index.html) provides us many open-source tools to aid us in the problem solving process. Now let's take a look at the `BinaryQuadraticModel` class available in the `dimod` package of the Ocean <span style="font-variant: small-caps;">sdk</span>.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The `BinaryQuadraticModel` class helps us to formulate our <span style="font-variant: small-caps;">qubo</span> or Ising Model problems into a form suitable to be run on D-Wave. Let us quickly recall the objective functions of <span style="font-variant: small-caps;">qubo</span> and Ising Model.

    The objective function of a <span style="font-variant: small-caps;">qubo</span> is given by:

    $$\sum\limits_{i} {Q_{i, i} x_i} + \sum\limits_{i < j} {Q_{i, j} x_i x_j} \qquad\qquad x_i\in \{0,1\}$$

    where the variables can take the values $0$ and $1$.

    The objective function of an Ising Model is given by:

    $$\sum\limits_{i} h_i s_i + \sum\limits_{i<j} J_{i,j} s_i s_j   \qquad\qquad s_i \in\{-1,+1\}$$

    where the variables can take the values $-1$ and $+1$ corresponding to the physical Ising spins.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The objective function of a Binary Quadratic Model (<span style="font-variant: small-caps;">bqm</span>) is given by:

    $$\sum\limits_{i=1} a_i v_i + \sum\limits_{i<j} b_{i,j} v_i v_j + c \qquad\qquad v_i \in \{0,1\} \text{  or } \{-1,+1\}$$

    Note that the variable $v_i$ can correspond either to $\{0, 1\}$ or to the physical Ising spins $\{-1, +1\}$. This way a <span style="font-variant: small-caps;">bqm</span> can conveniently represent both a <span style="font-variant: small-caps;">qubo</span> and an Ising Model.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Creating an Instance of a `BinaryQuadraticModel`

    Let us first take a look at some of the essential parameters required to create an instance of the `BinaryQuadraticModel` class.

    **Parameters:**

    - **`linear`**
        - The linear terms of the objective function should be defined as a dictionary.
        - The keys of the dictionary should be the variables and their respective values should be the coefficients associated with these variables. For example:
            ```python
            {'x1': 3, 'x2': 5, 'x3': 4, 'x4': 7}
            ```
    - **`quadratic`**
        - The quadratic terms of the objective function should be defined as a dictionary.
        - The keys of the dictionary should be the pairs of variables defined as tuples and their respective values should be the coefficients associated with these pairs of variables. For example:
        ```python
        {('x1', 'x2'): 2, ('x2', 'x3'): 5}
        ```
    - **`offset`**
        - Constant energy offset value associated with the <span style="font-variant: small-caps;">bqm</span> can be set using this parameter. 
        - If there is no offset, then there is no need to specify it.

    - **`vartype`**
        - This parameter sets the variable type of the <span style="font-variant: small-caps;">bqm</span>. To create a <span style="font-variant: small-caps;">qubo</span> instance, set this parameter to `'BINARY'`.
        - To create an Ising Model instance, set this parameter to `'SPIN'`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Example

    Let us now try to create a <span style="font-variant: small-caps;">bqm</span> instance for the following <span style="font-variant: small-caps;">qubo</span> formulation:

    $$- 5x_1 - 3x_2 - 8x_3 - 6x_4 + 4x_1 x_2 + 8x_1 x_3 + 2x_2 x_3 + 10x_3 x_4$$

    We should define the linear and quadratic parts of the objective function as dictionaries and pass them as `linear` and `quadratic` arguments. In the objective function:

    - The linear part is $- 5x_1 - 3x_2 - 8x_3 - 6x_4$. The corresponding dictionary can be defined as:

        ```python
        {'x1': -5, 'x2': -3, 'x3': -8, 'x4': -6}
        ```

    - The quadratic part is $4x_1 x_2 + 8x_1 x_3 + 2x_2 x_3 + 10x_3 x_4$. The corresponding dictionary can be defined as:

        ```python
        {('x1', 'x2'): 4, ('x1', 'x3'): 8, ('x2', 'x3'): 2, ('x3', 'x4'): 10}
        ```  

    - There is no offset, so we don't need to specify it.

    We can create a <span style="font-variant: small-caps;">qubo</span> instance of <span style="font-variant: small-caps;">bqm</span> by setting the `vartype` parameter to `'BINARY'`.
    """
    )
    return


@app.cell
def _():
    from dimod import BQM

    _linear = {"x1": -5, "x2": -3, "x3": -8, "x4": -6}
    _quadratic = {
        ("x1", "x2"): 4,
        ("x1", "x3"): 8,
        ("x2", "x3"): 2,
        ("x3", "x4"): 10,
    }
    _vartype = "BINARY"

    bqm_qubo = BQM(_linear, _quadratic, _vartype)
    print(bqm_qubo)
    return (BQM,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Similarly, we can create an Ising instance of <span style="font-variant: small-caps;">bqm</span> by setting the `vartype` parameter to `'SPIN'`.

    > **Note:** The `vartype` parameter just sets the variable type for a <span style="font-variant: small-caps;">bqm</span> and doesn't automatically convert between a <span style="font-variant: small-caps;">qubo</span> and an Ising Model. Recall that in order to convert a <span style="font-variant: small-caps;">qubo</span> to an Ising Model, the following transformation should be used:
    > 
    > $$ x_j \mapsto \frac{1+s_j}{2} $$
    >     
    > There are methods in the Ocean <span style="font-variant: small-caps;">sdk</span> for converting between the formulations. We will learn about them later on.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 1**

    Create a <span style="font-variant: small-caps;">qubo</span> instance of <span style="font-variant: small-caps;">bqm</span> for the following objective function:

    $$5x_1 + 7x_1 x_2 - 3x_2 + 2$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 2**

    Create an Ising instance of <span style="font-variant: small-caps;">bqm</span> for the following objective function:

    $$s_1 + s_2 + s_3 + s_4 - 6s_1 s_3 - 6s_1 s_4 - 6s_3 s_4 - 6s_1 s_2$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Finding the Lowest Energy Using a Classical Sampler

    Ocean <span style="font-variant: small-caps;">sdk</span> provides classical, quantum and hybrid samplers to help us find optimal solutions to our problems. A sampler tries to sample low energy states for a given <span style="font-variant: small-caps;">bqm</span> and returns an iterable of samples in the ascending order of the energy values.

    We are going to use `ExactSolver` to classically sample our problems. It works by finding the energy values of all the possible samples for a given <span style="font-variant: small-caps;">bqm</span>. As you can guess, this is not an efficient process but it is good enough for small problems. The general limit is 18 variables beyond which the process becomes very slow. `ExactSolver` can be helpful to test our code during development.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Example: <span style="font-variant: small-caps;">qubo</span>

    Let us try to create a <span style="font-variant: small-caps;">qubo</span> instance of <span style="font-variant: small-caps;">bqm</span> and find the energy values for the following objective function:

    $$f(x_1, x_2, x_3, x_4) = 3x_1 - 7x_2 + 11x_3 - x_4 + 9x_1 x_2 + x_1 x_3 + 2x_2 x_3 + 8x_3 x_4$$

    In the objective function:

    - The linear part is $3x_1 - 7x_2 + 11x_3 - x_4$
    - The quadratic part is $9x_1 x_2 + x_1 x_3 + 2x_2 x_3 + 8x_3 x_4$
    """
    )
    return


@app.cell
def _(BQM):
    from dimod.reference.samplers import ExactSolver

    _linear_ex = {"x1": 3, "x2": -7, "x3": 11, "x4": -1}
    _quadratic_ex = {
        ("x1", "x2"): 9,
        ("x1", "x3"): 1,
        ("x2", "x3"): 2,
        ("x3", "x4"): 8,
    }
    _vartype_ex = "BINARY"

    bqm_example = BQM(_linear_ex, _quadratic_ex, _vartype_ex)
    return ExactSolver, bqm_example


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Now that we have created a <span style="font-variant: small-caps;">qubo</span> instance, we can then assign `ExactSolver` to a variable for convenience. Then we should pass the instance `bqm_example` as an argument to the `sample` method of `ExactSolver` and assign it to another variable. This variable would then contain all the possible samples in the ascending order of their energy values."""
    )
    return


@app.cell
def _(ExactSolver, bqm_example):
    _sampler = ExactSolver()
    sampleset_example = _sampler.sample(bqm_example)
    print(sampleset_example)
    return (sampleset_example,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In the above output:

    - First column represents the serial number
    - The next four columns represent the different values for the four variables present in the objective function
    - `energy` column refers to the value of the objective function for each sample
    - `num_oc.` refers to the number of occurrences for each sample. Since the classical sampler exactly determines the energy value for each and every sample, number of occurrence for each sample is just 1.

    We can observe from the output that the first sample minimises the objective function. The energy values of the subsequent samples are in ascending order.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Accessing the single optimal solution

    The optimal solution that produces the lowest energy value can be accessed using the `first` attribute.
    """
    )
    return


@app.cell
def _(mo, sampleset_example):
    mo.md(
        f"""
    **Optimal solution:**
    ```
    {sampleset_example.first}
    ```

    **Just the sample values:**
    ```
    {sampleset_example.first.sample}
    ```

    **Just the energy value:**
    ```
    {sampleset_example.first.energy}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Accessing multiple optimal solutions

    In case our problem has multiple samples that produce the lowest energy value, we can access all of those optimal solutions at once using the `lowest` method. For example, let's consider the sampleset of the following simple <span style="font-variant: small-caps;">qubo</span> instance that has multiple optimal solutions.
    """
    )
    return


@app.cell
def _(BQM, ExactSolver, mo):
    _quadratic_simple = {("x1", "x2"): 1}
    _vartype_simple = "BINARY"

    bqm_simple = BQM(_quadratic_simple, _vartype_simple)

    _sampler_simple = ExactSolver()
    sampleset_simple = _sampler_simple.sample(bqm_simple)

    mo.md(
        f"""
    **Simple <span style="font-variant: small-caps;">qubo</span>:** $f(x_1, x_2) = x_1x_2$

    ```
    {sampleset_simple}
    ```
    """
    )
    return (sampleset_simple,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Here, the function to optimise is simple $x_1x_2$. 

    In the above sampleset we can observe that there are three optimal solutions that produce the lowest energy of $0$. Using the `first` attribute here would display only one of those three optimal solutions.
    """
    )
    return


@app.cell
def _(mo, sampleset_simple):
    mo.md(
        f"""
    **Using `.first` (shows only one optimal solution):**
    ```
    {sampleset_simple.first}
    ```

    **Using `.lowest()` (shows all optimal solutions):**
    ```
    {sampleset_simple.lowest()}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 3**

    Find the optimal sample of the <span style="font-variant: small-caps;">qubo</span> instance that produces the lowest energy value for the objective function used in Task 1:

    $$f(x_1, x_2) = 5x_1 + 7x_1 x_2 - 3x_2 + 2$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Example: Ising

    Let us now try to create an Ising instance and find the optimal solution for the following objective function:

    $$f(s_1, s_2, s_3, s_4) = 3s_1 - 7s_2 + 11s_3 - s_4 + 9s_1 s_2 + s_1 s_3 + 2s_2 s_3 + 8s_3 s_4$$
    """
    )
    return


@app.cell(hide_code=True)
def _(BQM, ExactSolver, mo):
    _linear_ising = {"s1": 3, "s2": -7, "s3": 11, "s4": -1}
    _quadratic_ising = {
        ("s1", "s2"): 9,
        ("s1", "s3"): 1,
        ("s2", "s3"): 2,
        ("s3", "s4"): 8,
    }
    _offset_ising = 0
    _vartype_ising = "SPIN"

    bqm_ising_example = BQM(
        _linear_ising, _quadratic_ising, _offset_ising, _vartype_ising
    )

    _sampler_ising = ExactSolver()
    sampleset_ising = _sampler_ising.sample(bqm_ising_example)

    mo.md(
        f"""
    **Ising example - All samples:**
    ```
    {sampleset_ising}
    ```

    **Optimal solution:**
    ```
    {sampleset_ising.first}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Therefore the sample `{'s1': -1, 's2': 1, 's3': -1, 's4': 1}` minimises the objective function to an energy value of `-40`."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 4**

    Find the optimal sample of the Ising instance that produces the lowest energy value for the objective function used in Task 2:

    $$f(s_1, s_2, s_3, s_4) = s_1 + s_2 + s_3 + s_4 - 6s_1 s_3 - 6s_1 s_4 - 6s_3 s_4 - 6s_1 s_2$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 5**

    Create a <span style="font-variant: small-caps;">qubo</span> instance of <span style="font-variant: small-caps;">bqm</span> for the given objective function and find the optimal solution:

    $$f(x_1, x_2, x_3, x_4) = 3x_1 - x_2 + 10x_3 + 7x_4 + 2x_1 x_2 - 5x_1 x_3 + 3x_2 x_3 + 11x_3 x_4$$
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
