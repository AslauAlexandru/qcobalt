import marimo

__generated_with = "0.16.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # <span style="font-variant: small-caps;">bqm</span> Graph Colouring

    In this notebook, we will learn how to formulate Binary Quadratic Models for the graph colouring problem using the Ocean <span style="font-variant: small-caps;">sdk</span> tools.

    To briefly recall, given a graph, the goal of graph colouring (or vertex colouring) is to decide whether the graph can be coloured so that the adjacent vertices have different colours from a set of $K$ colours.

    The <span style="font-variant: small-caps;">qubo</span> formulation is given as:

    $$\sum_{i=0}^{N-1} \left(1-\sum_{c=0}^{K-1}x_{i,c}\right)^2 +  \sum_{(i,j) \in E} \sum_{c=0}^{K-1} x_{i,c}x_{j,c}$$

    The Ocean <span style="font-variant: small-caps;">sdk</span> provides built-in implementations for common graph problems like graph colouring through the `dwave-networkx` package. We can either use these ready-made algorithms or solve by formulating the problem as a <span style="font-variant: small-caps;">bqm</span> from scratch. We will explore both approaches.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Built-in Function

    `vertex_color` is the built-in function in the `dwave-networkx` package for solving the graph colouring problem.

    ### Parameters

    - `G` - The NetworkX graph
    - `colors` - List of colours
    - `sampler` - <span style="font-variant: small-caps;">bqm</span> sampler for solving the NetworkX graph

    Let's examine the example graph from our earlier <span style="font-variant: small-caps;">qubo</span> formulation studies.
    """
    )
    return


@app.cell
def _():
    from foundation import Graph

    _edges = [(0, 1), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (3, 4)]
    graph = Graph(_edges)
    graph.display()
    return (graph,)


@app.cell
def _(graph):
    from dwave.samplers import SimulatedAnnealingSampler
    from dwave_networkx import vertex_color

    _colours = [0, 1, 2]
    _sampler = SimulatedAnnealingSampler()
    _colouring = vertex_color(graph, _colours, _sampler)
    print("Colouring result:", _colouring)
    return SimulatedAnnealingSampler, vertex_color


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We can verify whether the found colouring is feasible (i.e., all adjacent nodes have different colours) using the built-in validation function:"""
    )
    return


@app.cell
def _(SimulatedAnnealingSampler, graph, vertex_color):
    from dwave_networkx import is_vertex_coloring

    _colours = [0, 1, 2]
    _sampler = SimulatedAnnealingSampler()
    _colouring = vertex_color(graph, _colours, _sampler)
    _is_valid = is_vertex_coloring(graph, _colouring)
    print("Is colouring valid?", _is_valid)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's visualise the result:""")
    return


@app.cell
def _(SimulatedAnnealingSampler, graph, vertex_color):
    _colours = [0, 1, 2]
    _sampler = SimulatedAnnealingSampler()
    _colouring = vertex_color(graph, _colours, _sampler)
    graph.display_colouring(_colouring)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Formulating <span style="font-variant: small-caps;">bqm</span> using Ocean <span style="font-variant: small-caps;">sdk</span> Functions

    Now let's build the <span style="font-variant: small-caps;">bqm</span> manually using Ocean <span style="font-variant: small-caps;">sdk</span> functionality. Instead of using the penalty method, we'll add constraints directly to the <span style="font-variant: small-caps;">bqm</span>.

    ### Step 1 - Define an Empty <span style="font-variant: small-caps;">bqm</span>
    """
    )
    return


@app.cell
def _():
    from dimod import BQM

    bqm = BQM("BINARY")
    return (bqm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Step 2 - Add the Constraints to the <span style="font-variant: small-caps;">bqm</span>

    Using the functionality of the `BQM` class, we will add the constraints directly instead of using the penalty method.

    #### Constraint 1

    Each node should be coloured exactly once:

    $$\sum_{c=0}^{K-1} x_{i,c} = 1 \quad \forall  i=0,\dots,N-1$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 1**

    Add the first constraint to the <span style="font-variant: small-caps;">bqm</span> for our example graph using 3 colours.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Constraint 2

    Adjacent nodes should have different colours:

    $$x_{i,c} + x_{j,c} \leq 1 \quad \forall  c=1,\dots,K \text{ and } (i,j) \in E$$

    We use the function `add_linear_inequality_constraint` to add linear inequality constraints of the form:

    $$lb \leq c_1x_1+c_2x_2+\dots+c_nx_n+c \leq ub$$

    The coefficients for the binary variables should be provided as a list:

    $$[(x_1,c_1), (x_2,c_2), \dots, (x_n,c_n)]$$

    followed by the constant term $c$ and the `lagrange_multiplier` parameter which corresponds to the penalty coefficient.

    `lb` and `ub` are the lower and upper bounds which are by default set to 0. The `label` parameter is required to identify the inequality constraint.
    """
    )
    return


@app.cell
def _(bqm, graph):
    _colours = [0, 1, 2]

    for _c in _colours:
        for _i, _j in graph.edges:
            _constraint_vars = [(f"x_{_i}_{_c}", 1), (f"x_{_j}_{_c}", 1)]
            bqm.add_linear_inequality_constraint(
                _constraint_vars,
                lagrange_multiplier=1.0,
                label=f"edge_{_i}_{_j}_colour_{_c}",
            )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Note that in the sampleset, there will be additional **slack** variables named using the label of the inequality constraint. These are created when converting inequality to equality constraints.

    ### Step 3 - Solve the <span style="font-variant: small-caps;">bqm</span>
    """
    )
    return


@app.cell
def _(SimulatedAnnealingSampler, bqm):
    _sampler = SimulatedAnnealingSampler()
    sampleset = _sampler.sample(bqm, num_reads=100)
    print(sampleset.truncate(10))
    return (sampleset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Step 4 - Interpret and Check the Feasibility of the Samples

    As a result of simulated annealing, we obtain a sample where some variables are set to 1 and some to 0. We need to check if any sample corresponds to a feasible solution.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 2**

    Write a Python function named `is_sample_feasible` that takes a sample containing binary variables named `x_i_c` and their values, the list of colours, the list of edges, and the number of nodes, and returns `True` if the sample corresponds to a feasible colouring and `False` otherwise.
    """
    )
    return


@app.function
def is_sample_feasible(_sample, _colours, _edges, _N):
    pass


@app.cell
def _(graph, sampleset):
    _colours = [0, 1, 2]
    _N = len(graph.nodes)
    _first_sample = sampleset.first.sample
    _feasible = is_sample_feasible(_first_sample, _colours, graph.edges, _N)
    print("Is first sample feasible?", _feasible)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""If the first sample is not feasible, we can search for another solution among the sampleset:"""
    )
    return


@app.cell
def _(graph, sampleset):
    _colours = [0, 1, 2]
    _N = len(graph.nodes)


    def _best_solution(_sampleset, _colours, _edges, _N):
        for _sample, _energy in _sampleset.data(fields=["sample", "energy"]):
            if is_sample_feasible(_sample, _colours, _edges, _N):
                return _sample, _energy
        return None, None


    _sample, _energy = _best_solution(sampleset, _colours, graph.edges, _N)
    if _sample:
        print("Found feasible solution with energy:", _energy)
    else:
        print("No feasible solution found")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 3**

    Write a function named `sample_to_colouring` that takes a sample containing binary variables named `x_i_c` and their values, the list of colours and the number of vertices, and returns a dictionary where the keys are the nodes and the values are the colours.
    """
    )
    return


@app.function
def sample_to_colouring(_sample, _colours, _N):
    pass


@app.cell
def _(graph, sampleset):
    _colours = [0, 1, 2]
    _N = len(graph.nodes)
    _sample = sampleset.first.sample
    _colouring = sample_to_colouring(_sample, _colours, _N)
    print("Colouring from sample:", _colouring)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Step 5 - Visualise the Output""")
    return


@app.cell
def _(graph, sampleset):
    _colours = [0, 1, 2]
    _N = len(graph.nodes)
    _sample = sampleset.first.sample
    _colouring = sample_to_colouring(_sample, _colours, _N)
    # graph.display_colouring(_colouring)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 4**

    Create a function named `graph_colouring_bqm` that takes as input the list of colours, the list of edges, number of vertices and returns the binary quadratic model for the graph colouring problem.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 5**

    Using the function you have created in Task 4, try colouring the same graph using two colours only and interpret the result.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 6**

    For the Petersen graph displayed below, determine the minimum number of colours needed to properly colour the graph.

    > Starting with 2 colours, increase the number of colours you use inside a loop until a feasible colouring is found.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
