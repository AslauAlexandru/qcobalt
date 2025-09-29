import marimo

__generated_with = "0.16.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # <span style="font-variant: small-caps;">bqm</span> Max-Cut

    So far we have learnt how to use the Ocean <span style="font-variant: small-caps;">sdk</span> tools. Now let's use those tools to formulate Binary Quadratic Models for some combinatorial optimisation problems. In this notebook, we will learn how to formulate <span style="font-variant: small-caps;">bqm</span> for the max-cut.

    To briefly recall, the goal of the maximum cut problem is to partition a set of vertices of a graph into two disjoint sets such that the number of edges that are cut by the partition is maximised.

    The <span style="font-variant: small-caps;">qubo</span> objective function for a graph with edge set $E$ is:

    $$\min \sum_{(i, j) \in E} -x_i - x_j + 2x_ix_j$$

    The Ising objective function for a graph with edge set $E$ is:

    $$\min \sum_{(i, j) \in E} s_i s_j$$

    The `dwave-networkx` package in the Ocean <span style="font-variant: small-caps;">sdk</span> has implementations of graph-theory algorithms for some combinatorial optimisation problems like max-cut, graph colouring, travelling salesman, etc.

    We can either use these already implemented algorithms or solve by formulating the problem as a <span style="font-variant: small-caps;">qubo</span> or Ising Model from scratch. We will look at both approaches.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Built-in Function

    `maximum_cut` is the built-in function in the `dwave-networkx` package for solving the maximum cut problem.

    ### Parameters

    - `G` - NetworkX graph
    - `sampler` - <span style="font-variant: small-caps;">bqm</span> sampler for solving the NetworkX graph

    We are going to use the classical solver `ExactSolver()` for solving this problem.

    Let's try to solve the Max-Cut problem for the following graph.
    """
    )
    return


@app.cell
def _():
    from foundation import Graph
    from networkx import house_x_graph

    G = Graph(house_x_graph())
    G.display()
    return G, Graph


@app.cell
def _(G):
    from dimod.reference.samplers import ExactSolver
    from dwave_networkx import maximum_cut

    sampler = ExactSolver()

    cut = maximum_cut(G, sampler)
    print(cut)
    return ExactSolver, cut, sampler


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The solution says that the vertices 2 and 3 should belong to the same group.

    Knowing this, we can deduce that the number of edges in the cut is 6.
    """
    )
    return


@app.cell
def _(G, cut):
    G.display_cut(cut)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 1**

    Find the maximum cut for the following graph using the built-in function and visualise the result.
    """
    )
    return


@app.cell
def _(Graph):
    from networkx import turan_graph

    G1 = Graph(turan_graph(5, 3))
    G1.display()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Formulating <span style="font-variant: small-caps;">bqm</span> from Scratch using <span style="font-variant: small-caps;">qubo</span> Formulation

    Now let's learn how to solve the problem by formulating the <span style="font-variant: small-caps;">bqm</span> from scratch using <span style="font-variant: small-caps;">qubo</span> formulation.

    ### Step 1 - Define <span style="font-variant: small-caps;">qubo</span>

    Define the <span style="font-variant: small-caps;">qubo</span> in dictionary form using the objective function:

    $$\min \sum_{(i, j) \in E} -x_i - x_j + 2x_ix_j$$
    """
    )
    return


@app.cell
def _(G):
    from collections import defaultdict

    # defaultdict(int) initialises all dictionary values to 0
    _d = defaultdict(int)

    # We consider each edge one by one and update the coefficients accordingly
    for i, j in G.edges:
        _d[(i, i)] += -1
        _d[(j, j)] += -1
        _d[(i, j)] += 2

    qubo_dict = dict(_d)
    print(qubo_dict)
    return (qubo_dict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Step 2 - Create <span style="font-variant: small-caps;">bqm</span> instance out of <span style="font-variant: small-caps;">qubo</span>

    Create an instance of <span style="font-variant: small-caps;">bqm</span> using the <span style="font-variant: small-caps;">qubo</span> dictionary.
    """
    )
    return


@app.cell
def _(qubo_dict):
    from dimod import BQM

    bqm = BQM.from_qubo(qubo_dict)
    return BQM, bqm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""If necessary, we can inspect the <span style="font-variant: small-caps;">qubo</span> matrix."""
    )
    return


@app.cell
def _(bqm, np):
    # Get the numpy vectors for the BQM
    _linear, (_row, _col, _quadratic), _offset = bqm.to_numpy_vectors(
        sort_indices=True
    )

    # Reconstruct the Q matrix
    _n = bqm.num_variables
    _Q = np.zeros((_n, _n))
    np.fill_diagonal(_Q, _linear)
    _Q[_row, _col] = _quadratic

    print(_Q)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Step 3 - Solve the <span style="font-variant: small-caps;">bqm</span>

    Solve it using `ExactSolver()`.
    """
    )
    return


@app.cell
def _(bqm, sampler):
    sampleset = sampler.sample(bqm)
    print(sampleset)
    return (sampleset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Step 4 - Get the best sample

    We simply take the sample with the lowest energy.
    """
    )
    return


@app.cell
def _(sampleset):
    best_sample = sampleset.first.sample
    print(best_sample)
    return (best_sample,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Checking the solution, we see that vertices 2 and 3 should belong to the same group. In this case, we had observed that there would be 6 edges in the cut, which is the energy returned by the solver.

    Also note that the second solution also has energy $-6$, as it is symmetric to the first solution.

    ### Step 5 - Visualise the result
    """
    )
    return


@app.cell
def _(G, best_sample):
    G.display_cut(best_sample)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 2**

    Obtain the maximum cut for the following graph:

    1. Define the <span style="font-variant: small-caps;">qubo</span> dictionary
    2. Create an instance of <span style="font-variant: small-caps;">bqm</span> using the dictionary and solve it using the classical solver
    3. Visualise the output obtained from the sampleset
    """
    )
    return


@app.cell
def _(Graph):
    from networkx import truncated_tetrahedron_graph

    G2 = Graph(truncated_tetrahedron_graph())
    G2.display()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Formulating <span style="font-variant: small-caps;">bqm</span> from Scratch using Ising Model

    For formulating an Ising model for the problem the code used to populate the dictionary should be altered according to the Ising objective function. The remaining steps would be the same.

    ### Step 1 - Define Ising Model

    Define the Ising Model in dictionary form using the objective function:

    $$\min \sum_{(i, j) \in E} s_i s_j$$
    """
    )
    return


@app.cell
def _(G):
    h = {}
    J = {}

    for _i, _j in G.edges:
        J[(_i, _j)] = 1

    print("h:", h)
    print("J:", J)
    return J, h


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Step 2 - Create <span style="font-variant: small-caps;">bqm</span> instance out of Ising Model

    Create an instance of <span style="font-variant: small-caps;">bqm</span> from the dictionaries `h` and `J`.
    """
    )
    return


@app.cell
def _(BQM, J, h):
    bqm_ising = BQM.from_ising(h, J)
    return (bqm_ising,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Step 3 - Solve the <span style="font-variant: small-caps;">bqm</span>

    We will solve it using `ExactSolver()`.
    """
    )
    return


@app.cell
def _(ExactSolver, bqm_ising):
    _sampler_ising = ExactSolver()
    sampleset_ising = _sampler_ising.sample(bqm_ising)

    print(sampleset_ising)
    return (sampleset_ising,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    > **Note:** The energy in this case is not -6 but -4.
    >
    > The reason is that instead of using the equivalent Ising model which is $\min  \frac{1}{2} \sum_{(i, j) \in E} (s_is_j-1)$, we simply used $\min \sum_{(i,j) \in E} s_is_j$. If you would like to get the same energy, you should use the first formulation. In practice, once you get the solution, you can determine the number of edges in the cut.

    ### Step 4 - Get the best sample

    We get the first sample, sample with the lowest energy.
    """
    )
    return


@app.cell
def _(sampleset_ising):
    optimal_sample = sampleset_ising.first.sample
    return (optimal_sample,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Step 5 - Visualise the result""")
    return


@app.cell
def _(G, optimal_sample):
    G.display_cut(optimal_sample)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 3**

    Obtain the maximum cut for the following graph:

    1. Define the dictionaries `h` and `J`
    2. Create an instance of <span style="font-variant: small-caps;">bqm</span> from the dictionaries and solve it using the classical solver
    3. Visualise the output obtained from the sampleset
    """
    )
    return


@app.cell
def _(Graph):
    from networkx import complete_graph

    G3 = Graph(complete_graph(5))
    G3.display()
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    return mo, np


if __name__ == "__main__":
    app.run()
