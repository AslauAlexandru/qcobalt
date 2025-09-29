import marimo

__generated_with = "0.16.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # <span style="font-variant: small-caps;">bqm</span> Travelling Salesman Problem

    In this notebook, we will learn how to formulate <span style="font-variant: small-caps;">bqm</span> for the Travelling Salesman Problem. Instead of following the approach in the previous notebook, we will use the functions provided by Ocean <span style="font-variant: small-caps;">sdk</span> to formulate the binary quadratic model.

    To briefly recall, given a set of cities and corresponding distances between each pair of cities, the goal is to find the shortest possible route such that a salesman visits every city exactly once and returns to the starting point.

    <span style="font-variant: small-caps;">qubo</span> formulation for <span style="font-variant: small-caps;">tsp</span> was given as follows:

    $$P \cdot \sum_{t=0}^{N-1} \left(1-\sum_{i=0}^{N-1}x_{i,t}\right)^2 + P \cdot \sum_{i=0}^{N-1} \left(1-\sum_{t=0}^{N-1}x_{i,t}\right)^2 + \sum_{ \substack{i,j=0\\i\neq j}}^{N-1} w_{ij} \sum_{t=0}^{N-1} x_{i,t} x_{j,t+1} $$

    We will solve <span style="font-variant: small-caps;">tsp</span> problem both by using the built-in <span style="font-variant: small-caps;">bqm</span> constructor, by creating the <span style="font-variant: small-caps;">bqm</span> from scratch and by using the functionality of Ocean <span style="font-variant: small-caps;">sdk</span> to incorporate constraints into the model.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Built-in Function

    `travelling_salesperson` is the built-in function in the `dwave-networkx` package for solving the travelling salesman problem.

    ### Parameters

    - `G` - The NetworkX graph
    - `sampler` - <span style="font-variant: small-caps;">bqm</span> sampler for solving the NetworkX graph
    - `start` (optional) - Starting point of the tour

    ### Example 1

    Let us consider the following graph.
    """
    )
    return


@app.cell
def _():
    from foundation import Graph

    G = Graph()
    G.add_weighted_edges_from(
        {
            (0, 1, 0.1),
            (0, 2, 0.5),
            (0, 3, 0.1),
            (1, 2, 0.1),
            (1, 3, 0.5),
            (2, 3, 0.1),
        }
    )
    G.display()
    return G, Graph


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We are going to use the classical solver `ExactSolver` for solving this problem.""")
    return


@app.cell
def _(G):
    from dimod.reference.samplers import ExactSolver
    from dwave_networkx import traveling_salesperson

    sampler = ExactSolver()
    path = traveling_salesperson(G, sampler, start=0)
    print(path)
    return path, traveling_salesperson


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let us visualise the result.""")
    return


@app.cell
def _(G, path):
    G.display_path(path)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Example 2

    This time, let us create a complete graph with 6 vertices and assign random weights to each edge.
    """
    )
    return


@app.cell
def _(Graph):
    from numpy import random
    from networkx import complete_graph

    random.seed(45)

    G1 = Graph(complete_graph(6))
    for _u, _v in G1.edges():
        G1[_u][_v]["weight"] = random.randint(1, 5)

    G1.display()
    return G1, complete_graph, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""If you try using the `ExactSolver` with this instance, you may not be successful depending on your computer's memory. Hence we will use `SimulatedAnnealingSampler` instead.""")
    return


@app.cell
def _(G1, traveling_salesperson):
    from dwave.samplers import SimulatedAnnealingSampler

    _sampler = SimulatedAnnealingSampler()

    path1 = traveling_salesperson(G1, _sampler, start=0)
    print(path1)
    return SimulatedAnnealingSampler, path1


@app.cell
def _(G1, path1):
    G1.display_path(path1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 1**

    Find the optimal route for the given graph using simulated annealer and the built-in function.
    """
    )
    return


@app.cell
def _(Graph):
    G_task1 = Graph()
    G_task1.add_weighted_edges_from(
        {(0, 1, 1), (0, 2, 5), (0, 3, 2), (1, 2, 4), (1, 3, 5), (2, 3, 3)}
    )
    G_task1.display()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Formulating <span style="font-variant: small-caps;">bqm</span> using Ocean <span style="font-variant: small-caps;">sdk</span> Functions

    ### Step 1 - Define an empty <span style="font-variant: small-caps;">bqm</span> and add the cost function you want to minimise

    In case of <span style="font-variant: small-caps;">tsp</span>, this is the third term corresponding to the cost of the tour:

    $$\sum_{ \substack{i,j=0\\i\neq j}}^{N-1} w_{ij} \sum_{t=0}^{N-1} x_{i,t} x_{j,t+1}$$

    We use the function `add_quadratic` and provide the terms and the coefficient. Syntax is `(x_i, x_j, Q_{ij})`.

    We will use the following graph.
    """
    )
    return


@app.cell
def _(G):
    G.display()
    return


@app.cell
def _(G):
    from dimod import BQM

    bqm = BQM("BINARY")

    _N = len(G.nodes)
    for _i in range(_N):
        for _j in range(_N):
            if _i != _j:
                for _t in range(_N - 1):
                    bqm.add_quadratic(
                        f"x_{_i}_{_t}", f"x_{_j}_{_t+1}", G[_i][_j]["weight"]
                    )

                # Remember that we were assuming N=0 in the sum
                bqm.add_quadratic(
                    f"x_{_i}_{_N-1}", f"x_{_j}_{0}", G[_i][_j]["weight"]
                )

    N = _N
    return BQM, N, bqm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    > **Note:** In case your graph is not defined through networkx package but through a cost matrix `W`, you should replace `G[i][j]["weight"]` with `W[i][j]`.

    ### Step 2 - Add the Constraints to the <span style="font-variant: small-caps;">bqm</span>

    Instead of the penalty method, <span style="font-variant: small-caps;">bqm</span> class allows us the functionality to add constraints directly.

    #### Constraint 1

    Only one city should be visited at a time.

    $$\sum_{i=0}^{N-1}x_{i,t}=1 \text{ for all }t$$

    We use the function `add_linear_equality_constraint` through which you can add linear equality constraints of the form:

    $$c_1x_1+c_2x_2+\dots+c_nx_n+c=0$$

    The coefficients for the binary variables should be provided as a list:

    $$[(x_1,c_1), (x_2,c_2), \dots, (x_n,c_n)]$$

    followed by the constant term $c$ and the `lagrange_multiplier` parameter.

    Lagrange multiplier is exactly the penalty coefficient we have seen so far.

    Penalty method is implemented by Ocean automatically.

    Here is an example:
    """
    )
    return


@app.cell
def _(N, bqm):
    l1 = 5
    for _t in range(N):
        _c1 = [(f"x_{_i}_{_t}", 1) for _i in range(N)]  # coefficient list
        bqm.add_linear_equality_constraint(
            _c1, constant=-1, lagrange_multiplier=l1
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #### Constraint 2

    Each city should be visited one and only once.

    $$\sum_{t=0}^{N-1}x_{i,t}=1 \qquad \forall i$$

    ## **Task 2**

    Add the second constraint to the <span style="font-variant: small-caps;">bqm</span>. Let Lagrange multiplier = 5.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Step 3 - Solve the <span style="font-variant: small-caps;">bqm</span>

    We are going to use the `SimulatedAnnealingSampler` to solve the <span style="font-variant: small-caps;">bqm</span>.
    """
    )
    return


@app.cell
def _(SimulatedAnnealingSampler, bqm):
    _sampler_sa = SimulatedAnnealingSampler()
    sampleset = _sampler_sa.sample(bqm, num_reads=1000)
    print(sampleset.truncate(10))
    return (sampleset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Step 4 - Interpret and check the feasibility of the samples in the sampleset and find the optimum sample

    As a result of simulated annealing, we obtain a sample where some of the variables are set to 1 and some of the variables are set to 0.

    Given a sample, we may want to check if it corresponds to a feasible solution or not, i.e. whether each city is visited exactly once and at each time point exactly one city is visited.

    ## **Task 3**

    Write a Python function named `is_sample_feasible` that takes as parameter a sample containing binary variables named `x_i_p` and their values and the number of cities, and returns `True` if the sample corresponds to a feasible path and `False` otherwise.
    """
    )
    return


@app.cell
def _(N, sampleset):
    def is_sample_feasible(_sample, _N):
        pass


    first_sample = sampleset.first.sample
    print(is_sample_feasible(first_sample, N))
    return (is_sample_feasible,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In case the first sample is not feasible, we may search for another solution among the sampleset which is feasible. This can be accomplished by the following code:""")
    return


@app.cell
def _(N, is_sample_feasible, sampleset):
    def best_solution(_sampleset, _N):
        for _sample, _energy in _sampleset.data(fields=["sample", "energy"]):
            if is_sample_feasible(_sample, _N):
                return _sample, _energy
        return None, None


    best_solution(sampleset, N)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Suppose we verified that the sample is feasible. Then we would like to obtain the path it corresponds to.

    In the next Task, your goal is to convert a given sample into a path in the form of a list containing city numbers.

    ## **Task 4**

    Write a Python function named `sample_to_path` that takes as parameter a sample containing binary variables named `x_i_p` and their values and the number of cities, and returns a list of cities corresponding to the sample.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    > **Note:** In case some constraint is violated, the energy value does not exactly give the cost, but cost + the penalty incurred.

    ### Step 5 - Visualise the Output

    Since we have obtained the path as a list of cities, we can use `display_path` function to visualise the result.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 5**

    Write a function named `tsp_bqm` that takes as parameter a networkx graph $G$ and the penalty coefficient and returns the binary quadratic model for the travelling salesman problem.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 6**

    For the graph `G1` defined above, construct the <span style="font-variant: small-caps;">bqm</span> and find the optimal path.

    Don't forget to set the penalty coefficient to a suitable value.

    Let's define the graph again.
    """
    )
    return


@app.cell
def _(Graph, complete_graph, random):
    random.seed(45)
    G1_task6 = Graph(complete_graph(6))
    for _u, _v in G1_task6.edges():
        G1_task6[_u][_v]["weight"] = random.randint(1, 5)
    N_task6 = len(G1_task6.nodes)

    G1_task6.display()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Formulating <span style="font-variant: small-caps;">bqm</span> from Scratch (Optional)

    Although we have built-in functions to define binary quadratic model through inequality and equality constraints, one can also define a <span style="font-variant: small-caps;">bqm</span> from scratch. Now we will see how we can accomplish this.
    """
    )
    return


@app.cell
def _(G):
    G.display()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Step 1 - Define <span style="font-variant: small-caps;">qubo</span>

    Define the <span style="font-variant: small-caps;">qubo</span> in dictionary, recalling the following:

    - Each $x_{i,t}$ appears with coefficient $-2P$.
    - For each fixed $t$, all possible 2-combinations of $x_{i,t}$ appears with coefficient $2P$.
    - For each fixed $i$, all possible 2-combinations of $x_{i,t}$ appears with coefficient $2P$.
    - Each $x_{i,t} x_{j,t+1}$ appears with the coefficient $w_{ij}$.
    - There is a constant coefficient of $2NP$.
    """
    )
    return


@app.cell
def _(G):
    _d = {}
    _P = 5

    _N = len(G.nodes)

    for _i in range(_N):
        for _t in range(_N):
            _d[(f"x_{_i}_{_t}", f"x_{_i}_{_t}")] = -2 * _P

    for _t in range(_N):
        for _i in range(_N):
            for _j in range(_i + 1, _N):
                _d[(f"x_{_i}_{_t}", f"x_{_j}_{_t}")] = 2 * _P

    for _t in range(_N):
        for _tp in range(_t + 1, _N):
            for _i in range(_N):
                _d[(f"x_{_i}_{_t}", f"x_{_i}_{_tp}")] = 2 * _P

    for _i in range(_N):
        for _j in range(_N):
            if _i != _j:
                _d[(f"x_{_i}_{_N-1}", f"x_{_j}_{0}")] = G[_i][_j]["weight"]
                for _t in range(_N - 1):
                    _d[(f"x_{_i}_{_t}", f"x_{_j}_{_t+1}")] = G[_i][_j]["weight"]

    qubo_dict_scratch = _d
    P_scratch = _P
    N_scratch = _N
    return N_scratch, P_scratch, qubo_dict_scratch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Step 2 - Create <span style="font-variant: small-caps;">bqm</span> instance out of <span style="font-variant: small-caps;">qubo</span>

    Create an instance of <span style="font-variant: small-caps;">bqm</span> from the <span style="font-variant: small-caps;">qubo</span> dictionary.

    Don't forget to add the offset.
    """
    )
    return


@app.cell
def _(BQM, N_scratch, P_scratch, qubo_dict_scratch):
    bqm_scratch = BQM.from_qubo(
        qubo_dict_scratch, offset=2 * N_scratch * P_scratch
    )
    return (bqm_scratch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Step 3 - Solve the <span style="font-variant: small-caps;">bqm</span>

    Solve it using `SimulatedAnnealingSampler()`.
    """
    )
    return


@app.cell
def _(SimulatedAnnealingSampler, bqm_scratch):
    _sampler_scratch = SimulatedAnnealingSampler()
    sampleset_scratch = _sampler_scratch.sample(bqm_scratch, num_reads=1000)
    print(sampleset_scratch.truncate(10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""You can also check the corresponding $Q$ matrix.""")
    return


@app.cell
def _(bqm_scratch):
    from numpy import zeros, fill_diagonal

    # Get the numpy vectors for the BQM
    _linear, (_row, _col, _quadratic), _offset = bqm_scratch.to_numpy_vectors(
        sort_indices=True
    )

    # Reconstruct the Q matrix
    _n = bqm_scratch.num_variables
    _Q = zeros((_n, _n))
    fill_diagonal(_Q, _linear)
    _Q[_row, _col] = _quadratic

    print(_Q)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
