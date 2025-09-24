"""Marimo notebook showing how to encode the max-cut problem as a QUBO."""

import marimo

__generated_with = "0.16.0"
app = marimo.App(app_title="QUBO Max-Cut – QCobalt", html_head_file="")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# <span style="font-variant-caps: small-caps">qubo</span> Max-Cut""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Quadratic unconstrained binary optimisation (<span style="font-variant-caps: small-caps">qubo</span>) problems can model a wide range of combinatorial optimisation tasks. In this lesson we focus on the max-cut problem and show how to encode it as a <span style="font-variant-caps: small-caps">qubo</span> objective.

    We start by recalling the problem definition, develop the binary encoding for individual edges, and then build the global objective and its matrix representation. Along the way you will experiment with small graphs and interpret the solutions returned by a simple exhaustive solver.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Review

    Given an undirected graph, $G = (V, E)$, the max-cut problem asks us to divide the vertices into two complementary groups so that as many edges as possible run between them. Any particular division of the vertices is a _cut_; its quality is measured by the number of crossing edges.

    To build a <span style="font-variant-caps: small-caps">qubo</span> we assign a binary variable to each vertex. For vertex $i$, we define
    $$x_i = \begin{cases}
    0 & \text{if vertex } i \text{ belongs to group } 1,\\
    1 & \text{if vertex } i \text{ belongs to group } 2.
    \end{cases}$$
    It is convenient to imagine a function, `edge_count(i, j)`, that returns $1$ when the endpoints of $(i, j)$ lie in different groups and $0$ otherwise. Our aim is to maximise the sum of `edge_count` across all edges.

    | $x_i$ | $x_j$ | `edge_count` | Comment |
    | :--: | :--: | :--: | :-- |
    | $0$ | $0$ | $0$ | Vertices share the same group |
    | $0$ | $1$ | $1$ | Vertices are separated |
    | $1$ | $0$ | $1$ | Vertices are separated |
    | $1$ | $1$ | $0$ | Vertices share the same group |

    The expression $x_i + x_j - 2 x_i x_j$ reproduces the values computed by `edge_count` and therefore captures the cut contribution from a single edge.

    > The truth table mirrors the xor operation from classical logic: the term evaluates to one exactly when the bits differ.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 1**

    Show algebraically that $x_i + x_j - 2 x_i x_j$ takes the values listed in the table above for all four binary assignments.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Constructing the Objective Function

    Summing the contribution over every edge leads to the max-cut objective
    $$\max_{\mathbf{x} \in \{0, 1\}^{|V|}} \sum_{(i, j) \in E} \bigl(x_i + x_j - 2 x_i x_j\bigr).$$
    Because <span style="font-variant-caps: small-caps">qubo</span> formulations minimise objectives, we multiply by $-1$ to obtain
    $$\min_{\mathbf{x} \in \{0, 1\}^{|V|}} \sum_{(i, j) \in E} \bigl(-x_i - x_j + 2 x_i x_j\bigr).$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Worked Example

    We will be working with the following graph of $5$ vertices and $6$ edges:
    """
    )
    return


@app.cell(hide_code=True)
def _(Graph, nodes, sparse_edges):
    sparse_graph = Graph()
    sparse_graph.add_nodes_from(nodes)
    sparse_graph.add_edges_from(sparse_edges)
    sparse_graph.display()
    return (sparse_graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Algebraic Form

    The objective for this graph is the sum of $-x_i - x_j + 2 x_i x_j$ over each edge $(i, j)$:

    $$\begin{aligned}
    \min\, &(-x_1 - x_2 + 2 x_1 x_2) + (-x_1 - x_3 + 2 x_1 x_3) \\
             &+ (-x_2 - x_4 + 2 x_2 x_4) + (-x_3 - x_4 + 2 x_3 x_4) \\
             &+ (-x_3 - x_5 + 2 x_3 x_5) + (-x_4 - x_5 + 2 x_4 x_5).
    \end{aligned}$$

    Collecting like terms we obtain
    $$\min(-2 x_1 - 2 x_2 - 3 x_3 - 3 x_4 - 2 x_5 + 2 x_1 x_2 + 2 x_1 x_3 + 2 x_2 x_4 + 2 x_3 x_4 + 2 x_3 x_5 + 2 x_4 x_5).$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Matrix Form

    Because the decision variables are binary we can replace each $x_i^2$ term with $x_i$, allowing the linear coefficients to sit on the diagonal of the <span style="font-variant-caps: small-caps">qubo</span> matrix. The objective therefore reads

    $$\min_{\mathbf{x} \in \{0, 1\}^5} \mathbf{x}^T Q \mathbf{x}$$

    where the assignment vector is

    $$\mathbf{x} = \begin{pmatrix}
    x_1 \\\\
    x_2 \\\\
    x_3 \\\\
    x_4 \\\\
    x_5
    \end{pmatrix}$$

    and the upper-triangular matrix $Q$ is

    $$Q = \begin{pmatrix}
    -2 & 2 & 2 & 0 & 0 \\
    0 & -2 & 0 & 2 & 0 \\
    0 & 0 & -3 & 2 & 2 \\
    0 & 0 & 0 & -3 & 2 \\
    0 & 0 & 0 & 0 & -2
    \end{pmatrix}.$$

    Every diagonal entry captures the (negative) number of incident edges at the corresponding vertex, while the off-diagonal entries record the quadratic terms contributed by each edge.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, nodes, sparse_graph, x1, x2, x3, x4, x5):
    _x = [x_i.value for x_i in [x1, x2, x3, x4, x5]]
    _cut = dict(zip(nodes, _x))
    _cut_size = sparse_graph.cut_size(_cut)

    sparse_graph.display_cut(_cut)

    mo.vstack(
        [
            x1,
            x2,
            x3,
            x4,
            x5,
            mo.md(f"Cut size: ${_cut_size}$"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 2**

    Use the switches to reproduce each cut shown in the worked example. Record the binary vector, determine the cut size, and confirm the objective value given by $\mathbf{x}^T Q \mathbf{x}$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 3**

    Use `optimise(example_q)` to identify the minimiser of $\mathbf{x}^T Q \mathbf{x}$ for the worked example. Interpret the resulting vector as a cut of the graph and compare it with the cuts from Task&nbsp;2.
    """
    )
    return


@app.cell
def _():
    from foundation import optimise
    return


@app.cell(hide_code=True)
def _(Graph, dense_edges, mo, nodes):
    dense_graph = Graph(config={"node_size": 480, "font_size": 14})
    dense_graph.add_nodes_from(nodes)
    dense_graph.add_edges_from(dense_edges)
    dense_graph.display()
    mo.md(
        r"""
    	A seventh edge connects vertices 1 and 4, giving us a denser example to analyse.
    	"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 4**

    Derive the <span style="font-variant-caps: small-caps">qubo</span> matrix, $Q$, for the dense graph. Confirm that each diagonal entry equals the negative degree of the corresponding vertex and that the off-diagonal entries reflect the new edge.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 5**

    Use `optimise` to find the optimal cut for the denser graph. Compare its objective value with that of the sparser graph.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    from foundation import Graph

    nodes = [1, 2, 3, 4, 5]
    sparse_edges = [(1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5)]
    dense_edges = [(1, 2), (1, 3), (2, 4), (1, 4), (3, 4), (3, 5), (4, 5)]
    return Graph, dense_edges, nodes, sparse_edges


@app.cell(hide_code=True)
def _(mo):
    x1 = mo.ui.switch(value=False, label="$x_1$")
    x2 = mo.ui.switch(value=False, label="$x_2$")
    x3 = mo.ui.switch(value=False, label="$x_3$")
    x4 = mo.ui.switch(value=False, label="$x_4$")
    x5 = mo.ui.switch(value=False, label="$x_5$")
    switches = [x1, x2, x3, x4, x5]
    return x1, x2, x3, x4, x5


if __name__ == "__main__":
    app.run()
