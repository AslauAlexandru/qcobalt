"""This module contains a Marimo application for testing and demonstration purposes."""

import marimo

__generated_with = "0.16.0"
app = marimo.App(app_title="Intro – QCobalt", html_head_file="")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Combinatorial Optimisation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Combinatorial optimisation is about finding the best possible solution from a vast but finite set of options, much like choosing the most efficient route from millions of possible paths.

    You'll encounter these problems everywhere in daily life: from navigation systems calculating your quickest journey home, to schools arranging their timetables, and delivery companies working out the most cost-effective way to distribute parcels. The key challenge here is that simply trying every possible combination would take far too long for real-world applications (imagine a delivery van having to test every conceivable route through a city before making its first delivery). Instead, we use mathematical techniques to find solutions that are optimal (or close to optimal) within a reasonable timeframe.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Travelling Salesman Problem (TSP)

    Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?

    This deceptively simple question turns out to be very hard to answer in practice. To begin to answer it, we need to mathematically formalise the problem: instead of having a list of cities, we'll have a [graph](https://wikipedia.org/wiki/Graph_(discrete_mathematics)) where each vertex represents a city and the edges between vertices represent the distances between cities.

    The graph below is an _undirected_ graph, where the cost of moving between cities is the same in both direction. These graph may also be directed, in which case different costs may be assigned for each direction.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    from foundation import Graph

    # Create a simple weighted undirected graph representing cities and distances
    _g = Graph()
    _g.add_node("A")
    _g.add_node("B")
    _g.add_node("C")
    _g.add_node("D")
    _g.add_edge("A", "B", weight=10)
    _g.add_edge("B", "C", weight=35)
    _g.add_edge("C", "D", weight=30)
    _g.add_edge("D", "A", weight=20)
    _g.add_edge("A", "C", weight=15)
    _g.add_edge("B", "D", weight=25)
    _g.display()
    return (Graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Naïve solution: search every path

    This approach involves examining every possible route to guarantee finding the shortest path. Although this brute-force technique delivers the mathematically perfect answer rather than just a good approximation, it becomes utterly impractical when dealing with a large number of cities.

    Note that it doesn't matter which city serves as your starting point, since the routes `A-B-C-D-A` and `B-C-D-A-B` represent identical journeys travelled in the same sequence. By fixing one city as the departure point, we can eliminate this redundancy when comparing the costs of all potential tours.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 1**

    Solve the travelling salesman problem for the following graph.
    """
    )
    return


@app.cell(hide_code=True)
def _(Graph):
    tsp_graph = Graph()
    tsp_graph.add_weighted_edges_from(
    	[(0, 1, 12), (0, 2, 14), (0, 3, 17), (1, 2, 15), (1, 3, 18), (2, 3, 29)]
    )
    tsp_graph.display()
    return (tsp_graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Visualisation

    You can use Python to visualise graphs using `networkx` and `matplotlib`. QCobalt's library has built a wrapper around these packages to help you visualise solutions.

    There are many ways to initialise graphs in `networkx` allowing you to specify vertices and (weighted) edges in different forms. Going through the code cells in this notebook is a good place to start; go through [`networkx`'s](networkx.org) documentation for a deeper dive.

    In the code above, we are using natural numbers to name the vertices (unlike English alphabets in the earlier graph). Next, note the array of triplets supplied to `add_weighted_edges_from`. The first two numbers represent two vertices in the graph, and the third number is the weight of the edge connecting them. This is a concise way of defining a `networkx` graph.

    For <span style="font-variant-caps: small-caps">tsp</span> graphs, we are usually interested in visualising paths within the graph, like `0-2-1-3`. We do that in the following way:
    """
    )
    return


@app.cell
def _(tsp_graph):
    tsp_graph.display_path([0, 2, 1, 3])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Graph Colouring

    Graph colouring involves assigning colours to each vertex in a graph while ensuring that no two adjacent vertices share the same colour. The goal is to accomplish this using as few colours as possible, and the minimum number required is termed the _chromatic number_ of that particular graph.

    This problem is [NP-hard](https://wikipedia.org/wiki/NP-hardness). There's also a related decision version that poses the question "Can we colour this graph using exactly $k$ colours?" This variant becomes [NP-complete](https://wikipedia.org/wiki/NP-completeness) when $k \ge 3$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Visualisation

    You can use the visualisation tool we have created by defining the colours of the vertices as a `dict` and calling the `display_colouring` method on the graph.
    """
    )
    return


@app.cell
def _(Graph):
    colouring_graph = Graph()
    colouring_graph.add_edges_from(
    	[
    		("A", "C"),
    		("A", "E"),
    		("B", "C"),
    		("B", "E"),
    		("B", "D"),
    		("C", "D"),
    		("D", "E"),
    	]
    )
    colouring = {"A": 0, "B": 0, "C": 1, "D": 2, "E": 1}
    colouring_graph.display_colouring(colouring)
    return (colouring_graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Greedy Solution

    The naïve approach involves testing every possible colour assignment, gradually increasing the number of colours until a valid solution emerges. However, there's a more practical greedy algorithm that produces approximate solutions much faster.

    This greedy method works by establishing an arbitrary ordering for both vertices and colours at the outset. You begin with the first vertex and assign it the first colour. Moving to the next vertex, you examine which colours are already used by adjacent vertices, then select the lowest-numbered available colour that won't create a conflict. This process continues systematically until every vertex receives a colour assignment, delivering a workable solution quickly even if it occasionally uses more colours than the theoretical minimum.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Explore the greedy heuristic

    The next two tasks use the same graph. Try running your own implementation of the greedy algorithm (or work carefully by hand) to see how the chosen ordering influences the number of colours required.
    """
    )
    return


@app.cell(hide_code=True)
def _(Graph):
    greedy_task_graph = Graph()
    greedy_task_graph.add_edges_from(
    	[
    		("A", "B"),
    		("A", "E"),
    		("A", "F"),
    		("B", "C"),
    		("B", "G"),
    		("C", "D"),
    		("C", "H"),
    		("D", "E"),
    		("D", "I"),
    		("E", "J"),
    		("F", "H"),
    		("F", "I"),
    		("G", "I"),
    		("G", "J"),
    		("H", "J"),
    	]
    )
    greedy_task_graph.display()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 2**

    Use greedy colouring with the vertex order `A, B, E, F, G, H, C, I, J, D` and the colour order `red → blue → green`. Can you colour the graph above using only three colours?

    /// admonition | Note
    If you test your answer by visualising the colouring, you may see a different layout. This is fine as long as the adjacency structure matches.
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 3**

    Repeat the greedy procedure, but this time use the vertex order `A, H, I, B, E, J, G, F, C, D` with the same colour priority `red → blue → green`. Do you still obtain a three-colour solution?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The outcome of Task 3 demonstrates that the greedy algorithm can settle on a suboptimal colouring when the vertex order is unfavourable, even though a three-colour solution exists (as you observed in Task 2).""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 4** <span style="font-variant-caps: small-caps">optional</span>

    Can you suggest a lower bound on the chromatic number of a graph? Explain your reasoning.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Max-Cut

    Given a graph, the aim is to partition the vertices into two disjoint sets (a _cut_) so that the number of edges between the two sets is as large as possible. The size of the cut is simply the number of edges that cross between the sets. This problem is generally NP-hard, although particular graph families admit efficient strategies.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let us revisit the five-vertex graph from earlier. By splitting the vertices into two sets we can highlight the edges counted by the cut.""")
    return


@app.cell
def _(colouring_graph):
    colouring_graph.display_cut(["A", "B"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### A simple case: bipartite graphs

    A [bipartite graph](https://wikipedia.org/wiki/Bipartite_graph) has its vertices split into two sets $U$ and $V$ with no edges inside either set. Every edge connects a vertex in $U$ to a vertex in $V$. For such graphs the max cut equals the total number of edges, because every edge already crosses between the sets. Recognising bipartite graphs can be done in polynomial time, so finding the max cut is easy in this special case.
    """
    )
    return


@app.cell
def _(Graph):
    bipartite_graph = Graph()
    bipartite_graph.add_edges_from(
    	[
    		("u0", "v0"),
    		("u0", "v1"),
    		("u1", "v0"),
    		("u1", "v2"),
    		("u2", "v1"),
    		("u2", "v2"),
    	]
    )
    bipartite_graph.display_cut({"u0": 0, "u1": 0, "u2": 0, "v0": 1, "v1": 1, "v2": 1})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 5**

    Solve the max-cut problem for the graph shown below. Describe both the partition and the size of the cut you obtain. Remember that multiple optimal cuts may exist.
    """
    )
    return


@app.cell(hide_code=True)
def _(Graph):
    task5_graph = Graph()
    task5_graph.add_edges_from(
    	[("A", "D"), ("B", "D"), ("C", "D"), ("E", "D"), ("F", "D")]
    )
    task5_graph.display()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 6**

    In the next graph the maximum cut has already been marked by the partition $\{\,B, D, G\,\}$. Determine how many edges belong to this cut.
    """
    )
    return


@app.cell(hide_code=True)
def _(Graph):
    task6_graph = Graph()
    task6_graph.add_edges_from(
    	[
    		("A", "B"),
    		("A", "E"),
    		("B", "C"),
    		("B", "E"),
    		("C", "D"),
    		("C", "F"),
    		("D", "E"),
    		("D", "F"),
    		("E", "F"),
    		("E", "G"),
    		("F", "G"),
    	]
    )
    task6_graph.display_cut(["B", "D", "G"])
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
