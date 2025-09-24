"""Marimo notebook deriving a QUBO formulation for the travelling salesman problem."""

import marimo

__generated_with = "0.16.0"
app = marimo.App(app_title="QUBO TSP – QCobalt", html_head_file="")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# <span style="font-variant-caps: small-caps">qubo</span> Travelling Salesman Problem""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""This notebook translates the travelling salesman problem (<span style="font-variant-caps: small-caps">tsp</span>) into a quadratic non-linear objective that fits the quadratic unconstrained binary optimisation (<span style="font-variant-caps: small-caps">qubo</span>) framework. We revisit the penalty method, define binary decision variables that encode tours, and assemble the full <span style="font-variant-caps: small-caps">qubo</span> including its matrix representation.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Travelling Salesman Problem

    The classical <span style="font-variant-caps: small-caps">tsp</span> supplies a weighted complete graph whose vertices represent cities and whose edge weights store travel costs. A feasible tour visits each city exactly once before returning to the starting point. The aim is to minimise the total cost of the circuit.

    We write the vertex set as $V = \{0, 1, \dots, N - 1\}$ and the cost of travelling from city $i$ to city $j$ as $w_{ij}$. The costs need not be symmetric: $w_{ij}$ can differ from $w_{ji}$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Binary Encoding

    To express the tour with binary variables we introduce $N^2$ decision variables $x_{i, t}$. Index $i$ denotes the city and $t$ denotes the position in the tour (both count from zero):

    $$x_{i, t} = \begin{cases}
    1, & \text{if city } i \text{ is visited at time } t, \\
    0, & \text{otherwise.}
    \end{cases}$$

    The resulting $N \times N$ binary grid records which city is selected at every time step of the tour.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Constraints

    A valid <span style="font-variant-caps: small-caps">tsp</span> tour obeys two structural rules.

    1. **Every city is visited exactly once.** For each city index, $i$, the tour must select that city at precisely one time:
       $$\sum_{t = 0}^{N-1} x_{i, t} = 1 \qquad \forall i.$$
       Translating the restriction into a penalty gives
       $$P \cdot \sum_{i = 0}^{N-1} \left(1 - \sum_{t = 0}^{N-1} x_{i, t}\right)^2.$$
       Expanding the square and using $x_{i, t}^2 = x_{i, t}$ reveals a linear coefficient of $-P$ on every $x_{i, t}$ and quadratic coefficients of $+2P$ for each pair $x_{i, t} x_{i, t'}$ with $t \neq t'.$

    2. **Exactly one city is chosen at every time step.** For each position, $t$, the tour must select a single city:
       $$\sum_{i = 0}^{N-1} x_{i, t} = 1 \qquad \forall t.$$
       The matching penalty reads
       $$P \cdot \sum_{t = 0}^{N-1} \left(1 - \sum_{i = 0}^{N-1} x_{i, t}\right)^2.$$
       Its expansion mirrors the previous case with indices swapped, so we again obtain $-P$ contributions on the diagonal and $+2P$ between variables vying for the same time slot.

    Each family adds a constant offset of $NP$. We keep that offset separate from the quadratic form and restore it after optimisation. Selecting $P$ larger than the largest edge weight ensures the optimiser prefers feasible assignments.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Tour Cost

    The final ingredient is the travel cost. Whenever the tour visits city $i$ at time $t$ and city $j$ at time $t + 1$, the edge $(i, j)$ contributes $w_{ij}$ to the objective. Using the binary variables, the cost term is

    $$\sum_{\substack{i, j = 0 \\ i \neq j}}^{N-1} w_{ij} \sum_{t=0}^{N-1} x_{i, t} x_{j, (t + 1) \bmod N}.$$

    The final summand ensures that the return journey from the final city to the start is included. A small penalty coefficient can accidentally favour tours that violate the constraints, so $P$ must exceed the largest edge weight.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Complete Formulation

    Combining the penalties with the travel cost produces the <span style="font-variant-caps: small-caps">qubo</span>

    $$\begin{aligned}
    H(\mathbf{x}) = &\; P \cdot \sum_{i=0}^{N-1} \left(1 - \sum_{t=0}^{N-1} x_{i, t}\right)^2 + P \cdot \sum_{t=0}^{N-1} \left(1 - \sum_{i=0}^{N-1} x_{i, t}\right)^2 \\
    &+ \sum_{\substack{i, j = 0 \\ i \neq j}}^{N-1} w_{ij} \sum_{t=0}^{N-1} x_{i, t} x_{j, (t + 1) \bmod N}.
    \end{aligned}$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Worked Example

    ![](public/tsp_3.png)

    The figure shows three cities labelled $0$, $1$, and $2$ with the cost matrix

    $$W = \begin{pmatrix} 0 & 10 & 7 \\ 15 & 0 & 9 \\ 14 & 8 & 0 \end{pmatrix}.$$

    We enumerate the nine binary variables in lexicographic order:

    $$x_{0,0}, x_{0,1}, x_{0,2}, x_{1,0}, x_{1,1}, x_{1,2}, x_{2,0}, x_{2,1}, x_{2,2}.$$

    The travel costs therefore satisfy

    * $w_{01} = 10$,
    * $w_{10} = 15$,
    * $w_{02} = 7$,
    * $w_{20} = 14$,
    * $w_{12} = 9$, and
    * $w_{21} = 8$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Time-indexed constraint

    The first penalty family enforces that exactly one city is chosen at each time step:
    $$P \cdot \sum_{t = 0}^{2} \left(1 - (x_{0,t} + x_{1,t} + x_{2,t})\right)^2.$$
    Expanding the sum, we obtain:
    $$P \cdot \left[\left(1 - (x_{0,0}+x_{1,0}+x_{2,0})\right)^2+\left(1 - (x_{0,1}+x_{1,1}+x_{2,1})\right)^2+\left(1 - (x_{0,2}+x_{1,2}+x_{2,2})\right)^2\right]$$
    For $t = 0$, let's expand $\left(1 - x_{0,0} - x_{1,0} - x_{2,0}\right)^2$:
    $$\left(1 - x_{0,0} - x_{1,0} - x_{2,0}\right)^2 = 1 + x_{0,0}^2 + x_{1,0}^2 + x_{2,0}^2 - 2x_{0,0} - 2x_{1,0} - 2x_{2,0} + 2x_{0,0}x_{1,0} + 2x_{0,0}x_{2,0} + 2x_{1,0}x_{2,0}$$
    Using the identity $x_{i,t}^2 \equiv x_{i,t}$ for binary variables, this simplifies to:
    $$1 - x_{0,0} - x_{1,0} - x_{2,0} + 2x_{0,0}x_{1,0} + 2x_{0,0}x_{2,0} + 2x_{1,0}x_{2,0}$$
    Note that each term $x_{i,0}$ appears once with coefficient $-1$, and all possible combinations of $x_{i,0}$ as quadratic terms have coefficient $+2$. There's also a constant of $+1$.

    The same structure applies to $\left(1 - (x_{0,1}+x_{1,1}+x_{2,1})\right)^2$ and $\left(1 - (x_{0,2}+x_{1,2}+x_{2,2})\right)^2$.

    Multiplying by $P$ and summing over all time steps shows that:

    - Each $x_{i,t}$ acquires a coefficient of $-P$
    - Each pair $x_{i,t} x_{j,t}$ with $i \neq j$ and the same time index gains $+2P$
    - The constant offset is $3P$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### City-indexed constraint

    The second penalty family ensures that each city is visited exactly once:
    $$P \cdot \sum_{i = 0}^{2} \left(1 - (x_{i,0} + x_{i,1} + x_{i,2})\right)^2.$$

    This calculation mirrors the time-indexed case with indices exchanged. Following the same algebraic expansion as above:

    - Each $x_{i,t}$ collects another coefficient of $-P$
    - Each pair $x_{i,t} x_{i,t'}$ with $t \neq t'$ and the same city index gains $+2P$
    - The offset gains a second contribution of $3P$

    Together, the two penalty families place $-2P$ on the diagonal entries of $Q$, $+2P$ between variables that share a city or a time, and accumulate the constant offset $2 N P = 6P$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Matrix Form

    Travel costs contribute
    $$\sum_{\substack{i, j = 0 \\ i \neq j}}^{2} w_{ij} \bigl(x_{i,0} x_{j,1} + x_{i,1} x_{j,2} + x_{i,2} x_{j,0}\bigr),$$

    There are 6 possible $(i,j)$ pairs: $(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)$. The summation expands to:
    $$w_{01} (x_{0,0}x_{1,1} + x_{0,1}x_{1,2} + x_{0,2}x_{1,0}) + w_{02} (x_{0,0}x_{2,1} + x_{0,1}x_{2,2} + x_{0,2}x_{2,0}) + \ldots + w_{21} (x_{2,0}x_{1,1} + x_{2,1}x_{1,2} + x_{2,2}x_{1,0})$$

    Each term $x_{i,t} x_{j,(t+1) \bmod 3}$ appears with coefficient $w_{ij}$, adding the appropriate edge weights between consecutive time slots.

    The variables are ordered as: $x_{0,0}, x_{0,1}, x_{0,2}, x_{1,0}, x_{1,1}, x_{1,2}, x_{2,0}, x_{2,1}, x_{2,2}$.

    Combining all terms from penalties and travel costs, for each variable $x_{i,t}$:

    - Each $x_{i,t}$ has coefficient $-2P$ (from both constraint families)
    - Variables $x_{i,t}$ and $x_{j,t}$ with $i \neq j$ have coefficient $2P$
    - Variables $x_{i,t}$ and $x_{i,t'}$ with $t \neq t'$ have coefficient $2P$
    - Variables $x_{i,t}$ and $x_{j,(t+1) \bmod 3}$ with $i \neq j$ have coefficient $w_{ij}$

    Keeping only the upper-triangular portion produces:

    $$
    Q = \begin{pmatrix}
           -2P & 2P & 2P & 2P & w_{01} & w_{10} & 2P & w_{02} & w_{20} \\
            0  & -2P & 2P & w_{10} & 2P & w_{01} & w_{20} & 2P & w_{02} \\
            0  &  0  & -2P & w_{01} & w_{10} & 2P & w_{02} & w_{20} & 2P \\
            0  &  0  &  0  & -2P & 2P & 2P & 2P & w_{12} & w_{21} \\
            0  &  0  &  0  &  0  & -2P & 2P & w_{21} & 2P & w_{12} \\
            0  &  0  &  0  &  0  &  0  & -2P & w_{12} & w_{21} & 2P \\
            0  &  0  &  0  &  0  &  0  &  0  & -2P & 2P & 2P \\
            0  &  0  &  0  &  0  &  0  &  0  &  0  & -2P & 2P \\
            0  &  0  &  0  &  0  &  0  &  0  &  0  &  0  & -2P \\
        \end{pmatrix}.
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(np):
    city_labels = ["0", "1", "2"]
    sample_costs = np.array(
        [
            [0, 10, 7],
            [15, 0, 9],
            [14, 8, 0],
        ],
        dtype=float,
    )
    return city_labels, sample_costs


@app.cell(hide_code=True)
def _(mo):
    penalty = mo.ui.slider(
        start=5, stop=60, step=1, value=30, label="Penalty coefficient $P$"
    )
    return (penalty,)


@app.cell(hide_code=True)
def _(
    assignment_as_markdown,
    build_tsp_qubo,
    city_labels,
    decode_tour,
    mo,
    np,
    optimise,
    penalty,
    sample_costs,
    tour_cost,
):
    _qubo, _offset = build_tsp_qubo(sample_costs, penalty.value)
    _value, _assignment = optimise(_qubo.tolist())
    _grid = np.array(_assignment, dtype=int).reshape(
        len(city_labels), len(city_labels)
    )
    _schedule = decode_tour(_assignment, len(city_labels))
    route_labels = None
    _total_cost = None
    if None not in _schedule:
        route_labels = [city_labels[_city] for _city in _schedule]
        _total_cost = tour_cost(_schedule, sample_costs)
    _report = [
        mo.md(assignment_as_markdown(_grid, city_labels)),
    ]
    if route_labels is None:
        _report.append(
            mo.md(
                r"""The assignment violates the constraints. Increase $P$ to penalise invalid tours."""
            )
        )
    else:
        _sequence = " \\rightarrow ".join([*route_labels, route_labels[0]])
        _report.append(
            mo.md(rf"""Tour: ${_sequence}$ with total cost ${_total_cost}$.""")
        )
    mo.vstack(_report)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 1**

    Experiment with the penalty slider. Determine the smallest value of $P$ that yields a feasible tour for this instance, and explain why the cost returned by $x^T Q x$ must be adjusted by the offset.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Fixing the Starting City

    Because a tour can be rotated without changing its cost, we often fix the starting city to reduce the number of variables. Setting $x_{0, 0} = 1$ and removing all variables that involve city $0$ or time $0$ leaves $(N - 1)^2$ variables. The constraints now range over $\{1, \dots, N - 1\}$ and the coefficients in $Q$ adjust accordingly, though the cost terms connecting to the fixed city remain.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 2**

    Apply the fixed-start simplification to the three-city instance. Derive the $4 \times 4$ matrix $Q$ for the variables $x_{1, 1}, x_{1, 2}, x_{2, 1}, x_{2, 2}$, and interpret the optimiser's output as a tour.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    from foundation import optimise

    print(optimise([[1,2],[3,5]]))
    return (optimise,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    from numpy.typing import NDArray


    def build_tsp_qubo(costs: NDArray, penalty: float):
        costs = np.asarray(costs, dtype=float)
        if costs.ndim != 2 or costs.shape[0] != costs.shape[1]:
            raise ValueError("Cost matrix must be square.")
        n = costs.shape[0]
        size = n * n
        Q = np.zeros((size, size), dtype=float)
        penalty = float(penalty)

        def add(index_i: int, index_j: int, value: float) -> None:
            if index_i <= index_j:
                Q[index_i, index_j] += value
            else:
                Q[index_j, index_i] += value

        for city in range(n):
            for time in range(n):
                idx = city * n + time
                add(idx, idx, -2.0 * penalty)
        for time in range(n):
            for city_a in range(n):
                for city_b in range(city_a + 1, n):
                    idx_a = city_a * n + time
                    idx_b = city_b * n + time
                    add(idx_a, idx_b, 2.0 * penalty)
        for city in range(n):
            for time_a in range(n):
                for time_b in range(time_a + 1, n):
                    idx_a = city * n + time_a
                    idx_b = city * n + time_b
                    add(idx_a, idx_b, 2.0 * penalty)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for time in range(n):
                    idx_from = i * n + time
                    idx_to = j * n + ((time + 1) % n)
                    add(idx_from, idx_to, float(costs[i, j]))
        offset = 2.0 * penalty * n
        return Q, offset


    def decode_tour(assignment: list[int] | np.ndarray, n: int):
        vector = np.asarray(assignment, dtype=int)
        if vector.size != n * n:
            raise ValueError("Assignment size must equal n squared.")
        grid = vector.reshape(n, n)
        order: list[int | None] = []
        for time in range(n):
            column = grid[:, time]
            selected = np.flatnonzero(column == 1)
            if selected.size == 1:
                order.append(int(selected[0]))
            else:
                order.append(None)
        return order


    def tour_cost(order: list[int | None], costs: np.ndarray):
        if any(city is None for city in order):
            return None
        costs = np.asarray(costs, dtype=float)
        total = 0.0
        n = len(order)
        for time in range(n):
            current = order[time]
            nxt = order[(time + 1) % n]
            total += float(costs[current, nxt])
        return total


    def assignment_as_markdown(grid: np.ndarray, labels: list[str]):
        n = grid.shape[0]
        header = "| Time | " + " | ".join(labels) + " |"
        separator = "| :--: | " + " | ".join([":--:"] * n) + " |"
        rows = []
        for time in range(n):
            cells = " | ".join(str(int(grid[city, time])) for city in range(n))
            rows.append(f"| {time} | {cells} |")
        return "\n".join([header, separator, *rows])
    return assignment_as_markdown, build_tsp_qubo, decode_tour, np, tour_cost


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
