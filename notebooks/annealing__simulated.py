import marimo

__generated_with = "0.16.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Simulated Annealing

    _Simulated annealing_ is a stochastic global search optimisation algorithm.

    The algorithm is inspired by annealing in _metallurgy_ where metal is heated to a high temperature quickly, then cooled slowly.

    The physical annealing process works by first exciting the atoms in the material at a high temperature, allowing the atoms to move around a lot, then decreasing their excitement slowly, allowing the atoms to fall into a new, more stable configuration.

    Simulated annealing mimics the physical annealing process. (We would like to point out that it is not a physical process but it is an analogy).

    It can be considered as a modified version of stochastic hill climbing. Stochastic hill climbing maintains a single candidate solution and takes steps of a random but constrained size from the candidate in the search space. If the new point is better than the current point, then the current point is replaced with the new point. This process continues for a fixed number of iterations.

    The following animation illustrates the simulated annealing process for a two-dimensional function optimisation:

    ![Simulated Annealing Animation](https://upload.wikimedia.org/wikipedia/commons/d/d5/Hill_Climbing_with_Simulated_Annealing.gif)

    _Animation showing simulated annealing finding the global minimum of a function. Image from [Wikipedia](https://en.wikipedia.org/wiki/Simulated_annealing)._

    Simulated annealing executes the search in the same way.

    The main difference is that new points that are not as good as the current point (worse points) are accepted sometimes. A worse point is accepted probabilistically where the likelihood of accepting a solution worse than the current solution is a function of the _temperature_ of the search and haow much worse the solution is than the current solution. Moving to worse solutions allows escaping from local minima. The temperature is decreased gradually, making unfavourable changes less probable as the process continues. Zero temperature is simply the hill climbing algorithm.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Algorithm

    **Step 1:** Start with an initial feasible solution $s=s_0$ and temperature $t=t_0$.

    **Step 2:** Until the termination conditions are reached, repeat the following:

    * Pick a solution $s'$ from the neighbourhood of solutions $N(s)$.
    * Let $\Delta$ be the difference between cost of $s'$ and $s$.
    * If $\Delta < 0$, accept the new solution, i.e. $s=s'$. Otherwise, pick a random number $p$ between $0$ and $1$. Accept $s'$ if $e^{-\Delta c/t} > p$.
    * Calculate the new temperature $t$ according to the _annealing schedule_.

    ### Notes

    - The annealing schedule describes how temperature decreases in time. Most common choices are:

        - linear ($t = t - a$), and
        - geometric ($t = t \cdot a$).

    - The neighbourhood of solutions is obtained by altering the current state.

    - The termination condition can be a fixed number of iterations or reaching some acceptable threshold of performance.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## D-Wave Simulated Annealer

    Now we will investigate how we can run simulated annealing algorithm from D-Wave library, `dwave-neal`. Let's demonstrate simulated annealing with a simple <span style="font-variant: small-caps;">BQM</span> example:
    """
    )
    return


@app.cell
def _():
    from dimod import BQM
    from dwave.samplers import SimulatedAnnealingSampler

    _sampler = SimulatedAnnealingSampler()

    _linear = {"x1": -5, "x2": -3, "x3": -8, "x4": -6}
    _quadratic = {
        ("x1", "x2"): 4,
        ("x1", "x3"): 8,
        ("x2", "x3"): 2,
        ("x3", "x4"): 10,
    }
    _vartype = "BINARY"

    _bqm = BQM(_linear, _quadratic, _vartype)

    _sampleset = _sampler.sample(_bqm, num_reads=10)
    print(_sampleset)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In the example above, we use `SimulatedAnnealingSampler` to find the ground state of the <span style="font-variant: small-caps;">BQM</span>.

    One parameter we have used is the `num_reads`, which determines how many runs of the simulated annealing algorithm we would like to call. Each line in the output corresponds to solution found in one run of the algorithm.

    There are also additional parameters you can provide such as `beta_schedule` and `num_sweeps` but we will not go into detail.

    Note that since the algorithm is stochastic, having multiple runs helps us to estimate better the minimum energy sample.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 1**

    Find out what assignment of $x_1$ and $x_2$ minimises the following objective function using simulated annealing. Set number of reads to 1000.

    $$5x_1 + 7x_1 x_2 - 3x_2 + 2$$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    There are additional parameters that you can define when running simulated annealing.

    It is also possible to input a <span style="font-variant: small-caps;">QUBO</span> dictionary for the sampler through the function `sample_qubo` and an Ising model by providing `h` and `J` using function `sample_ising`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 2**

    Use simulated annealing to find out the assignment that gives the minimum energy for the following <span style="font-variant: small-caps;">qubo</span> dictionary. Set number of reads to $1000$.
    """
    )
    return


@app.cell
def _(mo):
    Q_dict = {
        ("x1", "x1"): 3,
        ("x2", "x2"): -7,
        ("x3", "x3"): 11,
        ("x4", "x4"): -1,
        ("x1", "x2"): 9,
        ("x1", "x3"): 1,
        ("x2", "x3"): 2,
        ("x3", "x4"): 8,
    }

    mo.md(
        f'**<span style="font-variant: small-caps;">QUBO</span> dictionary:** `{Q_dict}`'
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 3**

    Use simulated annealing to find out the assignment that gives the minimum energy for the following Ising Model defined through `h` and `J` parameters. Set number of reads to $1000$.
    """
    )
    return


@app.cell
def _(mo):
    h = {"s1": 3, "s2": 1, "s3": 4, "s4": 2}
    J = {("s1", "s2"): 4, ("s1", "s3"): 1, ("s1", "s4"): 6, ("s3", "s4"): 7}

    mo.md(
        f"""
    **Ising Model parameters:**

    - h (external fields): `{h}`
    - J (coupling strengths): `{J}`
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
