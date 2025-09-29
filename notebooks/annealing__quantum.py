import marimo

__generated_with = "0.16.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Quantum Annealing

    Quantum annealing (<span style="font-variant: small-caps;">qa</span>) is a physical optimisation process that evolves a
    quantum system towards the ground state of a problem Hamiltonian. In the
    context of combinatorial optimisation, we encode our objective as an
    Ising model or a quadratic unconstrained binary optimisation
    (<span style="font-variant: small-caps;">qubo</span>) and ask the
    hardware to sample low-energy states.

    In this notebook, we will:

    - prepare a small Max-Cut instance as a <span style="font-variant: small-caps;">bqm</span>,
    - attempt to solve it on a quantum processing unit (<span style="font-variant: small-caps;">qpu</span>) if configured,
    - otherwise, fall back to a classical simulated annealer, and
    - visualise the best sample returned by the sampler.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Background and Adiabatic Picture

    Quantum annealing (<span style="font-variant: small-caps;">qa</span>) is a heuristic method motivated by the
    adiabatic model of quantum computation. The key idea is to prepare a
    quantum system in an easy-to-prepare ground state and then evolve it
    slowly so that it remains in the instantaneous ground state of a
    time-dependent Hamiltonian.

    > Quantum Adiabatic Theorem (informal): A quantum system that starts in
    > the ground state of a time-dependent Hamiltonian remains in the ground
    > state provided the Hamiltonian changes sufficiently slowly.

    A common interpolation is

    $$
    H(t) = \Bigl(1 - \tfrac{t}{\tau}\Bigr)\, H_0 \; + \; \tfrac{t}{\tau}\, H_p,
    $$

    where $H_0$ is an initial Hamiltonian with a ground state that is simple
    to prepare and $H_p$ is the problem Hamiltonian whose ground state encodes
    the solution. At $t=0$ only $H_0$ acts; at $t=\tau$ only $H_p$ acts.
    If the evolution is slow enough, the system finishes in the ground state
    of $H_p$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## <span style="font-variant: small-caps;">qa</span> versus <span style="font-variant: small-caps;">aqc</span>

    - In practice, quantum annealing relaxes some assumptions of adiabatic
      quantum computing: devices are open systems and the schedule is not
      necessarily adiabatic. <span style="font-variant: small-caps;">qa</span> is therefore heuristic.
    - The problem Hamiltonian has a restricted, Ising-like form corresponding
      to a classical objective function.
    - Adiabatic quantum computing is universal (gate-model equivalent),
      whereas <span style="font-variant: small-caps;">qa</span> is not designed for universal quantum computation.

    Current commercial QA hardware is offered by D-Wave Systems.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Initial and Problem Hamiltonians

    A typical choice for the initial Hamiltonian is a transverse field,

    $$
    H_0 = - \sum_{i=1}^n \sigma_i^x,
    $$

    whose ground state is $|+\rangle^{\otimes n}$, easy to prepare.

    On D-Wave hardware, the problem Hamiltonian has the Ising form

    $$
    H_p = \sum_{i} h_i\, \sigma_i^z \; + \; \sum_{(i,j)} J_{ij}\, \sigma_i^z\,\sigma_j^z,
    $$

    where $\sigma^z$ has eigenstates $|0\rangle$ and $|1\rangle$ with
    eigenvalues $+1$ and $-1$, respectively. If we denote spins by
    $s_i\in\{-1,1\}$, the energy of a computational basis state becomes

    $$
    E(\mathbf{s}) = \sum_i h_i s_i + \sum_{(i,j)} J_{ij} s_i s_j,
    $$

    which is exactly the Ising model energy. Consequently, if we formulate a
    problem as an Ising model (or equivalently a <span style=\"font-variant: small-caps;\">qubo</span>),
    we can target it with quantum annealing.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Preparing a small Max-Cut instance

    We will reuse the house-with-an-X graph (five vertices) and build the
    corresponding <span style="font-variant: small-caps;">qubo</span>
    using the standard Max-Cut formulation:

    $$\min\sum_{(i,j)\in E} -x_i - x_j + 2x_ix_j.$$
    """
    )
    return


@app.cell
def _():
    from foundation import Graph
    from networkx import house_x_graph

    G = Graph(house_x_graph())
    G.display()
    return (G,)


@app.cell
def _(G):
    from collections import defaultdict

    _d = defaultdict(int)
    for _i, _j in G.edges:
        _d[(_i, _i)] += -1
        _d[(_j, _j)] += -1
        _d[(_i, _j)] += 2

    qubo_dict = dict(_d)
    return (qubo_dict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Choosing a sampler

    We try to construct a <span style="font-variant: small-caps;">qpu</span>-backed sampler via
    `EmbeddingComposite(DWaveSampler())`. If this is not available (for
    example, no API configuration), we fall back to a classical simulated
    annealer. This allows you to run the notebook locally without a QPU.
    """
    )
    return


@app.cell
def _():
    is_qpu = False
    sampler = None
    try:
        from dwave.system import DWaveSampler, EmbeddingComposite

        sampler = EmbeddingComposite(DWaveSampler())
        is_qpu = True
    except Exception:  # noqa: BLE001 - handle import/config/runtime issues
        from dwave.samplers import SimulatedAnnealingSampler

        sampler = SimulatedAnnealingSampler()
        is_qpu = False
    return is_qpu, sampler


@app.cell(hide_code=True)
def _(is_qpu, mo):
    _msg = (
        "Using <span style=\"font-variant: small-caps;\">qpu</span>-backed sampler (EmbeddingComposite + DWaveSampler)."
        if is_qpu
        else "<span style=\"font-variant: small-caps;\">qpu</span> not available; using SimulatedAnnealingSampler."
    )
    mo.md(
        rf"""
    > {_msg}
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Sampling parameters

    Typical knobs when using quantum annealing include number of reads, chain
    strength (when minor-embedding is required), and annealing time. We keep
    parameters conservative here so the example runs both on <span style="font-variant: small-caps;">qpu</span> and locally.
    """
    )
    return


@app.cell
def _():
    num_reads = 100
    annealing_time = 20.0  # microseconds; ignored by classical fallback
    chain_strength = None  # let the composite choose a default if QPU is used
    return annealing_time, chain_strength, num_reads


@app.cell
def _(annealing_time, chain_strength, num_reads, qubo_dict, sampler):
    from dimod import BQM

    bqm = BQM.from_qubo(qubo_dict)

    _kwargs: dict[str, object] = {"num_reads": num_reads}
    if chain_strength is not None:
        _kwargs["chain_strength"] = chain_strength
    if annealing_time is not None:
        _kwargs["annealing_time"] = annealing_time

    sampleset = sampler.sample(bqm, **_kwargs)
    return (sampleset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We recover the lowest-energy sample and visualise the cut. For Max-Cut,
    the objective value equals the number of cut edges up to a sign and
    constant shift; our plotting helper highlights edges in the cut.
    """
    )
    return


@app.cell
def _(G, sampleset):
    best_sample = sampleset.first.sample
    print(sampleset)
    G.display_cut(best_sample)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 1**

    Change `num_reads`, `annealing_time`, and (if on a <span style="font-variant: small-caps;">qpu</span>) `chain_strength`.
    Rerun the sampler cell and observe how frequently the best solution is
    found. Record your observations.

    ## **Task 2**

    Replace the graph with a different small instance (for example,
    `turan_graph(5, 3)` or `truncated_tetrahedron_graph()` from
    `networkx`). Rebuild the <span style="font-variant: small-caps;">qubo</span>
    dictionary, sample, and visualise the resulting cut.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
