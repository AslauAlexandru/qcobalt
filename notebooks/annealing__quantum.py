import marimo

__generated_with = "0.16.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Quantum Annealing

    Quantum annealing is a physical optimisation process that evolves a
    quantum system towards the ground state of a problem Hamiltonian. In the
    context of combinatorial optimisation, we encode our objective as an
    Ising model or a quadratic unconstrained binary optimisation
    (<span style="font-variant: small-caps;">qubo</span>) and ask the
    hardware to sample low-energy states.

    In this notebook, we will:

    - prepare a small Max-Cut instance as a <span style="font-variant: small-caps;">bqm</span>,
    - attempt to solve it on a quantum processing unit (QPU) if configured,
    - otherwise, fall back to a classical simulated annealer, and
    - visualise the best sample returned by the sampler.
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

    We try to construct a QPU-backed sampler via
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
        "Using QPU-backed sampler (EmbeddingComposite + DWaveSampler)."
        if is_qpu
        else "QPU not available; using SimulatedAnnealingSampler."
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
    parameters conservative here so the example runs both on QPU and locally.
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

    Change `num_reads`, `annealing_time`, and (if on a QPU) `chain_strength`.
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
