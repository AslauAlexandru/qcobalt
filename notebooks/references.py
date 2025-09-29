import marimo

__generated_with = "0.16.0"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Acknowledgements""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## People

    - Dr Özlem Salehi Köken — lead and author of multiple notebooks
    - AkashNarayanan B
    - Paul Joseph Robin
    - Sabah Ud Din Ahmad
    - Sourabh Nutakki
    - Manan Sood
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## References

    - D‑Wave Documentation. Solving Problems with D‑Wave Solvers. https://docs.dwavesys.com/docs/latest/c_gs_3.html
    - D‑Wave Ocean docs. Concept: Minor‑Embedding. https://docs.ocean.dwavesys.com/en/stable/concepts/embedding.html
    - D‑Wave Systems. Systems overview. https://www.dwavesys.com/solutions-and-products/systems/
    - `dimod` documentation: Binary Quadratic Models. https://test-projecttemplate-dimod.readthedocs.io/en/latest/reference/bqm/index.html
    - D‑Wave Ocean documentation: Classical Solvers. https://docs.ocean.dwavesys.com/en/stable/overview/cpu.html#
    - D‑Wave Ocean documentation: Exact Solver. https://docs.ocean.dwavesys.com/en/stable/docs_dimod/reference/sampler_composites/samplers.html#exact-solver
    - D‑Wave Examples: Maximum Cut. https://github.com/dwave-examples/maximum-cut
    - D‑Wave Examples: Graph Coloring. https://github.com/dwave-examples/graph-coloring
    - D‑Wave Examples: Map Coloring. https://github.com/dwave-examples/map-coloring
    - Qiskit tutorials: Max‑Cut and Travelling Salesman Problem. https://qiskit.org/documentation/optimization/tutorials/06_examples_max_cut_and_tsp.html
    - Glover, Fred; Kochenberger, Gary; Du, Yu (2019). Quantum Bridge Analytics I: A Tutorial on Formulating and Using "QUBO" Models. arXiv:1811.11538.
    - Lucas, Andrew (2014). Ising formulations of many NP problems. Frontiers in Physics 2:5. https://doi.org/10.3389/fphy.2014.00005
    - McGeoch, Catherine C. (2014). Adiabatic quantum computation and quantum annealing: Theory and practice. Synthesis Lectures on Quantum Computing 5(2):1–93.
    - Salehi, Özlem; Glos, Adam; Miszczak, Jarosław Adam (2021). Unconstrained Binary Models of the Travelling Salesman Problem Variants for Quantum Optimization. arXiv:2106.09056.
    - Verma, Amit; Lewis, Mark; Kochenberger, Gary (2021). Efficient QUBO transformation for Higher Degree Pseudo Boolean Functions. arXiv:2107.11695.
    - Dattani, Nike (2019). Quadratization in Discrete Optimization and Quantum Mechanics. arXiv:1901.04405.
    - Kofler, Christian; et al. (2014). A Penalty Function Approach to Max 3‑SAT Problems. Karl‑Franzens‑University Graz working paper. https://static.uni-graz.at/fileadmin/sowi/Working_Paper/2014-04_Kofler_Greistorfer_Wang_Kochenberger.pdf
    - Barahona, Francisco; Grötschel, Martin; Jünger, Michael; Reinelt, Gerhard. An Application of Combinatorial Optimization to Statistical Physics and Circuit Layout Design. https://www.jstor.org/stable/170992
    - Richard Fitzpatrick. The Ising Model. https://farside.ph.utexas.edu/teaching/329/lectures/node110.html
    - Wikipedia: Quadratic unconstrained binary optimization. https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization
    - Wikipedia: Loss function. https://en.wikipedia.org/wiki/Loss_function
    - Wikipedia: Penalty method. https://en.wikipedia.org/wiki/Penalty_method
    - Classical Ising Model (Quantum Machine Learning MOOC: Peter Wittek) — YouTube. https://youtu.be/Wy9YoEYv-fA
    - Ising Model (Prof. G. Ceder) — PDF. http://web.mit.edu/ceder/publications/Ising%20Model.pdf
    - Steven Herbert, Quantum Computing Lecture 15: Adiabatic Quantum Computing. https://www.cl.cam.ac.uk/teaching/1920/QuantComp/Quantum_Computing_Lecture_15.pdf
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
