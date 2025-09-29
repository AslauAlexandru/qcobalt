"""This module contains a Marimo application for testing and demonstration purposes."""

import marimo

__generated_with = "0.16.0"
app = marimo.App(app_title="QCobalt", html_head_file="")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# QCobalt""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Welcome to QWorld's introductory tutorial on **quantum annealing.**

    We will start with the basics of combinatorial optimisation and why it matters. Next, we will introduce mathematical foundations for _quadratic unconstrained binary optimisation_ (<span style="font-variant-caps: small-caps">qubo</span>) and _Ising models_. In the last section, we will finally learn about _simulated_ and _quantum annealing._
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Contents


    1. [Combinatorial Optimisation](?file=notebooks/combinatorial_optimisation.py)
    1. Quadratic Unconstrained Binary Optimisation
        1. [Definition](?file=notebooks/qubo__definition.py)
        1. [Max-Cut](?file=notebooks/qubo__max_cut.py)
        1. [Penalty Method](?file=notebooks/qubo__penalty_method.py)
        1. [Travelling Salesman Problem](?file=notebooks/qubo__tsp.py)
        1. [Graph Colouring](?file=notebooks/qubo__graph_colouring.py)
        1. [Higher-Order Binary Optimisation](?file=notebooks/qubo__hobo.py)
    1. Ising Models
        1. [Definition](?file=notebooks/ising__definition.py)
        1. [Conversion: Ising Model → QUBO](?file=notebooks/ising__qubo.py)
    1. Annealing
        1. [`BinaryQuadraticModel`](?file=notebooks/annealing__bqm.py)
        1. [Conversions](?file=notebooks/annealing__conversions.py)
        1. [Max-Cut](?file=notebooks/annealing__max_cut.py)
        1. [Simulated Annealing](?file=notebooks/annealing__simulated.py)
        1. [Travelling Salesman Problem](?file=notebooks/annealing__tsp.py)
        1. [Graph Colouring](?file=notebooks/annealing__graph_colouring.py)
        1. [Quantum Annealing](?file=notebooks/annealing__quantum.py)
    1. [References](?file=notebooks/references.py)
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
