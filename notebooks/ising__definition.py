import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # The Ising Model

    In this notebook, we will go over the definition of the Ising model and see how we can formulate some combinatorial optimisation problems using it. 

    Why we focus on the Ising model will become clear later on when we learn about quantum annealing.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Definition

    Electrons have a quantum mechanical property called _spin_ which is the angular momentum of the particle. When it is measured, it is either $\frac{h}{2}$ (spin up) or $-\frac{h}{2}$ (spin down) where $h$ is Planck's constant.

    An electron's spin is closely related to its magnetic moment, so that an electron behaves like a tiny bar magnet with a North (N) and a South (S) pole. _Ferromagnetism_ arises when a collection of atomic spins align such that their associated magnetic moments all point in the same direction, and the spins behave like a big magnet with a net macroscopic magnetic moment.
    ![](public/spin.jpeg)
    The **Ising Model** is a _mathematical model_ to study ferromagnetism in statistical physics. The Ising model was first proposed by Wilhelm Lenz who gave it as a problem to his graduate student Ernst Ising, after whom this model is named. 

    For simplicity, we will say each spin takes either the value $s=1$ (up) or $s=-1$ (down).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    When spins are arranged on a 1-D line, so that each spin interacts only with its right and left neighbours, the model is called the **1-Dimensional Ising Model**.

    ![](public/Ising_1D.png)

    When the spins are arranged on a 2-D lattice, so that each spin interacts with its right, left, up and down neighbours, the model is also known as the **2-Dimensional Ising Model**.

    ![](public/Ising_2D.png)

    The configuration of spins yielding the lowest energy is known as the _ground state_. It is NP-Hard to find the ground state of a 2-D Ising model. Thus, finding the ground state is as hard as problems like the Max-Cut problem and the Travelling Salesman Problem.

    Note that spins can be arranged in any other configuration.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Energy of the System

    We would like to express the energy of every possible configuration of the spins in the system. We will assume that all possible couplings are possible between any two spins.

    - The spins interact with the external magnetic field $h$, if present. 
    - Each spin state (variable) interacts with its neighbours. The _coupling strength_, of this spin-spin interaction, is characterised by the constant $J$.
    - Each spin variable $s_i$ take the values $\{-1,1\}$.

    Based on those assumptions, the energy of the Ising Model is given as:

    $$ E_{\text{ising}}(\mathbf{s}) =   \sum_{i<j} J_{i,j} s_i s_j + \sum_i h_{i} s_i $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 1**

    Given the Ising Model with the following properties, write a function to calculate the energy for different spin assignments and use that function to find the lowest energy state.

    - There are 3 spins $s_0, s_1, s_2$.
    - $h_0=4, h_1=2, h_2=-6$.
    - $J_{0,1}=3, J_{0,2}=-1.3, J_{1,2}=2$

    Your function should take as input the values for $s_0,s_1,s_2$ and return the energy.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Ising Models for Combinatorial Optimisation

    Note that the Ising model, where we have spin variables instead of binary variables, gives us another tool for expressing combinatorial optimisation problems, so that minimising the energy of the Ising model yields us the optimal solution.

    We will consider two problems.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Max-Cut

    The Max-Cut problem results in a natural Ising model formulation. Let us recall the definition.

    Given a graph, the problem requires splitting the vertices/nodes into two disjoint groups so that there are as many edges as possible between the groups. The partition of two adjacent vertices into disjoint sets is called a cut. The goal of this problem is to find a cut in such a way that the cut covers the maximum number of edges.

    Like in QUBO formulation, first we will decide what our spin variables represent. For each vertex $i$, we will use a spin variable $s_i$ to decide which group it should belong to:

    $$s_{i}=
    \left\{
    \begin{array}{ll} 
          1, & \text{if vertex $i$ is in Group 1} \\
          -1, & \text{if vertex $i$ is in Group 2} \\
    \end{array}
    \right.$$

    **Our objective is to maximise the number of edges in the cut.**
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Note that for an edge $(i,j)$, $s_is_j=1$ if the vertices are in the same group and $s_is_j=-1$ otherwise. 

    Hence, we can express the exact number of edges in the cut as $ \frac{1}{2} \sum_{(i, j) \in E} (1 - s_is_j)$, which is a maximisation problem. The equivalent minimisation problem is given by:

    $$ \min  \frac{1}{2} \sum_{(i, j) \in E} (s_is_j-1) $$

    Note that in practice, it would be enough to minimise:

    $$\min \sum_{(i,j) \in E} s_is_j$$

    So, the spin configuration minimising the energy of the above problem yields the optimal solution to the max-cut problem.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Number Partitioning

    Given $N$ positive numbers $S = \{n_1, n_2, ... , n_N\}$, consider a partition into two disjoint subsets $S_1$ and $S_2$ such that **the sum of elements in both subsets is the same**.

    This is more a decision problem than an optimisation problem, where we ask the question whether two such subsets exist.

    Still, you can consider this as an optimisation problem, by trying to minimise the difference between the sum of the two subsets. If this difference is 0, this means there is a solution to the problem.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## **Task 2**

    Construct an Ising model for the number partitioning problem.

    > Use $s_i$ to decide the group each number belongs to.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
