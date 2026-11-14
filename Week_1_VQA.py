import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Variational Quantum Algorithms

    ## What is a Variational Quantum Circuit (VQC)?

    Variational quantum circuits are quantum circuits whose gates depend on adjustable parameters $\bm{\theta} = (\theta_1, \theta_2, \dots, \theta_n)$. We think of these parameters $\bm{\theta}$ as trainable weights in an ML model.

    ## What is a Variational Quantum Algorithm (VQA)?

    A VQA is an algorithm that runs on a VQC that tries to minimize some cost function by using a QC to evaluate cost function and then performing classical calculations to update our parameters theta.

    ## Why VQAs?

    Oftentimes, to solve a problem, we want to minimize something. (add examples here).
    For any state $\ket{\Phi}$ we define
    $$E(\Phi)=\frac{\braket{\Phi | H | \Phi}}{\braket{\Phi|\Phi}} $$
    This is the Raleigh Quotient and has property: $E(\Phi) \ge E_0$ for all $\ket{\Phi}$ in Hilbert space.\\ \\
    This converts an \textbf{eigenvalue problem} (hard, requires diagonalizing a $2^N \times 2^N$ matrix) into an \textbf{optimization problem} (i.e. search over some family of trial states $\ket{\Phi(\bm{\theta})}$ for the one that minimizes $E(\Phi(\bm{\theta}))$.

    ## Parts of a VQA

    ### Cost Function

    A cost function is ...

    ### Gradient Descent

    Gradient descent is ...

    ### Ansatz

    An ansatz is ...
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # VQA Example (QAOA)
    """)
    return


@app.cell
def _():
    # INSERT IMPLEMENTATION HERE (with comments / important lines marked)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Talk about important lines of code, etc, make sure to clearly link the contents of the code to the explanations above
    """)
    return


if __name__ == "__main__":
    app.run()
