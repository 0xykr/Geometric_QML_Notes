import marimo

__generated_with = "0.25.1"
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

    ## Why VQCs?

    VQCs can do ...

    An example of a VQC is ...

    ## What is a Variational Quantum Algorithm (VQA)?

    A VQA is ...

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
