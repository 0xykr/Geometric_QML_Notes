# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Parameter-shift rule: exact gradients from circuit runs

    A variational quantum circuit has knobs $\theta_i$. We measure a cost
    $C(\theta)=\langle 0|U(\theta)^\dagger\, O\, U(\theta)|0\rangle$
    and want $\partial C/\partial\theta_i$ to train it.

    **The rule.** If knob $i$ sits in one gate $e^{-i\theta_i\sigma/2}$ with $\sigma^2=I$, then

    $$\frac{\partial C}{\partial \theta_i}=\frac12\Big[C\big(\theta+\tfrac{\pi}{2}e_i\big)-C\big(\theta-\tfrac{\pi}{2}e_i\big)\Big]$$

    Two circuit runs per knob, no small step size, no approximation.
    Everything below is a live simulation of a 2-qubit circuit. Move the sliders.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell(hide_code=True)
def _(np):
    # ---- a tiny 2-qubit simulator (pure numpy, no quantum library needed) ----
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    CNOT = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    )

    def rot(P, t):
        """Rotation gate exp(-i t P / 2) = cos(t/2) I - i sin(t/2) P."""
        return np.cos(t / 2) * I2 - 1j * np.sin(t / 2) * P

    # Observable O: the "question" we ask. Its lowest eigenvalue is the best possible cost.
    O = np.kron(Z, Z) + 0.5 * np.kron(X, I2)
    ground = float(np.linalg.eigvalsh(O)[0])

    def cost(theta):
        """C(theta) = <0| U^dagger O U |0>. Circuit: RY(t0) RX(t1), CNOT, RY(t2) RY(t3)."""
        t0, t1, t2, t3 = theta
        layer1 = np.kron(rot(Y, t0), rot(X, t1))
        layer2 = np.kron(rot(Y, t2), rot(Y, t3))
        psi = layer2 @ CNOT @ layer1 @ np.array([1, 0, 0, 0], dtype=complex)
        return float(np.real(psi.conj() @ O @ psi))

    def shift_grad(theta):
        """Parameter-shift gradient: 2 circuit runs per knob."""
        theta = np.asarray(theta, dtype=float)
        g = np.zeros(len(theta))
        for i in range(len(theta)):
            e = np.zeros(len(theta))
            e[i] = np.pi / 2
            g[i] = 0.5 * (cost(theta + e) - cost(theta - e))
        return g

    def numeric_grad(theta, h=1e-6):
        """Reference answer: central difference with a tiny step."""
        theta = np.asarray(theta, dtype=float)
        g = np.zeros(len(theta))
        for i in range(len(theta)):
            e = np.zeros(len(theta))
            e[i] = h
            g[i] = (cost(theta + e) - cost(theta - e)) / (2 * h)
        return g

    def naive_grad(theta):
        """Ordinary finite difference but with the same big step pi/2. This is WRONG on purpose."""
        theta = np.asarray(theta, dtype=float)
        g = np.zeros(len(theta))
        c0 = cost(theta)
        for i in range(len(theta)):
            e = np.zeros(len(theta))
            e[i] = np.pi / 2
            g[i] = (cost(theta + e) - c0) / (np.pi / 2)
        return g

    return cost, ground, naive_grad, numeric_grad, shift_grad


@app.cell(hide_code=True)
def _(mo, np):
    s0 = mo.ui.slider(-np.pi, np.pi, step=0.05, value=0.8, label="θ0", full_width=True, show_value=True)
    s1 = mo.ui.slider(-np.pi, np.pi, step=0.05, value=-0.6, label="θ1", full_width=True, show_value=True)
    s2 = mo.ui.slider(-np.pi, np.pi, step=0.05, value=1.1, label="θ2", full_width=True, show_value=True)
    s3 = mo.ui.slider(-np.pi, np.pi, step=0.05, value=0.4, label="θ3", full_width=True, show_value=True)
    mo.vstack(
        [
            mo.md("## 1. Pick a point $\\theta$ and read the gradient"),
            mo.hstack([mo.vstack([s0, s1]), mo.vstack([s2, s3])], widths="equal", gap=2),
        ]
    )
    return s0, s1, s2, s3


@app.cell(hide_code=True)
def _(cost, mo, naive_grad, np, numeric_grad, s0, s1, s2, s3, shift_grad):
    th = np.array([s0.value, s1.value, s2.value, s3.value])
    c_now = cost(th)
    g_shift = shift_grad(th)
    g_true = numeric_grad(th)
    g_naive = naive_grad(th)

    table = "| knob | parameter-shift | tiny-step reference | naive step π/2 |\n|:--|--:|--:|--:|\n"
    for _j in range(4):
        table += (
            f"| θ{_j} | {g_shift[_j]:+.6f} | {g_true[_j]:+.6f} | {g_naive[_j]:+.6f} |\n"
        )
    max_err = float(np.max(np.abs(g_shift - g_true)))
    mo.vstack(
        [
            mo.md(f"**Cost** $C(\\theta)={c_now:+.5f}$"),
            mo.md(table),
            mo.md(
                f"Largest gap between parameter-shift and the tiny-step reference: "
                f"`{max_err:.1e}` (zero up to rounding). "
                "Same step size π/2 with an ordinary finite difference is visibly off."
            ),
        ]
    )
    return (th,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Why it is exact

    Fix every knob except $\theta_i$. Then $C$ is a pure sinusoid in $\theta_i$:
    $C(\theta_i)=a+b\cos\theta_i+d\sin\theta_i$. The two shifted points sit a quarter
    turn either side of $\theta_i$ and their difference isolates exactly the slope.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    which = mo.ui.dropdown(
        options={"θ0": 0, "θ1": 1, "θ2": 2, "θ3": 3}, value="θ0", label="Knob to scan"
    )
    which
    return (which,)


@app.cell(hide_code=True)
def _(cost, np, plt, shift_grad, th, which):
    i = which.value
    xs = np.linspace(-np.pi, np.pi, 400)

    def at(v):
        t = th.copy()
        t[i] = v
        return cost(t)

    curve = np.array([at(v) for v in xs])
    x0 = th[i]
    slope = shift_grad(th)[i]
    c0 = at(x0)
    cp, cm = at(x0 + np.pi / 2), at(x0 - np.pi / 2)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(xs, curve, color="#4c6ef5", lw=2.2, label=f"C as θ{i} varies")
    tx = np.linspace(x0 - 1.0, x0 + 1.0, 10)
    ax.plot(tx, c0 + slope * (tx - x0), color="#e8590c", lw=2, ls="--",
            label="tangent from parameter-shift slope")
    ax.scatter([x0], [c0], color="black", zorder=5, label="current θ")
    ax.scatter([x0 + np.pi / 2, x0 - np.pi / 2], [cp, cm], color="#e8590c", zorder=5,
               s=60, label="the two circuit runs (±π/2)")
    ax.set_xlabel(f"θ{i}")
    ax.set_ylabel("cost C")
    ax.set_xlim(-np.pi, np.pi)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(ground, mo):
    mo.md(
        f"""
    ## 3. Use it: train the circuit by gradient descent

    Each step updates every knob with $\\theta_i\\leftarrow\\theta_i-\\eta\\,\\partial C/\\partial\\theta_i$,
    where the gradient comes only from parameter-shift circuit runs.
    The best possible cost is the lowest eigenvalue of $O$, which is $\\approx {ground:.4f}$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    lr = mo.ui.slider(0.02, 0.6, step=0.02, value=0.2, label="learning rate η", full_width=True, show_value=True)
    steps = mo.ui.slider(5, 100, step=5, value=40, label="steps", full_width=True, show_value=True)
    mo.hstack([lr, steps], widths="equal")
    return lr, steps


@app.cell(hide_code=True)
def _(cost, ground, lr, mo, plt, shift_grad, steps, th):
    t = th.copy()
    history = [cost(t)]
    for _ in range(steps.value):
        t = t - lr.value * shift_grad(t)
        history.append(cost(t))
    runs = steps.value * 2 * len(t)

    fig2, ax2 = plt.subplots(figsize=(8, 3.6))
    ax2.plot(history, color="#4c6ef5", lw=2.2, label="cost during training")
    ax2.axhline(ground, color="#2f9e44", ls="--", label="lowest possible (ground energy)")
    ax2.set_xlabel("step")
    ax2.set_ylabel("cost C")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    fig2.tight_layout()
    mo.vstack(
        [
            mo.md(
                f"Cost went from **{history[0]:+.4f}** to **{history[-1]:+.4f}** "
                f"using {runs} circuit runs in total."
            ),
            fig2,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    - Parameter-shift gives the **exact** gradient from just $2P$ circuit runs for $P$ knobs, so it works on real hardware where you cannot peek inside the state.
    - Cost: $2P$ runs per gradient. Backpropagation on a classical network costs about one extra pass total. Closing that gap is the "backprop analogue" question.
    - Next in our sequence: **quantum natural gradient**, which corrects the plain gradient using the geometry of the state space.
    """)
    return


if __name__ == "__main__":
    app.run()
