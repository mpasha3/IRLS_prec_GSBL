"""Synthetic data, plot styling, and result tables for the 1D notebooks."""
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import time
import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.fft import dct, idct
import operators


class DCT1D(operators._CustomLinearOperator):
    """Orthonormal type-II DCT."""
    def __init__(self, n):
        super().__init__((n, n),
                         lambda x: dct(np.squeeze(x), type=2, norm="ortho"),
                         lambda x: idct(np.squeeze(x), type=2, norm="ortho"),
                         device="cpu", dtype=None)


class FirstDifferenceInverse(operators._CustomLinearOperator):
    """Exact inverse of the zero-boundary first difference, applied by sums.

    L has +1 on its diagonal and -1 on the superdiagonal. Hence Linv is the
    upper-triangular all-ones matrix. This avoids dense O(n^2) products without
    changing the linear map; floating-point summation order can differ.
    """
    def __init__(self, n):
        def forward(x):
            return np.cumsum(np.asarray(x)[::-1], axis=0)[::-1]
        def adjoint(x):
            return np.cumsum(np.asarray(x), axis=0)
        super().__init__((n, n), forward, adjoint, matmat=forward,
                         rmatmat=adjoint, dtype=np.dtype(float), device="cpu")


def make_problem():
    """Generate the n=1000, m=50, seed=0, 3% noise problem."""
    n = 1000
    ground_truth = np.empty(n)
    ground_truth[:250] = 2.5
    ground_truth[250:400] = -0.5
    ground_truth[400:625] = 4.5
    ground_truth[625:825] = 2.5
    ground_truth[825:] = 0.5
    mask = np.zeros(n)
    mask[:50] = 1
    A = operators.Subsampling1DOperator(mask) @ DCT1D(n)
    b_true = A @ ground_truth
    np.random.seed(0)
    noise = np.random.normal(size=A.shape[0])
    noise /= np.linalg.norm(noise)
    noise *= 0.03 * np.linalg.norm(b_true)
    noise_var = (0.03 * np.linalg.norm(b_true))**2 / A.shape[0]
    b = b_true + noise
    Lmat, _ = operators.first_order_derivative_1d(n, boundary="zero")
    L = operators.MatrixLinearOperator(Lmat)
    Linv = FirstDifferenceInverse(n)
    def Lpinv_factory(wr):
        return Linv @ operators.MatrixLinearOperator(sps.diags(1.0 / wr))
    return A, L, Linv, Lpinv_factory, b, ground_truth, noise_var


def set_style():
    """Set the figure color cycle."""
    import matplotlib.pyplot as plt
    palette = [(106,192,205),(235,186,63),(172,46,55),(32,82,145),
               (83,165,106),(239,145,75),(208,92,89),(80,67,157),
               (82,82,82),(53,53,53),(30,27,28)]
    plt.rc("axes", prop_cycle=plt.cycler(color=[tuple(x/255 for x in rgb) for rgb in palette]))


def logged_solve(log_path, label, solve, **kwargs):
    """Write solver diagnostics to a file and print concise progress."""
    start = time.perf_counter()
    print(f"Running {label} ...", flush=True)
    with Path(log_path).open("a") as stream, redirect_stdout(stream), redirect_stderr(stream):
        print(f"\n--- {label} ---")
        result = solve(**kwargs)
    duration = time.perf_counter() - start
    print(f"{label}: {result['n_completed_iterations']} iterations, "
          f"RRE={result['rre_history'][-1]:.6g}, {duration:.1f} seconds", flush=True)
    return result


def summarize(results):
    """Summarize final metrics and matrix-vector product counts."""
    rows = []
    for (weights, method), data in results.items():
        rows.append({
            "Weights": weights, "Method": method,
            "mu": float(data["lambda_history"][-1]),
            "n_iter": int(data["n_completed_iterations"]),
            "RRE": float(data["rre_history"][-1]),
            "SSIM": float(data["ssim_history"][-1]),
            "Gini index": float(data["gini_indices"][-1]),
            "kappa": float(data["cond_ests"][-1]),
            "n_A": int(data["tot_matvec_A"]),
            "n_Psi": int(data["tot_matvec_L"]),
            "n_Psi_inv": int(data.get("tot_matvec_WLpinv", data.get("tot_matvec_Linv", 0))),
        })
    return pd.DataFrame(rows)
