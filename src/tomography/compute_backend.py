"""Use CuPy when it is available and usable, otherwise use NumPy/SciPy."""

try:
    import cupy as cp
    from cupy.linalg import qr
    from cupyx.scipy.fft import dctn, idctn
    from cupyx.scipy.linalg import solve_triangular
    from cupyx.scipy.sparse import csc_matrix
    from cupyx.scipy.sparse._base import spmatrix
    from cupyx.scipy.sparse.linalg import cg

    # Importing CuPy can succeed even when there is no usable CUDA device.
    _probe = cp.zeros(1)
    cp.cuda.get_current_stream().synchronize()
    del _probe
    DEVICE = "gpu"
except Exception:
    cp = None
    DEVICE = "cpu"
