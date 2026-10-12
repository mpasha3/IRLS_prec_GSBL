"""CPU linear operators for the one-dimensional notebooks.

NumPy and SciPy implement matrix products and operator arithmetic.
The wrappers provide the CPU attributes used by the solvers.
"""
import numpy as np
import scipy.sparse as sparse
from scipy.linalg import qr, solve_triangular
from scipy.sparse.linalg import LinearOperator as _SciPyLinearOperator


def _cpu_result(value):
    """Keep CPU metadata on composed SciPy operators without changing products."""
    if isinstance(value, _SciPyLinearOperator) and not isinstance(value, LinearOperator):
        return _CustomLinearOperator(value.shape, value.matvec, value.rmatvec,
                                     value.matmat, value.dtype, value.rmatmat)
    return value


class LinearOperator(_SciPyLinearOperator):
    """SciPy's linear operator with the CPU attributes expected by the solvers."""
    device = "cpu"

    def to_cpu(self):
        return self

    def __mul__(self, other):
        return _cpu_result(super().__mul__(other))

    def __rmul__(self, other):
        return _cpu_result(super().__rmul__(other))

    def __add__(self, other):
        return _cpu_result(super().__add__(other))

    def _transpose(self):
        return _cpu_result(super()._transpose())


class _CustomLinearOperator(LinearOperator):
    """Linear operator defined by forward and adjoint callbacks."""

    def __init__(self, shape, matvec, rmatvec=None, matmat=None, dtype=None,
                 rmatmat=None, device="cpu"):
        if device != "cpu":
            raise ValueError("The reproduction operator layer supports CPU arrays only")
        self._forward = matvec
        self._backward = rmatvec
        self._forward_block = matmat
        self._backward_block = rmatmat
        super().__init__(dtype, shape)
        self._init_dtype()

    def _matvec(self, x):
        return self._forward(x)

    def _rmatvec(self, x):
        if self._backward is None:
            raise NotImplementedError("No adjoint callback was supplied")
        return self._backward(x)

    def _matmat(self, x):
        if self._forward_block is not None:
            return self._forward_block(x)
        return super()._matmat(x)

    def _rmatmat(self, x):
        if self._backward_block is not None:
            return self._backward_block(x)
        return super()._rmatmat(x)

    def _adjoint(self):
        return _CustomLinearOperator(self.shape[::-1], self._backward, self._forward,
                                     self._backward_block, self.dtype, self._forward_block)


class MatrixLinearOperator(LinearOperator):
    """Dense or sparse matrix operator retaining its underlying matrix as ``A``."""

    def __init__(self, matrix):
        if not sparse.issparse(matrix):
            matrix = np.asarray(matrix)
            if matrix.ndim == 1:
                matrix = matrix[:, None]
        self.A = matrix
        super().__init__(matrix.dtype, matrix.shape)

    def _matmat(self, x):
        return self.A.dot(x)

    def _adjoint(self):
        return MatrixLinearOperator(self.A.T.conj())


class IdentityOperator(_CustomLinearOperator):
    def __init__(self, shape, dtype=None, device="cpu"):
        if shape[0] != shape[1]:
            raise ValueError("Identity operator must be square")
        super().__init__(shape, lambda x: x, lambda x: x, lambda x: x,
                         np.float64 if dtype is None else dtype, lambda x: x, device)


class DiagonalOperator(MatrixLinearOperator):
    def __init__(self, diagonal):
        self.diagonal = np.asarray(diagonal)
        if self.diagonal.ndim != 1:
            raise ValueError("Diagonal must be one dimensional")
        super().__init__(sparse.diags(self.diagonal))


class Subsampling1DOperator(_CustomLinearOperator):
    """Select mask entries equal to one; the adjoint inserts zeros elsewhere."""

    def __init__(self, mask):
        mask = np.asarray(mask)
        if mask.ndim != 1 or not np.all((mask == 0) | (mask == 1)):
            raise ValueError("Subsampling mask must be a binary vector")
        self.mask = mask.astype(np.int32)
        self.idxs_observed = np.where(self.mask == 1)
        n = len(mask)

        def backward(x):
            result = np.zeros(n)
            result[self.idxs_observed] = np.squeeze(x)
            return result

        super().__init__((int(mask.sum()), n), lambda x: x[self.idxs_observed], backward)


class QRPinvOperator(_CustomLinearOperator):
    """QR pseudoinverse of a dense, full-column-rank matrix."""

    def __init__(self, matrix):
        if not isinstance(matrix, MatrixLinearOperator):
            matrix = MatrixLinearOperator(matrix)
        self.original_op = matrix
        q, r = qr(matrix.A, mode="economic")
        super().__init__(matrix.shape[::-1],
                         lambda x: solve_triangular(r, q.T @ x, lower=False),
                         lambda x: q @ solve_triangular(r.T, x, lower=True))


def first_order_derivative_1d(n, boundary="none"):
    """First-difference matrix and a basis for its kernel.

    All notebooks use ``boundary='zero'``: diagonal +1, superdiagonal -1,
    and a final row containing +1. Consequently its inverse is the upper
    triangular all-ones matrix.
    """
    matrix = sparse.eye(n)
    matrix.setdiag(-1, k=1)
    matrix = matrix.tolil()
    kernel = None
    if boundary == "zero":
        pass
    elif boundary == "none":
        matrix = matrix[:-1, :]
        kernel = np.ones((n, 1))
    elif boundary == "periodic":
        matrix[-1, 0] = -1
        kernel = np.ones((n, 1))
    elif boundary == "reflexive":
        matrix[-1, -1] = 0
        kernel = np.ones((n, 1))
    elif boundary == "zero_sym":
        matrix = sparse.vstack([sparse.csc_matrix((1, n)), matrix.tocsc()]).tolil()
        matrix[0, 0] = -1
    else:
        raise ValueError(f"Unknown first-difference boundary: {boundary}")
    return matrix, kernel


def black_box_to_dense(operator):
    """Materialize a small operator with one forward call per matrix column."""
    n = operator.shape[1]
    return np.column_stack([operator.matvec(np.eye(1, n, index).ravel())
                            for index in range(n)])


def issparse(value):
    return sparse.issparse(value)


def get_device(value):
    return "cpu"
