"""Weighted Neumann pseudoinverses with automatic device selection."""
import numpy as np
import math
from compute_backend import DEVICE, cp
from jlinops.base import MatrixLinearOperator, _CustomLinearOperator
from jlinops.diagonal import DiagonalOperator
from jlinops.derivatives import Neumann2D
from jlinops.linalg.dct import dct_pinv
from jlinops.util import get_device
from jlinops import QRPinvOperator, CGPreconditionedPinvModOperator


class CGWeightedNeumann2DPinvOperator(_CustomLinearOperator):
    """Represents the pseudoinverse (R_w)^\dagger of a linear operator R_w = D_w R, where
    D_w is a diagonal matrix of weights and R is a Neumann2D operator.
    Here matvecs/rmatvecs are applied approximately using a preconditioned conjugate
    gradient method, where the preconditioner is based on the operator with identity weights.

    All array operations and inner solves use NumPy and SciPy.
    """

    def __init__(self, grid_shape, weights, warmstart_prev=False, check=False, which="scipy", dct_eps=1e-14, *args, **kwargs):

        # An explicit zero absolute tolerance keeps the relative CG criterion consistent.
        kwargs.setdefault("atol", 0.0)
        assert 2*math.prod(grid_shape) == len(weights), "Weights incompatible!"
        self.weights = weights
        self.grid_shape = grid_shape
        self.warmstart_prev = warmstart_prev
        self.check = check
        self.which = which
        self.dct_eps = dct_eps
        self.args = args
        self.kwargs = kwargs

        # Figure out device
        device = get_device(weights)

        # Build R and R_w
        self.R = Neumann2D(grid_shape, device=device)
        self.RtR = self.R.T @ self.R
        self.Dw = DiagonalOperator(weights)
        self.Rw = self.Dw @ self.R

        # Get Rpinv (with identity weights)
        self.RtRpinv = dct_pinv( self.RtR, grid_shape, eps=dct_eps )

        # Take care of W (columns span the kernel of R)
        W = np.ones((self.R.shape[1],1))

        self.W = MatrixLinearOperator(W)
        self.Wpinv = QRPinvOperator(self.W)

        # Make Rwpinv
        self.Rwpinv = CGPreconditionedPinvModOperator(self.Rw, self.W, self.Wpinv, self.RtRpinv, warmstart_prev=warmstart_prev, check=check, which=which, *args, **kwargs)

        def _matvec(x):
            return self.Rwpinv.matvec(x)

        def _rmatvec(x):
            return self.Rwpinv.rmatvec(x)

        super().__init__( self.Rwpinv.shape, _matvec, _rmatvec, dtype=np.float64, device=device)


    def warmstarted_matvec(self, x, initialization=None):

        return self.Rwpinv.warmstarted_matvec(x, initialization)


class CGWeightedNeumann2DPinvOperatorGPU(_CustomLinearOperator):
    """Represents the pseudoinverse (R_w)^\dagger of a linear operator R_w = D_w R, where
    D_w is a diagonal matrix of weights and R is a Neumann2D operator.
    Here matvecs/rmatvecs are applied approximately using a preconditioned conjugate
    gradient method, where the preconditioner is based on the operator with identity weights. 

    The outer interface uses CPU arrays; each inner solve transfers its input
    to the GPU and returns its result to the CPU.
    """

    def __init__(self, grid_shape, weights, warmstart_prev=False, check=False, which="scipy", dct_eps=1e-14, *args, **kwargs):

        assert 2*math.prod(grid_shape) == len(weights), "Weights incompatible!"
        self.weights = weights
        self.grid_shape = grid_shape
        self.warmstart_prev = warmstart_prev
        self.check = check
        self.which = which
        self.dct_eps = dct_eps
        self.args = args
        self.kwargs = kwargs
        
        # Figure out device
        device = get_device(weights)

        # Build R and R_w
        self.R = Neumann2D(grid_shape, device=device)
        self.RtR = self.R.T @ self.R
        self.Dw = DiagonalOperator(weights)
        self.Rw = self.Dw @ self.R

        # Get Rpinv (with identity weights)
        self.RtRpinv = dct_pinv( self.RtR, grid_shape, eps=dct_eps )

        # Take care of W (columns span the kernel of R)
        if device == "cpu":
            W = np.ones((self.R.shape[1],1))
        else:
            W = cp.ones((self.R.shape[1],1))
            
        self.W = MatrixLinearOperator(W)
        self.Wpinv = QRPinvOperator(self.W)

        # Make Rwpinv
        self.Rwpinv = CGPreconditionedPinvModOperator(self.Rw, self.W, self.Wpinv, self.RtRpinv, warmstart_prev=warmstart_prev, check=check, which=which, *args, **kwargs).to_gpu()

        def _matvec(x):
            return cp.asnumpy( self.Rwpinv.matvec( cp.asarray(x) ) )

        def _rmatvec(x):
            return cp.asnumpy( self.Rwpinv.rmatvec( cp.asarray(x) ) )

        super().__init__( self.Rwpinv.shape, _matvec, _rmatvec, dtype=np.float64, device=device)
        

    def warmstarted_matvec(self, x, initialization=None):

        if initialization is None:

            sol, count = self.Rwpinv.warmstarted_matvec( cp.asarray(x), initialization )
            
            return cp.asnumpy(sol), count
            
        else:

            sol, count = self.Rwpinv.warmstarted_matvec( cp.asarray(x), cp.asarray(initialization) ) 

            return cp.asnumpy(sol), count


    def to_gpu(self):
        raise NotImplementedError
        #return CGWeightedNeumann2DPinvOperator(self.grid_shape, cp.asarray(self.weights), warmstart_prev=self.warmstart_prev, check=self.check, which=self.which, dct_eps=self.dct_eps, *self.args, **self.kwargs)
    
    def to_cpu(self):
        raise NotImplementedError


SelectedWeightedNeumann2DPinvOperator = (
    CGWeightedNeumann2DPinvOperatorGPU
    if DEVICE == "gpu"
    else CGWeightedNeumann2DPinvOperator
)
