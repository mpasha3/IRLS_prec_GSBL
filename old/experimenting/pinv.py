import numpy as np
import math

from scipy.linalg import solve_triangular as scipy_solve_triangular
from scipy.linalg import qr as sp_qr
from scipy.linalg import solve_triangular as sp_solve_triangular
from scipy.sparse.linalg import SuperLU as sp_SuperLU
from scipy.sparse.linalg import cg as sp_cg
import scipy.sparse as sps

from scipy.fft import dct as sp_dct
from scipy.fft import idct as sp_idct


from jlinops.base import MatrixLinearOperator, _CustomLinearOperator, IdentityOperator
from jlinops.linalg.factorizations import banded_cholesky
from jlinops.diagonal import DiagonalOperator
from jlinops.derivatives import Neumann2D, Dirichlet2DSym
# from .cginv import CGDirichlet2DSymLapInvOperator, CGWeightedDirichlet2DSymLapInvOperator
from jlinops.linalg.dct import dct_sqrt_pinv, dct_pinv, DCT
from jlinops.linalg.dst import dst_sqrt_pinv, dst_pinv
from jlinops.linear_solvers import cg
from jlinops.util import issparse, tosparse, get_device
from jlinops.linalg import dct_sqrt, dct_sqrt_pinv


from jlinops.derivatives import get_neumann2d_laplacian_diagonal, get_neumann2d_laplacian_tridiagonal, first_order_derivative_1d
from jlinops.inv import TridiagInvOperator



from jlinops import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular as cp_solve_triangular
    from cupy.linalg import qr as cp_qr
    from cupyx.scipy.sparse.linalg import SuperLU as cp_SuperLU
    from cupyx.scipy.sparse.linalg import cg as cupy_cg

from jlinops import QRPinvOperator
from jlinops import CGPreconditionedPinvModOperator


class CGWeightedNeumann2DPinvOperatorGPU(_CustomLinearOperator):
    """Represents the pseudoinverse (R_w)^\dagger of a linear operator R_w = D_w R, where
    D_w is a diagonal matrix of weights and R is a Neumann2D operator.
    Here matvecs/rmatvecs are applied approximately using a preconditioned conjugate
    gradient method, where the preconditioner is based on the operator with identity weights. 

    The GPU operator does not work. This is a "fake" GPU operator, 
    meaning that the arrays are brought onto/back to the cpu immediately 
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
        

    def to_gpu(self):
        raise NotImplementedError
        #return CGWeightedNeumann2DPinvOperator(self.grid_shape, cp.asarray(self.weights), warmstart_prev=self.warmstart_prev, check=self.check, which=self.which, dct_eps=self.dct_eps, *self.args, **self.kwargs)
    
    def to_cpu(self):
        raise NotImplementedError
        #return CGWeightedNeumann2DPinvOperator(self.grid_shape, cp.numpy(self.weights), warmstart_prev=self.warmstart_prev, check=self.check, which=self.which, dct_eps=self.dct_eps, *self.args, **self.kwargs)
    













# class CGPreconditionedPinvModOperator(_CustomLinearOperator):
#     """Returns a linear operator that approximately computes the pseudoinverse of a matrix A using 
#     a conjugate gradient method. Modifed so that it only ever solves systems with A^T A. 
    
#     W: a LinearOperator representing a matrix with linearly independent columns that spans null(A).
#     Wpinv: a LinearOperator represening the pseudoinverse of W.
#     Lpinv: 
#     """

#     def __init__(self, A, W, Wpinv, Mpinv, warmstart_prev=False, check=False, which="scipy", *args, **kwargs):

#         assert which in ["jlinops", "scipy"], "Invalid choice for which!"

#         # Device
#         device = A.device
        
#         # Store operator
#         self.A = A
#         self.W = W
#         self.Wpinv = Wpinv
#         self.Mpinv = Mpinv
        
#         # Shape
#         m, n = A.shape
#         shape = (n, m)

#         # Setup
#         self.which = which
#         self.check = check
#         self.warmstart_prev = warmstart_prev
#         self.in_shape = self.A.shape[0]
#         self.out_shape = self.A.shape[1]
        
#         if device == "cpu":
#             self.prev_eval = np.zeros(self.out_shape)
#             self.prev_eval_t = np.zeros(self.out_shape)
#         else:
#             self.prev_eval = cp.zeros(self.out_shape)
#             self.prev_eval_t = cp.zeros(self.out_shape)

#         # Build both operators we need
#         self.C = self.A.T @ self.A
#         self.MpinvC = self.Mpinv @ self.C

        
#         if device == "cpu":
            
           
#             if self.which == "scipy":
                
#                 def _matvec(x):
#                     sol, converged = sp_cg(self.C, self.A.rmatvec(x), x0=self.prev_eval, M=self.Mpinv, *args, **kwargs) 
#                     if self.check:
#                         assert converged == 0, "CG algorithm did not converge!"
                    
#                     if self.warmstart_prev:
#                         self.prev_eval = sol.copy()
                    
#                     return sol
                
#                 def _rmatvec(x):
                    
#                     # Project x onto range(A^T A) = range(A^T).
#                     if (self.W is not None) and (self.Wpinv is not None):
#                         z = x - (self.W @ (self.Wpinv @ x))
#                     else:
#                         z = x 
                    
#                     sol, converged = sp_cg(self.C, z, x0=self.prev_eval_t, M=self.Mpinv, *args, **kwargs) 
#                     if self.check:
#                         assert converged == 0, "CG algorithm did not converge!"
                    
#                     if self.warmstart_prev:
#                         self.prev_eval_t = sol.copy()
                    
#                     return self.A @ sol
                
#             else:
#                 raise NotImplementedError
                
#         else:
            
#             if self.which == "scipy":
                
#                 def _matvec(x):
#                     sol, converged = cupy_cg(self.C, self.A.rmatvec(x), M=self.Mpinv, x0=self.prev_eval, *args, **kwargs)
#                     if self.check:
#                         assert converged == 0, "CG algorithm did not converge!"
                    
#                     if self.warmstart_prev:
#                         self.prev_eval = sol.copy()
                        
#                     return sol
                
#                 def _rmatvec(x):

#                     # Project x onto range(A^T A) = range(A^T).
#                     if (self.W is not None) and (self.Wpinv is not None):
#                         z = x - (self.W @ (self.Wpinv @ x))
#                     else:
#                         z = x
                    
#                     sol, converged = cupy_cg(self.C, z, x0=self.prev_eval_t, M=self.Mpinv, *args, **kwargs) 
#                     if self.check:
#                         assert converged == 0, "CG algorithm did not converge!"
                    
#                     if self.warmstart_prev:
#                         self.prev_eval_t = sol.copy()
                        
#                     return self.A @ sol
                
#             else:
#                 raise NotImplementedError
        
        
#         super().__init__( shape, _matvec, _rmatvec, dtype=np.float64, device=device)
        
        
#     def to_gpu(self):
#         return CGPreconditionedPinvModOperator(self.A.to_gpu(), self.W.to_gpu(), self.Wpinv.to_gpu(), self.Mpinv.to_gpu(), warmstart_prev=self.warmstart_prev, which=self.which, check=self.check, *self.args, **self.kwargs)
    
        
#     def to_cpu(self):
#         return CGPreconditionedPinvModOperator(self.A.to_cpu(), self.W.to_cpu(), self.Wpinv.to_cpu(), self.Mpinv.to_cpu(), warmstart_prev=self.warmstart_prev, which=self.which, check=self.check, *self.args, **self.kwargs)
    
    