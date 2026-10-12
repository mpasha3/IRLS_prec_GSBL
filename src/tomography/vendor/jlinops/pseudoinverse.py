# QR and preconditioned conjugate-gradient pseudoinverse operators.
import numpy as np
from scipy.linalg import qr as sp_qr
from scipy.linalg import solve_triangular as sp_solve_triangular
from scipy.linalg import solve_triangular as scipy_solve_triangular
from scipy.sparse.linalg import cg as sp_cg
from .base import MatrixLinearOperator, _CustomLinearOperator
from . import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular as cp_solve_triangular
    from cupy.linalg import qr as cp_qr
    from cupyx.scipy.sparse.linalg import cg as cupy_cg


class QRPinvOperator(_CustomLinearOperator):
    """Takes a dense matrix A with full column rank, builds a linear operator representing the pseudo-inverse of A
    using the QR method.
    """

    def __init__(self, A):

        #assert isinstance(A, MatrixLinearOperator), "must give MatrixOperator as an input."

        if not isinstance(A, MatrixLinearOperator):
            A = MatrixLinearOperator(A)
        
        # Store original operator
        self.original_op = A
        k, n = A.shape
        
        # Device
        device = A.device
        
        if device == "cpu":
            
            Q_fac, R_fac = sp_qr(A.A, mode="economic")

            # Build matvec and rmatvec
            def _matvec(vec):
                tmp = Q_fac.T @ vec
                tmp = sp_solve_triangular(R_fac, tmp, lower=False)
                return tmp

            def _rmatvec(vec):
                tmp = scipy_solve_triangular(R_fac.T, vec, lower=True)
                tmp = Q_fac @ tmp
                return tmp
            
        else:
            
            Q_fac, R_fac = cp_qr(A.A, mode="reduced")

            # Build matvec and rmatvec
            def _matvec(vec):
                tmp = Q_fac.T @ vec
                tmp = cp_solve_triangular(R_fac, tmp, lower=False)
                return tmp

            def _rmatvec(vec):
                tmp = cp_solve_triangular(R_fac.T, vec, lower=True)
                tmp = Q_fac @ tmp
                return tmp

        super().__init__( (n, k), _matvec, _rmatvec , device=device)
        
        
    def to_gpu(self):
        return QRPinvOperator(self.original_op.to_gpu())
    
    def to_cpu(self):
        return QRPinvOperator(self.original_op.to_cpu())


class CGPreconditionedPinvModOperator(_CustomLinearOperator):
    """Returns a linear operator that approximately computes the pseudoinverse of a matrix A using 
    a conjugate gradient method. Modifed so that it only ever solves systems with A^T A. 
    
    W: a LinearOperator representing a matrix with linearly independent columns that spans null(A).
    Wpinv: a LinearOperator represening the pseudoinverse of W.
    Lpinv: 
    """

    def __init__(self, A, W, Wpinv, Mpinv, warmstart_prev=False, check=False, which="scipy", *args, **kwargs):

        assert which in ["jlinops", "scipy"], "Invalid choice for which!"

        # Device
        device = A.device

        self.args = args
        self.kwargs = kwargs
        
        # Store operator
        self.A = A
        self.W = W
        self.Wpinv = Wpinv
        self.Mpinv = Mpinv
        self.args = args
        self.kwargs = kwargs
        
        # Shape
        m, n = A.shape
        shape = (n, m)

        # Setup
        self.which = which
        self.check = check
        self.warmstart_prev = warmstart_prev
        self.in_shape = self.A.shape[0]
        self.out_shape = self.A.shape[1]
        
        if device == "cpu":
            self.prev_eval = np.zeros(self.out_shape)
            self.prev_eval_t = np.zeros(self.out_shape)
        else:
            self.prev_eval = cp.zeros(self.out_shape)
            self.prev_eval_t = cp.zeros(self.out_shape)

        # Build both operators we need
        self.C = self.A.T @ self.A
        self.MpinvC = self.Mpinv @ self.C

        
        if device == "cpu":
            
           
            if self.which == "scipy":
                
                def _matvec(x):
                    sol, converged = sp_cg(self.C, self.A.rmatvec(x), x0=self.prev_eval, M=self.Mpinv, *args, **kwargs) 
                    if self.check:
                        assert converged == 0, "CG algorithm did not converge!"
                    
                    if self.warmstart_prev:
                        self.prev_eval = sol.copy()
                    
                    return sol
                
                def _rmatvec(x):
                    
                    # Project x onto range(A^T A) = range(A^T).
                    if (self.W is not None) and (self.Wpinv is not None):
                        z = x - (self.W @ (self.Wpinv @ x))
                    else:
                        z = x 
                    
                    sol, converged = sp_cg(self.C, z, x0=self.prev_eval_t, M=self.Mpinv, *args, **kwargs) 
                    if self.check:
                        assert converged == 0, "CG algorithm did not converge!"
                    
                    if self.warmstart_prev:
                        self.prev_eval_t = sol.copy()
                    
                    return self.A @ sol
                
            else:
                raise NotImplementedError
                
        else:
            
            if self.which == "scipy":
                
                def _matvec(x):
                    sol, converged = cupy_cg(self.C, self.A.rmatvec(x), M=self.Mpinv, x0=self.prev_eval, *args, **kwargs)
                    if self.check:
                        assert converged == 0, "CG algorithm did not converge!"
                    
                    if self.warmstart_prev:
                        self.prev_eval = sol.copy()
                        
                    return sol
                
                def _rmatvec(x):

                    # Project x onto range(A^T A) = range(A^T).
                    if (self.W is not None) and (self.Wpinv is not None):
                        z = x - (self.W @ (self.Wpinv @ x))
                    else:
                        z = x
                    
                    sol, converged = cupy_cg(self.C, z, x0=self.prev_eval_t, M=self.Mpinv, *args, **kwargs) 
                    if self.check:
                        assert converged == 0, "CG algorithm did not converge!"
                    
                    if self.warmstart_prev:
                        self.prev_eval_t = sol.copy()
                        
                    return self.A @ sol
                
            else:
                raise NotImplementedError
        
        
        super().__init__( shape, _matvec, _rmatvec, dtype=np.float64, device=device)
        

    def warmstarted_matvec(self, x, initialization):

        class IterationCounter:
            def __init__(self):
                self.count = 0
            def __call__(self, xk):
                self.count += 1


        if self.device == "cpu":
                
            def _matvec(x):
                
                counter = IterationCounter()
                
                sol, converged = sp_cg(self.C, self.A.rmatvec(x), x0=initialization, M=self.Mpinv, callback=counter, *self.args, **self.kwargs) 
                if self.check:
                    assert converged == 0, "CG algorithm did not converge!"
                
                if self.warmstart_prev:
                    self.prev_eval = sol.copy()
                
                return sol, counter.count
            
            return _matvec(x)
        
        else:

            def _matvec(x):
                
                counter = IterationCounter()

                sol, converged = cupy_cg(self.C, self.A.rmatvec(x), M=self.Mpinv, x0=initialization, callback=counter, *self.args, **self.kwargs)
                if self.check:
                    assert converged == 0, "CG algorithm did not converge!"
                
                if self.warmstart_prev:
                    self.prev_eval = sol.copy()
                    
                return sol, counter.count
             
            return _matvec(x)

        
    def to_gpu(self):
        return CGPreconditionedPinvModOperator(self.A.to_gpu(), self.W.to_gpu(), self.Wpinv.to_gpu(), self.Mpinv.to_gpu(), warmstart_prev=self.warmstart_prev, which=self.which, check=self.check, *self.args, **self.kwargs)
    
        
    def to_cpu(self):
        return CGPreconditionedPinvModOperator(self.A.to_cpu(), self.W.to_cpu(), self.Wpinv.to_cpu(), self.Mpinv.to_cpu(), warmstart_prev=self.warmstart_prev, which=self.which, check=self.check, *self.args, **self.kwargs)
