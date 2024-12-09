### This contains stuff for the voxel gradients

import math
from scipy.fft import dctn as sp_dctn
from scipy.fft import idctn as sp_idctn

import numpy as np


from jlinops import _CustomLinearOperator
import jlinops


from jlinops import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular as cp_solve_triangular
    from cupy.linalg import qr as cp_qr
    from cupyx.scipy.sparse.linalg import SuperLU as cp_SuperLU
    from cupyx.scipy.sparse.linalg import cg as cupy_cg





class Voxel2DTemporalGrad(_CustomLinearOperator):
    """Given a stack of Nt images of shape (Nx,Ny), so the input vector x is the result of 
    .flatten() applied to a (Nx,Ny,Nt) array, returns the flattened voxel gradient in the temporal direction.
    """
    def __init__(self, voxel_shape, device="cpu"):
        

        self.Nx, self.Ny, self.Nt = voxel_shape # unpack shapes
        self.voxel_shape = voxel_shape
        n = math.prod(self.voxel_shape)
        shape = (n,n)
        #shape = ?
        

        def _matvec(x):
            x = x.reshape(self.voxel_shape) # reshape it
            result = np.zeros(self.voxel_shape)
            for j in range(self.Nt-1): # we skipped the last slice, since this should be zero...
                result[:,:,j] = x[:,:,j] - x[:,:,j+1]
            return result.flatten()


        def _rmatvec(x):
            x = x.reshape(self.voxel_shape)
            result = np.zeros(self.voxel_shape)
            result[:,:,0] = x[:,:,0]
            for j in range(1,self.Nt-1):
                result[:,:,j] = x[:,:,j] - x[:,:,j-1]
            result[:,:,-1] = -x[:,:,-2]
            return result.flatten()


        super().__init__(shape, _matvec, _rmatvec, device=device)
        
    
    # def to_gpu(self):
    #     return DCT3D(self.grid_shape, device="gpu", type=self.type)
    
    # def to_cpu(self):
    #     return DCT2D(self.grid_shape, device="cpu", type=self.type)




class VoxelNeumann2DSpatial(_CustomLinearOperator):
    """Given a stack of Nt images of shape (Nx,Ny), so the input vector x is the result of 
    .flatten() applied to a (Nx,Ny,Nt) array, returns the flattened voxel gradient in the spatial directions (stacked on top of each other)
    """
    def __init__(self, voxel_shape, device="cpu"):
        

        Nx, Ny, Nt = voxel_shape # unpack shapes
        self.voxel_shape = voxel_shape
        n = math.prod(self.voxel_shape)
        shape = (n,n)
        self.Lspatial = jlinops.Neumann2D((self.Nx, self.Ny))
        shape = ( self.Lspatial.shape[0]*self.Nt, n)    

        def _matvec(x):
            x = x.reshape(self.voxel_shape) # reshape it
            result = np.zeros((2*self.Nx*self.Ny, self.Nt))
            for j in range(self.Nt):
                result[:,j] = self.Lspatial @ x[:,:,j].flatten() 
            return result.flatten()


        def _rmatvec(x):
            result = []
            result = np.zeros((self.Nx, self.Ny, self.Nt))
            tmp = x.reshape((2*self.Nx*self.Ny, self.Nt))
            for j in range(self.Nt):
                result[:,:,j] = (self.Lspatial.T @ tmp[:,j]).reshape((self.Nx, self.Ny))

            return result.flatten()


        super().__init__(shape, _matvec, _rmatvec, device=device)
        
    
    # def to_gpu(self):
    #     return DCT2D(self.grid_shape, device="gpu", type=self.type)
    
    # def to_cpu(self):
    #     return DCT2D(self.grid_shape, device="cpu", type=self.type)






def build_spatiotemporal_neumann2d(voxel_shape):
    """Builds the spatio-temporal . 
    voxel shape should be (Nx, Ny, Nt).
    """
    Nx, Ny, Nt = voxel_shape

    Ls = VoxelNeumann2DSpatial((Nx, Ny, Nt))
    Lt = Voxel2DTemporalGrad((Nx, Ny, Nt))
    L = jlinops.StackedOperator([Ls, Lt])

    return L




# Operator for 3D DCT
class DCT3D(_CustomLinearOperator):
    """Represents a 3-dimensional DCT transform.
    Convention is (Nx, Ny, Nt).
    """
    def __init__(self, voxel_shape, device="cpu", type=2):
        
        # Handle shape
        self.voxel_shape = voxel_shape
        n = math.prod(self.voxel_shape)
        self.type = type
        shape = (n,n)
        
        if device == "cpu":
            
            def _matvec(x):
                return sp_dctn( x.reshape(self.voxel_shape), norm="ortho", type=self.type ).flatten()
            
            def _rmatvec(x):
                return sp_idctn( x.reshape(self.voxel_shape), norm="ortho", type=self.type ).flatten()
            
        else:
            
            def _matvec(x):
                return cp_dctn( x.reshape(self.voxel_shape), norm="ortho", type=self.type ).flatten()
            
            def _rmatvec(x):
                return cp_idctn( x.reshape(self.voxel_shape), norm="ortho", type=self.type ).flatten()
              
        super().__init__(shape, _matvec, _rmatvec, device=device)
        
    
    def to_gpu(self):
        return DCT3D(self.voxel_shape, device="gpu", type=self.type)
    
    def to_cpu(self):
        return DCT3D(self.voxel_shape, device="cpu", type=self.type)






def dct3d_get_eigvals(A, voxel_shape, make_pos=False, type=2):
    """Given an SSPD LinearOperator A that is diagonalized by the 3-dimensional DCT, computes its eigenvalues.
    """
    # Shape of dct
    Nx, Ny, Nt = voxel_shape
    
    device = A.device
    if device == "cpu":
        v = np.random.normal(size=voxel_shape) + 10.0
        #v = np.ones(voxel_shape)
        tmp = A @ ( sp_idctn( v, norm="ortho", type=type ).flatten()  )
        tmp = tmp.reshape(voxel_shape)
        tmp = sp_dctn( tmp, norm="ortho", type=type ).flatten()
        res = tmp/v.flatten()
        if make_pos:
            res = np.abs(res)
        return res
    else:
        v = cp.random.normal(size=voxel_shape) + 10.0
        #v = cp.ones(voxel_shape)
        tmp = A @ ( cp_idctn( v, norm="ortho", type=type ).flatten()  )
        tmp = tmp.reshape(voxel_shape)
        tmp = cp_dctn( tmp, norm="ortho", type=type ).flatten()
        res = tmp/v.flatten()
        if make_pos:
            res = cp.abs(res)
        return res





def dct3d_pinv(A, voxel_shape, eps=1e-14, type=2):
    """Given a LinearOperator A that is diagonalized by the DCT3D, performs the diagonalization (computes eigenvalues), returns a LinearOperator representing A^\dagger (pseudoinverse).
    """
    # Get eigenvalues
    eigvals = dct3d_get_eigvals(A, voxel_shape, type=type)
    device = jlinops.get_device(eigvals)

    # Take reciprocals of nonzero eigenvalues
    if device == "cpu":
        recip_eigvals = np.where( np.abs(eigvals) < eps, eigvals, 1.0 / np.clip(eigvals, a_min=eps, a_max=None) )
        recip_eigvals = np.where( np.abs(eigvals) < eps, np.zeros_like(eigvals), recip_eigvals )
    else:
        recip_eigvals = cp.where( cp.abs(eigvals) < eps, eigvals, 1.0 / cp.clip(eigvals, a_min=eps, a_max=None) )
        recip_eigvals = cp.where( cp.abs(eigvals) < eps, cp.zeros_like(eigvals), recip_eigvals )
    
    # DCT op
    P = DCT3D(voxel_shape, device=device, type=type)
    
    # Apinv op
    Apinv = P.T @ ( jlinops.DiagonalOperator(recip_eigvals) @ P)
    
    return Apinv







class CGWeightedVoxelGradPinvCPUOperator(_CustomLinearOperator):
    """
    """

    def __init__(self, voxel_shape, weights, warmstart_prev=False, check=False, which="scipy", dct_eps=1e-14, *args, **kwargs):

        self.weights = weights
        self.voxel_shape = voxel_shape
        self.warmstart_prev = warmstart_prev
        self.check = check
        self.which = which
        self.dct_eps = dct_eps
        self.args = args
        self.kwargs = kwargs
        
        # Figure out device
        device = jlinops.get_device(weights)

        # Build R and R_w
        self.R = build_spatiotemporal_neumann2d(self.voxel_shape)
        assert self.R.shape[0] == len(weights), "Weights incompatible!"
        self.RtR = self.R.T @ self.R
        self.Dw = jlinops.DiagonalOperator(weights)
        self.Rw = self.Dw @ self.R

        # Get Rpinv (with identity weights)
        self.RtRpinv = dct3d_pinv( self.RtR, self.voxel_shape, eps=dct_eps )

        # Take care of W (columns span the kernel of R)
        if device == "cpu":
            W = np.ones((self.R.shape[1],1))
        else:
            W = cp.ones((self.R.shape[1],1))
            
        self.W = jlinops.MatrixLinearOperator(W)
        self.Wpinv = jlinops.QRPinvOperator(self.W)

        # Make Rwpinv
        self.Rwpinv = jlinops.CGPreconditionedPinvModOperator(self.Rw, self.W, self.Wpinv, self.RtRpinv, warmstart_prev=warmstart_prev, check=check, which=which, *args, **kwargs)

        def _matvec(x):
            return self.Rwpinv.matvec( x )

        def _rmatvec(x):
            return  self.Rwpinv.rmatvec( x ) 

        # def _matvec(x):
        #     return cp.asnumpy( self.Rwpinv.matvec( cp.asarray(x) ) )

        # def _rmatvec(x):
        #     return cp.asnumpy( self.Rwpinv.rmatvec( cp.asarray(x) ) )

        super().__init__( self.Rwpinv.shape, _matvec, _rmatvec, dtype=np.float64, device=device)
        

    def to_gpu(self):
        raise NotImplementedError
        #return CGWeightedNeumann2DPinvOperator(self.grid_shape, cp.asarray(self.weights), warmstart_prev=self.warmstart_prev, check=self.check, which=self.which, dct_eps=self.dct_eps, *self.args, **self.kwargs)
    
    def to_cpu(self):
        raise NotImplementedError
        #return CGWeightedNeumann2DPinvOperator(self.grid_shape, cp.numpy(self.weights), warmstart_prev=self.warmstart_prev, check=self.check, which=self.which, dct_eps=self.dct_eps, *self.args, **self.kwargs)
    





class CGWeightedVoxelGradPinvGPUOperator(_CustomLinearOperator):
    """
    """

    def __init__(self, voxel_shape, weights, warmstart_prev=False, check=False, which="scipy", dct_eps=1e-14, *args, **kwargs):

        self.weights = weights
        self.voxel_shape = voxel_shape
        self.warmstart_prev = warmstart_prev
        self.check = check
        self.which = which
        self.dct_eps = dct_eps
        self.args = args
        self.kwargs = kwargs
        
        # Figure out device
        device = jlinops.get_device(weights)

        # Build R and R_w
        self.R = build_spatiotemporal_neumann2d(self.voxel_shape)
        assert self.R.shape[0] == len(weights), "Weights incompatible!"
        self.RtR = self.R.T @ self.R
        self.Dw = jlinops.DiagonalOperator(weights)
        self.Rw = self.Dw @ self.R

        # Get Rpinv (with identity weights)
        self.RtRpinv = dct3d_pinv( self.RtR, self.voxel_shape, eps=dct_eps )

        # Take care of W (columns span the kernel of R)
        if device == "cpu":
            W = np.ones((self.R.shape[1],1))
        else:
            W = cp.ones((self.R.shape[1],1))
            
        self.W = jlinops.MatrixLinearOperator(W)
        self.Wpinv = jlinops.QRPinvOperator(self.W)

        # Make Rwpinv
        self.Rwpinv = jlinops.CGPreconditionedPinvModOperator(self.Rw, self.W, self.Wpinv, self.RtRpinv, warmstart_prev=warmstart_prev, check=check, which=which, *args, **kwargs).to_gpu()

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
    

