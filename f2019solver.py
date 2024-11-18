import numpy as np
import matplotlib.pyplot as plt

# Add files up one level to the path
import sys
sys.path.append("..")

import jlinops

from matplotlib.backends.backend_pdf import PdfPages

from scipy.linalg import qr, qr_insert
from scipy.integrate import odeint
import scipy.optimize as op
import scipy.sparse as sps
from scipy import linalg as la
from scipy.optimize import root


from fastprogress import progress_bar

import warnings

import scipy.optimize as op
from scipy.sparse.linalg import svds
from scipy.linalg import null_space

import scipy.sparse as sps
from scipy.fft import dct, idct

from scipy.optimize import brentq


from skimage.metrics import structural_similarity


from haarpsi import haar_psi




# Try to import matlab.engine
try:
    import matlab.engine
except:
    pass # assume we don't need it
    # raise Exception("Error with matlab.engine, make sure it is installed correctly.")


# from trips.utilities.weights import smoothed_holder_weights
from trips.utilities.decompositions import golub_kahan

from trips.utilities.reg_param.discrepancy_principle import discrepancy_principle
from trips.utilities.reg_param.gcv import generalized_crossvalidation

from regparam_selection import mgcv, dp_cubic, matlab_gcv
from regparam_selection import optimal_regparam, optimal_regparam_ssim




# Define smoothed holder weighting function
def mm_weights(x, epsilon, p):
    z = (x**2 + epsilon**2)**( (p-2.0)/2.0 )
    return z




def ksparsity_index(vector, k):
    # Make a copy of the original vector
    result = np.zeros_like(vector)
    
    # Find the indices of the k largest entries in absolute value
    largest_indices = np.argpartition(np.abs(vector), -k)[-k:]
    
    # Set the k largest entries in the result
    result[largest_indices] = vector[largest_indices]

    # Evaluate the norm from original vector
    return np.linalg.norm(result - vector)



def hoyer_sparsity(x):
    """
    Computes the Hoyer sparsity measure for a given vector x.

    Parameters:
    x (numpy array): Input vector.

    Returns:
    float: Hoyer sparsity measure, between 0 and 1.
    """
    n = len(x)
    norm_1 = np.linalg.norm(x, 1)  # L1 norm of x
    norm_2 = np.linalg.norm(x, 2)  # L2 norm of x

    if norm_2 == 0:
        return 0  # A zero vector is fully sparse, returning 0.
    
    sparsity = (np.sqrt(n) - (norm_1 / norm_2)) / (np.sqrt(n) - 1)
    return sparsity


def gini_index(x):
    """
    Computes the Gini index for a given vector x.

    Parameters:
    x (numpy array): Input vector.

    Returns:
    float: Gini index, between 0 and 1.
    """
    if np.all(x == 0):
        return 0.0  # Gini index for a zero vector is 0 (least sparse)

    x = np.sort(np.abs(x))  # Sort the absolute values of the vector
    n = len(x)
    index = np.arange(1, n + 1)  # Index for each element in the sorted array

    # Gini formula
    gini = (2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n
    return gini







def to_upper_hessenberg(matrix):
    """
    Takes a matrix and returns its upper Hessenberg form.
    Elements below the first subdiagonal are set to zero.
    
    Parameters:
    matrix (np.ndarray): Input matrix (assumed to be 2D).
    
    Returns:
    np.ndarray: Upper Hessenberg matrix.
    """
    # Get the number of rows and columns
    rows, cols = matrix.shape
    
    # Make a copy of the matrix to zero out below the first subdiagonal
    hessenberg_matrix = np.copy(matrix)
    
    # Use NumPy's advanced indexing to zero out elements below the first subdiagonal
    i, j = np.tril_indices(rows, -2)
    hessenberg_matrix[i, j] = 0
    
    return hessenberg_matrix





def update_fgk(A, Z, U, M, V, T, L, breakdown_tol=1e-14):
    """Updates a FGK factorization by one step."""

    broke_down = False

    k = U.shape[1]
    v = A.T @ U[:,-1]
    assert k > 0, "invalid U"
    if k == 1:
        w = A.T @ U[:,-1]
        T = np.zeros((1,1))
        M = np.zeros((2,1))
    else:
        T = np.pad(T, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        M = np.pad(M, ((0, 1), (0, 1)), mode='constant', constant_values=0)

        for j in range(k-1):
            T[j,k-1] = np.dot(V[:,j], v)
            v = v - T[j,k-1]*V[:,j]
            
    T[k-1,k-1] = np.linalg.norm(v)
    # Check for breakdown
    if np.abs(T[k-1,k-1]) < breakdown_tol:
        broke_down = True
        return U, M[:-1,:-1], T[:-1,:-1], V, Z, broke_down

    v = v/T[k-1,k-1]

    z = L @ v
    u = A @ z
    for j in range(k):
        M[j,k-1] = np.dot(U[:,j], u)
        u -= M[j,k-1]*U[:,j]

    M[k,k-1] = np.linalg.norm(u)

    # Check for breakdown
    if np.abs(M[k,k-1]) < breakdown_tol:
        broke_down = True
        return U, M[:-1,:-1], T[:-1,:-1], V, Z, broke_down

    u = u/M[k,k-1]

    if k == 1:
        V = v[:,None]
        Z = z[:,None]
    else:
        vnew = v - (V @ (V.T @ v))
        V = np.column_stack((V,vnew))
        #V = np.column_stack((V,v))

        Z = np.column_stack((Z,z))

    unew = u - (U @ (U.T @ u))
    U = np.column_stack((U,unew))
    #U = np.column_stack((U,u))

    return U, M, T, V, Z, broke_down







class FGK2019Solver:
    """Implements a flexible Golub-Kahan solver.
    """

    def __init__(self, A, L, Linv, b, noise_var=None, K=None, img_shape=None):

        # Bind
        self._A = A
        self._L = L
        self._Linv = Linv
        self.A = A @ Linv
        self.L = jlinops.IdentityOperator((Linv.shape[1], Linv.shape[1]))
        self._b = b
        self.noise_var = noise_var
        self.K = K
        self.n = self.A.shape[1]
        self.m = self.A.shape[0]
        self.k = self.L.shape[0]
        self.img_shape = img_shape

        assert K is None, "K must be none."

        # Whiten?
        if self.noise_var is not None:
            self.noise_stdev = np.sqrt(self.noise_var)
            self.A = (1.0/self.noise_stdev)*self.A
            self.b = (1.0/self.noise_stdev)*self._b
        else:
            self.noise_stdev = None
            self.b = self._b


    def update_weights(self, x, weight_params):

        raise NotImplementedError


    def build_initial_basis(self, dim):
        """
        Builds the initial basis.
        """

        Atb = self.A.T @ self.b

        V = np.zeros((self.n, dim))
        V[:,0] = Atb
        tmp = Atb.copy()
        for j in range(1,dim):
            tmp = self.A.T @ (self.A @ tmp)
            V[:,j] = tmp.copy()

        # V = np.random.normal(size=(self.n, 5))

        # Orthogonalize
        qr_fac = np.linalg.qr(V)
        V = qr_fac.Q

        return V
    

    def build_restarted_basis(self, dim, x, w, lambdah):
        """
        Builds the restarted basis.
        """
        Atb = self.A.T @ self.b

        V = np.zeros((self.n, dim))
        V[:,0] = Atb
        tmp = Atb.copy()
        for j in range(1,dim):
            tmp = (self.A.T @ (self.A @ tmp)) + lambdah*(self.L.T @ (w * (w * (self.L @ tmp ) ) ) )
            V[:,j] = tmp.copy()

        V = np.column_stack([V, x])

        # Orthogonalize
        qr_fac = np.linalg.qr(V)
        V = qr_fac.Q

        return V
    

    def build_priorconditioning_initial_basis(self, Atilde, c, lambdah, dim):
        """
        Builds the initial basis for the priorconditioned algorithm.
        """

        Atildetc = Atilde.T @ c

        V = np.zeros((Atilde.shape[1], dim))
        V[:,0] = Atildetc
        tmp = Atildetc
        for j in range(1,dim):
            # print(j)
            # print(tmp)
            tmp = Atilde.T @ (Atilde @ tmp)
            #tmp = Atilde.T @ (Atilde @ tmp) + lambdah*tmp
            tmp = tmp/np.linalg.norm(tmp)
            V[:,j] = tmp.copy()

        # V = np.random.normal(size=(self.n, 5))

        # Orthogonalize
        qr_fac = np.linalg.qr(V)
        V = qr_fac.Q

        return V
    

    def select_regparam(self, Q_A, R_A, Q_L, R_L, b, regparam, c=None, eng=None, xker=None, x_true=None, V=None, priorconditioning=False, Lopinv=None, **regparam_kwargs):
        """Selects regularization parameter for the projected problem.
        """

        if c is None:
            c = b

        # Select new regularization parameter
        if regparam == "mgcv": # Jonathan's MGCV method
            proposed_lambdah = mgcv(Q_A, R_A, R_L, c, **regparam_kwargs)
        elif regparam == "dp": # Trips-Py dp
            proposed_lambdah = discrepancy_principle(Q_A, R_A, R_L, c, delta=np.sqrt(self.m), explicitProj=True, **regparam_kwargs)
        elif regparam == 'gcv': # Trips-Py gcv
            proposed_lambdah = generalized_crossvalidation(Q_A, R_A, R_L, c, **regparam_kwargs)
        elif regparam == "dp2": # Jonathan's 2nd-order DP code. Requires 2 LS solves per iteration.
            shift = np.linalg.norm( b -  ( Q_A @ (Q_A.T @ b) ), ord=2 )**2 # needed for calculation w.r.t. full-scale objective
            # Hacky try-except block since code is not completely debugged for small \beta!
            try:
                data = dp_quadratic(R_A, R_L, Q_A.T @ c, noise_var=1.0, shift=shift, m=len(b), beta0=self.beta0, **regparam_kwargs)
            except:
                # If it didn't work with initializing beta0, try again without initializing it (defaults to beta0 very small). 
                data = dp_quadratic(R_A, R_L, Q_A.T @ c, noise_var=1.0, shift=shift, m=len(b), beta0=None, **regparam_kwargs)
                #pass
            proposed_lambdah = data["lambda"]
            self.beta0 = 1.0/proposed_lambdah
        elif regparam == "dp3": # Jonathan's 3rd-order DP code. Requires 3 LS solves per iteration.
            shift = np.linalg.norm( b -  ( Q_A @ (Q_A.T @ b) ), ord=2 )**2 # needed for calculation w.r.t. full-scale objective
            # Hacky try-except block since code is not completely debugged for small \beta!
            try:
                data = dp_cubic(R_A, R_L, Q_A.T @ c, noise_var=1.0, shift=shift, m=len(b), beta0=self.beta0, **regparam_kwargs)
            except:
                # If it didn't work with initializing beta0, try again without initializing it (defaults to beta0 very small). 
                data = dp_cubic(R_A, R_L, Q_A.T @ c, noise_var=1.0, shift=shift, m=len(b), beta0=None, **regparam_kwargs)
                #pass
            proposed_lambdah = data["lambda"]
            self.beta0 = 1.0/proposed_lambdah
        elif regparam == "matlab_gcv": # MATLAB GCV code
            proposed_lambdah = matlab_gcv( R_A, R_L, Q_A.T @ c, eng=eng, **regparam_kwargs)
        elif regparam == "optimal":
            assert x_true is not None, "Must provide x_true to use optimal regparam method."
            assert V is not None, "Must pass V to use optimal regparam method."
            if not priorconditioning:
                proposed_lambdah = optimal_regparam(R_A, R_L, Q_A, Q_L, self.b, V, x_true, **regparam_kwargs)
            else:
                assert xker is not None, "Must pass xker."
                proposed_lambdah = optimal_regparam(R_A, R_L, Q_A, Q_L, c, V, x_true, Lopinv=Lopinv, xker=xker, **regparam_kwargs)
        elif regparam == "optimal_ssim":
            assert x_true is not None, "Must provide x_true to use optimal regparam method."
            assert V is not None, "Must pass V to use optimal regparam method."
            if not priorconditioning:
                proposed_lambdah = optimal_regparam_ssim(R_A, R_L, Q_A, Q_L, self.b, V, x_true, **regparam_kwargs)
            else:
                assert xker is not None, "Must pass xker."
                proposed_lambdah = optimal_regparam_ssim(R_A, R_L, Q_A, Q_L, c, V, x_true, Lopinv=Lopinv, xker=xker, **regparam_kwargs)
        elif not isinstance(regparam, str): # hope that regparam is a scalar
            proposed_lambdah = regparam
        else:
            raise NotImplementedError

        return proposed_lambdah
    

    def compress_basis(self, V, R_A, R_L, x, Q_A, kmin, lambdah):
        """Compresses the basis using TSVD approach.
        """

        # Make H matrix
        print(R_A.shape)
        print(R_L.shape)
        H = np.vstack([R_A, np.sqrt(lambdah)*R_L ])
        print(H.shape)

        # Truncated SVD
        # _, _, Wh = svds(H, k=kmin-1)
        # W = Wh.T
        W = compute_V_matrix_k(H, kmin-1 )

        # Get new basis
        Vtilde = V @ W
        #assert np.linalg.norm( (Vtilde.T @ Vtilde) - np.eye(Vtilde.shape[1]) ) < 1e-10, "New basis is not a basis"
        qr_fac = np.linalg.qr(Vtilde)
        Vtilde = qr_fac.Q

        #print(np.linalg.norm( (Vtilde.T @ Vtilde) - np.eye(Vtilde.shape[1]) ))
        assert np.linalg.norm( (Vtilde.T @ Vtilde) - np.eye(Vtilde.shape[1]) ) < 1e-10, "New basis is not a basis"

        xnew = x -  Vtilde @ (Vtilde.T @ x)
        xnew /= np.linalg.norm(xnew)
        V = np.column_stack((Vtilde, xnew))
        assert np.linalg.norm( (V.T @ V) - np.eye(V.shape[1]) ) < 1e-10, "New basis is not a basis"

        # Orthogonalize
        qr_fac = np.linalg.qr(V)
        V = qr_fac.Q

        return V
    

    def solve(self, n_iter=25, x0=None, regparam=None, x_true=None, lambdah0=None, mode="W", lambdah_min=1e-6, lambdah_max=1e6, ksparsity=None, weight_params=None, refresh_basis=False, breakdown_tol=1e-14):

        valid_regparams = ["dp3", "dp"]
        assert regparam or np.isscalar(regparam), f"Invalid choice of regparam, must be scalar or one of {valid_regparams}"
        assert mode in ["I", "R", "W", "IRW"], "Mode must be either I, R, W, or IRW."

        # Tracking
        tot_matvec_A = 0
        tot_matvec_L = 0 
        tot_matvec_WLpinv = 0
        lambda_history = []
        residual_history = []
        rre_history = []
        ssim_history = []
        haar_history = []
        cond_ests = []
        ksparsity_indices = []
        hoyer_indices = []
        gini_indices = []
        dp_ineq_lhs = []
        dp_ineq_rhs = []
        dp_ineq_chs = []
        broke = False
        broke_down_idx = None

        # Initialize lambdah? Doesn't always actually affect anything, depending on the weight updates...
        if lambdah0 is None:
            #lambdah = 1.0
            #lambdah = regparam
            lambdah = 1.0
        else:
            lambdah = lambdah0

        # Initialization
        if x0 is None:
            x = self.A.T @ self.b
            tot_matvec_A += 1 # matvec with A
            tot_matvec_WLpinv += 1 # matvec with WLpinv
        else:
            x = self._L @ x0
            tot_matvec_WLpinv += 1 # matvec with WLpinv
            
        # Metrics
        if x_true is not None:
            xbar = self._Linv @ x
            x_true_norm = np.linalg.norm(x_true)
            rre_history.append( np.linalg.norm(xbar - x_true)/x_true_norm )

            if self.img_shape is None:
                ssim_history.append( structural_similarity( xbar, x_true, data_range=(np.amax(x_true) - np.amin(x_true) ), gaussian_weights=True, sigma=1.5, use_sample_covariance=False ) )
            else:
                ssim_history.append( structural_similarity( xbar.reshape(self.img_shape), x_true.reshape(self.img_shape), data_range=(np.amax(x_true) - np.amin(x_true) ), gaussian_weights=True, sigma=1.5, use_sample_covariance=False ) )
            
            if self.img_shape is not None:
                haar_calc, _, _ = haar_psi(x_true.reshape(self.img_shape), xbar.reshape(self.img_shape))
                haar_history.append(haar_calc)
        

        # Gini and hoyer
        gini = gini_index(x)
        gini_indices.append(gini)
        hoyer_index = hoyer_sparsity(x)
        hoyer_indices.append(hoyer_index)


        # Compute k-sparsity?
        if ksparsity is not None:
            ksparsity_indices.append( ksparsity_index(self.L @ x, ksparsity) )


        # Get initial weights and pseudoinverses
        w = self.update_weights(self.L @ x, lambdah, weight_params)
        tot_matvec_L += 1 # matvec with L
        Lpinv = jlinops.MatrixLinearOperator(sps.diags(1.0/w))
        if self.K is not None:
            AKpinv = jlinops.QRPinvOperator( jlinops.MatrixLinearOperator(self.A.matmat(self.K.A)) )
            tot_matvec_A += self.K.A.shape[1] # matvecs with A
            Lopinv = ( jlinops.IdentityOperator( (self.A.shape[1], self.A.shape[1])  ) - self.K @ (AKpinv @ self.A)  ) @ Lpinv
            Atilde = self.A @ Lopinv
        else:
            Atilde = self.A @ Lpinv
            Lopinv = Lpinv

        xker = np.zeros(self.A.shape[1])
        c = self.b

        # Initial calculations
        #xinc = self.L @ (Lopinv @ ( self.L @ x ))
        #r = c - Atilde @ xinc
        r = c
        #r = self.b - self.A @ x
        u  = r / np.linalg.norm(r)
        U = u[:,None]


        for i in progress_bar(range(n_iter)):

            print(f"Starting iteration {i}")
            
            ### Get new weights

            # Get initial weights and pseudoinverses
            w = self.update_weights(self.L @ x, lambdah, weight_params)
            # tot_matvec_L += 1 # matvec with L
            Lpinv = jlinops.MatrixLinearOperator(sps.diags(1.0/w))
            # if mode in ["I", "R", "W"]:
            #     Lpinv = jlinops.MatrixLinearOperator(sps.diags(1.0/w))
            # else:
            #     Lpinv = jlinops.MatrixLinearOperator(sps.diags((1.0/w)**2))
            Lopinv = Lpinv


            ### Update the FGK factorization
            if i == 0:
                U, M, T, V, Z, broke = update_fgk(self.A, None, U, None, None, None, Lopinv, breakdown_tol=breakdown_tol )
            else:
                U, M, T, V, Z, broke = update_fgk(self.A, Z, U, M, V, T, Lopinv, breakdown_tol=breakdown_tol )
            tot_matvec_A += 2 # matvec with A
            tot_matvec_WLpinv += 2 # matvec with WLpinv (Lpinv)
            
            if broke:
                broke_down_idx = i
                break

            # Check V is semi-orthogonal? 
            V_test = np.linalg.norm( (V.T @ V) - np.eye(V.shape[1]) )
            if V_test > 1e-10:
                warnings.warn(f"V appears to no longer be semi-orthogonal, failes test with value: {V_test}.", UserWarning)

            # Check U is semi-orthogonal? 
            U_test = np.linalg.norm( (U.T @ U) - np.eye(U.shape[1]) )
            if U_test > 1e-10:
                warnings.warn(f"U appears to no longer be semi-orthogonal, fails test with value {U_test}.", UserWarning)

            # Other calculations
            W = sps.diags(w)
            e1 = np.zeros(M.shape[0])
            e1[0] = 1.0

            if mode == "R":
                # FLSQR-R (Z)
                N = Z
                (Q_N, R_N) = la.qr(N, mode='economic')
            elif mode == "I":
                # FLSQR-I (I)
                R_N = np.eye(M.shape[1])
            elif (mode == "W") or (mode == "IRW"):
                # FLSQR-W (W Z)
                N = W @ Z
                (Q_N, R_N) = la.qr(N, mode='economic')
            else:
                raise NotImplementedError
            

            try:
                proposed_lambdah = discrepancy_principle(np.eye(len(e1)), np.atleast_2d(M), R_N, np.linalg.norm(c)*e1[:,None], delta=np.sqrt(self.m), explicitProj=False)
            except:
                proposed_lambdah = 1e-14

            lambdah = max(lambdah_min, proposed_lambdah)
            lambdah = min(lambdah_max, lambdah)

            print(lambdah)
            lambda_history.append(lambdah)

            # Solve for fixed lambda
            y, _ , _, svals = np.linalg.lstsq(np.vstack( (M, np.sqrt(lambdah) * R_N ) ), np.concatenate(( np.linalg.norm(c)*e1,  np.zeros(R_N.shape[0]) )), rcond=-1)
            cond_est = np.amax(svals)/np.amin(svals)
            cond_ests.append(cond_est)


            # Recover full solution
            x = Z @ y # Project back onto z space
            xfull = x
            xbar = self._Linv @ xfull
            tot_matvec_WLpinv += 1 # matvec with WLpinv (Lpinv)


            # Metrics
            if x_true is not None:
                x_true_norm = np.linalg.norm(x_true)
                rre_history.append( np.linalg.norm(xbar - x_true)/x_true_norm )
                if self.img_shape is None:
                    ssim_history.append( structural_similarity( xbar, x_true, data_range=(np.amax(x_true) - np.amin(x_true) ), gaussian_weights=True, sigma=1.5, use_sample_covariance=False ) )
                else:
                    ssim_history.append( structural_similarity( xbar.reshape(self.img_shape), x_true.reshape(self.img_shape), data_range=(np.amax(x_true) - np.amin(x_true) ), gaussian_weights=True, sigma=1.5, use_sample_covariance=False ) )
                
                if self.img_shape is not None:
                    haar_calc, _, _ = haar_psi(x_true.reshape(self.img_shape), xbar.reshape(self.img_shape))
                    haar_history.append(haar_calc)

            # Gini and hoyer
            gini = gini_index(x)
            gini_indices.append(gini)
            hoyer_index = hoyer_sparsity(x)
            hoyer_indices.append(hoyer_index)

            # Compute k-sparsity?
            if ksparsity is not None:
                ksparsity_indices.append( ksparsity_index(self.L @ x, ksparsity) )


        # Package data
        data = {
            "x": xbar,
            "_x": x,
            "V": V,
            "U": U,
            "M": M,
            "Z": Z,
            "T": T,
            "w": w,
            "tot_matvec_A": tot_matvec_A,
            "tot_matvec_L": tot_matvec_L,
            "tot_matvec_WLpinv": tot_matvec_WLpinv,
            "lambda_history": np.asarray(lambda_history),
            "cond_ests": np.asarray(cond_ests),
            "broke_down": broke,
            "broke_down_idx": broke_down_idx,
        }

        if broke:
            data["n_completed_iterations"] = broke_down_idx
        else:
            data["n_completed_iterations"] = n_iter

        data["hoyer_indices"] = np.asarray(hoyer_indices)
        data["gini_indices"] = np.asarray(gini_indices)

        if x_true is not None:
            data["rre_history"] = np.asarray(rre_history)
            data["ssim_history"] = np.asarray(ssim_history)
            data["haar_history"] = np.asarray(haar_history)

        if ksparsity is not None:
            data["ksparsity_indices"] = np.asarray(ksparsity_indices)
        

        return data








## Specific update formulas

def conditional_mode_ode_rhs(varphi, z, r):
        """RHS of the ODE used for updating theta.
        """
        
        dvarphidz = (2*z*varphi)/((2*(r**2)*((varphi)**(r+1))) + (z**2))

        return dvarphidz



def theta_update(Lx, r, beta, vartheta):
    """Returns the theta-update IAS.
    """

    eta = r*beta - 1.5
    initial_value = (eta/r)**(1/r)

    # Rescale
    z = np.abs(  ( Lx )  / np.sqrt(vartheta) )

    # If r = 1, use exact formula
    if abs(r - 1) < 1e-5:    
            # print("Using exact formula for r = 1")
            xi = 0.5*( eta + np.sqrt( (eta**2) +  2*(z**2)  ) )
            new_theta = vartheta*xi
            
            return new_theta

    elif abs(r + 1) < 1e-5:
            
            # print("Using exact formula for r = -1")
            
            k = beta + 1.5
            xi = (1/(2*k)) * ( (z**2) + 2 )
            new_theta = vartheta*xi
            
            return new_theta

    # Otherwise, solve using the ODE method
    else:
        
        #print("Using ODE method")
        
        final_times = z

        # Sort the final times
        argsort = final_times.argsort()
        final_times_sorted = final_times[argsort]

        # We need to prepend zero to this for the solver
        final_times_sorted = np.insert(final_times_sorted, 0, 0)

        # Now solve the ODE
        ode_sol = odeint(conditional_mode_ode_rhs,
                            np.atleast_1d(initial_value), 
                            final_times_sorted,
                            args=(r,) # r parameter
                        )

        # Reshape and drop the first value corresponding to the dummy initial value
        ode_sol = ode_sol[1:,0]

        # Now back out the updated value of the hyper-parameter
        xi = np.zeros_like(final_times)
        xi[argsort] = ode_sol
        new_theta = vartheta*xi

        return new_theta




def f_tau(x, tau1, tau2):
    
    # Compute the absolute value of the input
    abs_x = np.abs(x)
    
    # Apply the piecewise function
    return np.where(abs_x >= tau1, abs_x, tau2)


class MM2019FGK2019(FGK2019Solver):

    """Uses version of MM weights given in Chung 2019.
    q corresponds to the lq norm. 
    tau1 and tau2 correspond to the thresholds.
    """
 

    def __init__(self,  A, L, Linv, b, noise_var=None, K=None, img_shape=None):

        super().__init__(A, L, Linv, b, noise_var=noise_var, K=K, img_shape=img_shape)


    def update_weights(self, Lx, lambdah, weight_params):
        
        w = np.sqrt(np.power(f_tau(Lx, weight_params["tau1"], weight_params["tau2"]), weight_params["q"]-2))

        return w



class MMFGK2019(FGK2019Solver):

    def __init__(self,  A, L, Linv, b, noise_var=None, K=None, img_shape=None):

        super().__init__(A, L, Linv, b, noise_var=noise_var, K=K, img_shape=img_shape)


    def update_weights(self, Lx, lambdah, weight_params):
        
        w = np.sqrt( mm_weights( Lx, epsilon=weight_params["epsilon"], p=weight_params["q"] ) )

        return w



class IASFGK2019(FGK2019Solver):

    def __init__(self,  A, L, Linv, b, noise_var=None, K=None, img_shape=None):

        super().__init__(A, L, Linv, b, noise_var=noise_var, K=K, img_shape=img_shape)


    def update_weights(self, Lx, lambdah, weight_params):

        theta = theta_update( Lx/np.sqrt(1.0/lambdah), weight_params["hyperpr_r"], weight_params["hyperpr_beta"], 1.0)
        w = 1.0/np.sqrt(theta)

        return w
    


class VanillaFGK2019(FGK2019Solver):

    def __init__(self,  A, L, Linv, b, noise_var=None, K=None, img_shape=None):

        super().__init__(A, L, Linv, b, noise_var=noise_var, K=K, img_shape=img_shape)


    def update_weights(self, Lx, lambdah, weight_params):

        return np.ones_like(Lx)
    












## MM methods for adherring to specific papers

class MM1FGK2019(FGK2019Solver):

    def __init__(self,  A, L, Linv, b, noise_var=None, K=None, img_shape=None):

        super().__init__(A, L, Linv, b, noise_var=noise_var, K=K, img_shape=img_shape)


    def update_weights(self, Lx, lambdah, weight_params):
        
        w = np.sqrt(  mm_weights( Lx, epsilon=1.0, p=weight_params["q"] ) )

        return w
    

class MM2FGK2019(FGK2019Solver):

    def __init__(self,  A, L, Linv, b, noise_var=None, K=None, img_shape=None):

        super().__init__(A, L, Linv, b, noise_var=noise_var, K=K, img_shape=img_shape)


    def update_weights(self, Lx, lambdah, weight_params):
        
        w = np.sqrt(  mm_weights( Lx, epsilon=0.01, p=weight_params["q"] ) )

        return w
    

class MM3FGK2019(FGK2019Solver):

    def __init__(self,  A, L, Linv, b, noise_var=None, K=None, img_shape=None):

        super().__init__(A, L, Linv, b, noise_var=noise_var, K=K, img_shape=img_shape)


    def update_weights(self, Lx, lambdah, weight_params):
        
        w = np.sqrt(  mm_weights( Lx, epsilon=1e-10, p=weight_params["q"] ) )

        return w
    


class MM4FGK2019(FGK2019Solver):

    def __init__(self,  A, L, Linv, b, noise_var=None, K=None, img_shape=None):

        super().__init__(A, L, Linv, b, noise_var=noise_var, K=K, img_shape=img_shape)


    def update_weights(self, Lx, lambdah, weight_params):
        
        w = np.sqrt( np.power(f_tau(Lx, 1e-10, 1e-16), weight_params["q"]-2) )

        return w
    




















