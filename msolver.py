import numpy as np
import matplotlib.pyplot as plt

# Add files up one level to the path
import sys
# sys.path.append("..")

import jlinops

from matplotlib.backends.backend_pdf import PdfPages

from scipy.linalg import qr, qr_insert
from scipy.integrate import odeint
import scipy.optimize as op
import scipy.sparse as sps
from scipy import linalg as la

from fastprogress import progress_bar

import warnings

import scipy.optimize as op
from scipy.sparse.linalg import svds
from scipy.linalg import null_space

from skimage.metrics import structural_similarity


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

from haarpsi import haar_psi

import time




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




def compute_V_matrix_k(A, k):
    """
    Computes the matrix V from the Singular Value Decomposition (SVD) of matrix A
    and returns only the first k columns of V.
    
    A = U * Σ * V^T

    Parameters:
    A (ndarray): Input matrix.
    k (int): Number of columns to return from matrix V.

    Returns:
    V_k (ndarray): The first k columns of the right singular vectors of A.
    """
    # Perform SVD
    U, Sigma, Vt = np.linalg.svd(A)
    
    # V is the transpose of Vt
    V = Vt.T
    
    # Return only the first k columns of V
    V_k = V[:, :k]
    
    return V_k



class GKSSolver:
    """Implements a generic GKS solver for problems of the form:

    If noise_var is provided, then the problem solved is 
        argmin_x  (1/noise_var)*|| A x - b ||_2^2 + \lambda || L x ||_q^q.

    """

    def __init__(self, A, L, b, noise_var=None, Lpinv_factory=None, K=None, img_shape=None):

        # Bind
        self._A = A
        self.L = L
        self._b = b
        self.noise_var = noise_var
        self.Lpinv_factory = Lpinv_factory
        self.K = K
        self.n = A.shape[1]
        self.m = A.shape[0]
        self.k = L.shape[0]
        self.img_shape = img_shape

        # Whiten?
        if self.noise_var is not None:
            self.noise_stdev = np.sqrt(self.noise_var)
            self.A = (1.0/self.noise_stdev)*self._A
            self.b = (1.0/self.noise_stdev)*self._b
        else:
            self.noise_stdev = None
            self.A = self._A
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
    

    def compress_priorconditioned_basis(self, V, R_Abar, R_L, z, Q_A, kmin, lambdah):
        """Compresses the priorconditioned basis using TSVD approach.
        """

        # Make H matrix
        H = np.vstack([R_Abar, np.sqrt(lambdah)*R_L ])

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

        znew = z -  Vtilde @ (Vtilde.T @ z)
        znew /= np.linalg.norm(znew)
        V = np.column_stack((Vtilde, znew))
        assert np.linalg.norm( (V.T @ V) - np.eye(V.shape[1]) ) < 1e-10, "New basis is not a basis"

        # Orthogonalize
        qr_fac = np.linalg.qr(V)
        V = qr_fac.Q

        return V



    def solve(self, n_iter=25, init_dim=5, regparam="dp", restart_dim=None, lambdah0=None, lambdah_min=1e-6, lambdah_max=1e6, Vinit=None, x0=None, x_true=None, err_norm_sq=None, weight_params=None, min_dim=None, max_dim=None, xker=None, ksparsity=None, regparam_kwargs={}, keep_solution_history=False, **kwargs):
        """Driver for GKS solver.
        """

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
        dp_ineq_lhs = []
        dp_ineq_rhs = []
        dp_ineq_chs = []
        ksparsity_indices = []
        hoyer_indices = []
        gini_indices = []
        end_time = time.time()
        solution_history = []

        # Initialize lambdah? Doesn't always actually affect anything, depending on the weight updates...
        if lambdah0 is None:
            lambdah = 1.0
        else:
            lambdah = lambdah0

        # Are we using recycling?
        recycling = False
        if min_dim is not None:
            assert max_dim is not None, "Must pass max_dim for recycling..."
        
        if max_dim is not None:
            assert min_dim is not None, "Must pass min_dim for recycling..."
            recycling = True

        # Are we restarting?
        restarting = False
        if restart_dim is not None:
            restarting = True
            assert not (restarting and recycling), "Can't do both restarting and recycling, currently..."

        # Initialization
        if x0 is None:
            x = self.A.T @ self.b
            tot_matvec_A += 1 # 1 A matvec for A^T b
        else:
            x = x0

        if keep_solution_history:
            solution_history.append(x)

        #if xker is not None:
            #c = self.b - (self.A @ xker)
            #tot_matvec_A += 1 # 1 A matvec for A x_ker

        if err_norm_sq is None:
            err_norm_sq = len(self.b)

        if x_true is not None:
            x_true_norm = np.linalg.norm(x_true)
            rre_history.append( np.linalg.norm(x - x_true)/x_true_norm )
            if self.img_shape is None:
                ssim_history.append( structural_similarity( x, x_true, data_range=(np.amax(x_true) - np.amin(x_true) ), gaussian_weights=True, sigma=1.5, use_sample_covariance=False ) )
            else:
                ssim_history.append( structural_similarity( x.reshape(self.img_shape), x_true.reshape(self.img_shape), data_range=(np.amax(x_true) - np.amin(x_true) ), gaussian_weights=True, sigma=1.5, use_sample_covariance=False ) )
            if self.img_shape is not None:
                haar_calc, _, _ = haar_psi(x_true.reshape(self.img_shape), x.reshape(self.img_shape))
                haar_history.append(haar_calc)

        # Gini and hoyer
        gini = gini_index(self.L @ x)
        gini_indices.append(gini)
        hoyer_index = hoyer_sparsity(self.L @ x)
        hoyer_indices.append(hoyer_index)

        # Compute k-sparsity?
        if ksparsity is not None:
            ksparsity_indices.append( ksparsity_index(self.L @ x, ksparsity) )

        # Compute k-sparsity?
        if ksparsity is not None:
            ksparsity_indices.append( ksparsity_index(self.L @ x, ksparsity) )

        # Build initial basis and QR factor of AV
        if Vinit is None: 
            V = self.build_initial_basis(init_dim)
        else:
            V = Vinit.copy()
        dim_size = init_dim
        tot_matvec_A += (2*(init_dim - 1)) + 1

        # If we passed x0, add it to the basis as another column.
        if x0 is not None:
            vnew = x0 - (V @ (V.T @ x0) )
            vnew /= np.linalg.norm(vnew)
            V = np.column_stack((V, vnew))
            dim_size += 1
            print("Augmented basis with initial guess!")

        AV = self.A @ V 
        tot_matvec_A += V.shape[1] # matvecs for AV
        Q_A, R_A = qr(AV, mode='economic') 
        

        # Get initial LV
        LV = np.hstack( [ (self.L @ V[:,j])[:,None] for j in range(V.shape[1]) ]  )
        #LV = self.L @ V
        #print(LV.shape)
        tot_matvec_L += V.shape[1] # matvecs for LV

        # For cubic dp
        self.beta0 = None

        # Initialize lambdah? Doesn't always actually affect anything, depending on the weight updates...
        lambdah = 1.0

        # Start matlab engine if regparam == matlab_gcv
        _eng = None
        if regparam == "matlab_gcv":
            # If this throws an error, it is because matlab engine is not set up correctly
            _eng = matlab.engine.start_matlab()


        for ii in progress_bar(range(n_iter)):

            # Update weights and get new QR factorization of W L V
            Lx = self.L @ x
            tot_matvec_L += 1 # one more L matvec
            w = self.update_weights(Lx, lambdah, weight_params)
            W = sps.diags(w)
            WLV = W @ LV
            Q_WL, R_WL = qr(WLV, mode='economic') 

            eta = 1.01
            if xker is not None:
                shift = np.linalg.norm(self.b - (Q_A @ (Q_A.T @ self.b ) ) )**2
                tmp, _, _, _ = np.linalg.lstsq(R_A, Q_A.T @ self.b, rcond=-1)
                #tmp = np.linalg.pinv(A) @ b
                lhs = (np.linalg.norm( (R_A @ tmp) - (Q_A.T @ self.b) )**2)
                dp_ineq_lhs.append(lhs)

                # Get reduced oblique pinv to get the rhs
                Kreduced = null_space(Q_WL)
                if Kreduced.shape[1] == 0:
                    rhs = np.linalg.norm(Q_A.T @ self.b)
                else:
                    tmp, _, _, _ = np.linalg.lstsq( Q_A @ Kreduced, Q_A.T @ self.b, rcond=-1 )
                    creduced = Kreduced @ tmp
                    rhs = np.linalg.norm(creduced)**2
                dp_ineq_rhs.append(rhs)

                inside = ((eta**2)*err_norm_sq) - shift
                dp_ineq_chs.append(inside)
                check1 = lhs <= inside
                check2 = inside <= rhs
                if check1 and check2:
                    pass
                else:
                    warnings.warn("Inequality is NOT satisfied, no solution to the DP problem...", UserWarning)

            
            # Select new regularization parameter
            proposed_lambdah = self.select_regparam(Q_A, R_A, Q_WL, R_WL, self.b, regparam=regparam, eng=_eng, V=V, x_true=x_true, priorconditioning=False, **regparam_kwargs)
          
            lambdah = max(lambdah_min, proposed_lambdah)
            lambdah = min(lambdah_max, lambdah)
            print(lambdah)

            # Screen the proposed lambdah somehow? Just accept for now...
            #print(proposed_lambdah)
            #lambdah = proposed_lambdah
            lambda_history.append(lambdah)
            try:
                beta0 = 1.0/lambdah
            except:
                beta0 = 1e10
            
            # Obtain new solution
            y, _ , _, svals = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_WL)), np.concatenate((Q_A.T @ self.b, np.zeros((R_WL.shape[0],)))), rcond=-1)        
            cond_est = np.amax(svals)/np.amin(svals)
            cond_ests.append(cond_est)
            x = V @ y

            if x_true is not None: 
                rre_history.append( np.linalg.norm(x - x_true)/x_true_norm )
                if self.img_shape is None:
                    ssim_history.append( structural_similarity( x, x_true, data_range=(np.amax(x_true) - np.amin(x_true) ), gaussian_weights=True, sigma=1.5, use_sample_covariance=False ) )
                else:
                    ssim_history.append( structural_similarity( x.reshape(self.img_shape), x_true.reshape(self.img_shape), data_range=(np.amax(x_true) - np.amin(x_true) ), gaussian_weights=True, sigma=1.5, use_sample_covariance=False ) )
                if self.img_shape is not None:
                    haar_calc, _, _ = haar_psi(x_true.reshape(self.img_shape), x.reshape(self.img_shape))
                    haar_history.append(haar_calc)

            # Gini and hoyer
            gini = gini_index(self.L @ x)
            gini_indices.append(gini)
            hoyer_index = hoyer_sparsity(self.L @ x)
            hoyer_indices.append(hoyer_index)

            # Compute k-sparsity?
            if ksparsity is not None:
                ksparsity_indices.append( ksparsity_index(self.L @ x, ksparsity) )

            if keep_solution_history:
                solution_history.append(x)

            # if ii >= R_WL.shape[0]:
            #     break
            
            # Get new residual
            v = AV @ y
            v = v - self.b
            u = LV @ y
            ra = AV @ y - self.b
            ra = self.A.T @ ra
            rb = W @ (LV @ y)
            rb = self.L.T @ rb
            r = ra + lambdah * rb
            r = r - V @ (V.T @ r) # re-orthogonalize
            vn = r / np.linalg.norm(r)
            V = np.column_stack((V, vn))
            Avn = self.A @ vn
            AV = np.column_stack((AV, Avn))
            Lvn = vn
            Lvn = self.L * vn
            LV = np.column_stack((LV, Lvn))
            tot_matvec_A += 2 # matvecs with A
            tot_matvec_L += 2 # matvecs with L
            residual_history.append(np.linalg.norm(r))
            dim_size += 1

            # Update QR factor for A
            Q_A, R_A = qr_insert( Q_A, R_A, Avn, AV.shape[1]-1, which="col")
            #Q_A, R_A = qr(AV, mode='economic') 

            # If dim_size == kmax, need to compress the basis
            if (dim_size == max_dim) and recycling:
                
                # Update weights and get new QR factorization of W L V
                #LV = self.L @ V # matvecs with L
                #tot_matvec_L += V.shape[1]
                WLV = W @ LV
                Q_WL, R_WL = qr(WLV, mode='economic') 
                #Q_L, R_L = qr(LV, mode='economic') 
                
                # New basis
                V = self.compress_basis(V.copy(), R_A, R_WL, x, Q_A, min_dim, lambdah)
                x = V @ (V.T @ x)
                
                # Other 
                dim_size = min_dim
                AV = self.A @ V 
                LV = self.L @ V
                tot_matvec_A += V.shape[1] # matvecs with A
                tot_matvec_L += V.shape[1] # matvecs with L
                Q_A, R_A = qr(AV, mode='economic')

            if (dim_size == restart_dim) and restarting:
                
                # Reset method with initial basis based on current x
                # Lx = self.L @ x
                # tot_matvec_L += 1
                # w = self.update_weights(Lx, lambdah, weight_params)
                # V = self.build_restarted_basis(init_dim, x, w, lambdah)
                # dim_size = init_dim

                # # Reset basis
                tmp = x.copy()
                tmp /= np.linalg.norm(tmp)
                V = tmp.copy()
                V = V[:,None]
                dim_size = 1
                
                # Update weights and get new QR factorization of W L V
                LV = self.L @ V
                tot_matvec_L += V.shape[1] # matvecs with L
                AV = self.A @ V 
                tot_matvec_A += V.shape[1] # matvecs with A
                #WLV = W @ LV
                #Q_WL, R_WL = qr(WLV, mode='economic') 
                Q_A, R_A = qr(AV, mode='economic')



        # Package output
        data = {
            "x": x,
            "w": w,
            "lambdah": lambdah,
            "tot_matvec_A": tot_matvec_A,
            "tot_matvec_L": tot_matvec_L,
            "tot_matvec_WLpinv": tot_matvec_WLpinv,
            "residual_history": np.asarray(residual_history),
            "lambda_history": np.asarray(lambda_history),
            "V": V,
            "cond_ests": np.asarray(cond_ests),
            "dp_ineq_lhs": np.asarray(dp_ineq_lhs),
            "dp_ineq_rhs": np.asarray(dp_ineq_rhs),
            "dp_ineq_chs": np.asarray(dp_ineq_chs),
            "n_completed_iterations": n_iter,
            "solution_history": solution_history,
        }

        if x_true is not None:
            data["rre_history"] = np.asarray(rre_history)
            data["ssim_history"] = np.asarray(ssim_history)
            data["haar_history"] = np.asarray(haar_history)

        data["hoyer_indices"] = np.asarray(hoyer_indices)
        data["gini_indices"] = np.asarray(gini_indices)


        if ksparsity is not None:
            data["ksparsity_indices"] = np.asarray(ksparsity_indices)

        # Close engine if need to
        if regparam == "matlab_gcv": _eng.close()

        return data
        


    def solve_with_priorconditioning(self, n_iter=25, init_dim=5, regparam="dp", restart_dim=None, lambdah0=None, lambdah_min=1e-6, lambdah_max=1e6, x0=None, x_true=None, weight_params=None, min_dim=None, max_dim=None, ksparsity=None, regparam_kwargs={}, err_norm_sq=0.0, keep_solution_history=False, **kwargs):
        """Driver for GKS solver.
        """
        valid_regparams = ["dp3", "dp"]
        assert regparam or np.isscalar(regparam), f"Invalid choice of regparam, must be scalar or one of {valid_regparams}"
        assert self.Lpinv_factory is not None, "Must pass Lpinv_factory at instantiation..."
        # assert self.K is not None, "Must pass K at instantiation..."

        if err_norm_sq is None:
            err_norm_sq = len(self.b)

        # Tracking
        tot_matvec_A = 0
        tot_matvec_L = 0 
        tot_matvec_WLpinv = 0
        lambda_history = []
        residual_history = []
        rre_history = []
        ssim_history = []
        ksparsity_indices = []
        hoyer_indices = []
        gini_indices = []
        haar_history = []
        cond_ests = []
        dp_ineq_lhs = []
        dp_ineq_rhs = []
        dp_ineq_chs = []
        solution_history = []

        # Initialize lambdah? Doesn't always actually affect anything, depending on the weight updates...
        if lambdah0 is None:
            lambdah = 1.0
        else:
            lambdah = lambdah0

        # Are we using recycling?
        recycling = False
        if min_dim is not None:
            assert max_dim is not None, "Must pass max_dim for recycling..."
        
        if max_dim is not None:
            assert min_dim is not None, "Must pass min_dim for recycling..."
            recycling = True

        # Are we restarting?
        restarting = False
        if restart_dim is not None:
            restarting = True
            assert not (restarting and recycling), "Can't do both restarting and recycling, currently..."

        # Initialization
        if x0 is None:
            x = self.A.T @ self.b
            tot_matvec_A += 1 # matvec with A
        else:
            x = x0

        if x_true is not None:
            x_true_norm = np.linalg.norm(x_true)
            rre_history.append( np.linalg.norm(x - x_true)/x_true_norm )
            if self.img_shape is None:
                ssim_history.append( structural_similarity( x, x_true, data_range=(np.amax(x_true) - np.amin(x_true) ), gaussian_weights=True, sigma=1.5, use_sample_covariance=False ) )
            else:
                ssim_history.append( structural_similarity( x.reshape(self.img_shape), x_true.reshape(self.img_shape), data_range=(np.amax(x_true) - np.amin(x_true) ), gaussian_weights=True, sigma=1.5, use_sample_covariance=False ) )
            if self.img_shape is not None:
                haar_calc, _, _ = haar_psi(x_true.reshape(self.img_shape), x.reshape(self.img_shape))
                haar_history.append(haar_calc)

        # Gini and hoyer
        gini = gini_index(self.L @ x)
        gini_indices.append(gini)
        hoyer_index = hoyer_sparsity(self.L @ x)
        hoyer_indices.append(hoyer_index)

        # Compute k-sparsity?
        if ksparsity is not None:
            ksparsity_indices.append( ksparsity_index(self.L @ x, ksparsity) )

        # Get initial weights
        w = self.update_weights(self.L @ x, lambdah, weight_params)
        tot_matvec_L += 1 # matvec with L
        Lpinv = self.Lpinv_factory(w)
        if self.K is not None:
            AKpinv = jlinops.QRPinvOperator( jlinops.MatrixLinearOperator(self.A.matmat(self.K.A)) )
            tot_matvec_A += self.K.A.shape[1] # matvecs with A for AKpinv
            Lopinv = ( jlinops.IdentityOperator( (self.A.shape[1], self.A.shape[1])  ) - self.K @ (AKpinv @ self.A)  ) @ Lpinv
            Atilde = self.A @ Lopinv
        else:
            Atilde = self.A @ Lpinv
            Lopinv = Lpinv

        if self.K is not None:
            xker = self.K @ (AKpinv @ self.b)
            Axker = self.A @ xker
            tot_matvec_A += 1 # A matvec for A x_ker
            c = self.b - Axker
        else:
            xker = np.zeros(self.A.shape[1])
            c = self.b


        # # Build initial basis and QR factor of AV # new initial basis method, 12/6
        # V = self.build_initial_basis(init_dim)
        # dim_size = init_dim
        # tot_matvec_A += (2*(init_dim - 1)) + 1

        # # If we passed x0, add it to the basis as another column.
        # if x0 is not None:
        #     vnew = x0 - (V @ (V.T @ x0) )
        #     vnew /= np.linalg.norm(vnew)
        #     V = np.column_stack((V, vnew))
        #     dim_size += 1
        #     print("Augmented basis with initial guess!")

        # # Transform V and qr factor to reorthogonalize
        # V = np.hstack([ self.L.matvec(V[:,j])[:,None] for j in range(V.shape[1])  ])
        # (V, _) = la.qr(V, mode='economic') 
        # tot_matvec_L += V.shape[1]
        # dim_size = V.shape[1]


        ## Old version of initial basis, before 12/6
        # Build initial basis
        V = self.build_priorconditioning_initial_basis(Atilde, c, lambdah, init_dim)
        #(U, B, V) = golub_kahan(Atilde, c, init_dim, False, **kwargs) 
        qr_fac = np.linalg.qr(V)
        V = qr_fac.Q
        tot_matvec_A += 2*(V.shape[1]-1) + 1 # matvec with A
        tot_matvec_WLpinv += 2*(V.shape[1]-1) + 1 # matvec with WLpinv
        dim_size = V.shape[1]
        

        # If x0 is passed, augment initial basis
        if x0 is not None:
            # tmp = x0 - xker
            tmp = self.L @ (Lopinv @ (self.L @ x0)) # implied initialization in transformed space
            tot_matvec_L += 2 # matvec with L
            tot_matvec_WLpinv += 1 # matvec with WLpinv
            vnew = tmp - (V @ (V.T @ tmp) )
            vnew /= np.linalg.norm(vnew)
            V = np.column_stack((V, vnew))
            #dim_size += 1
            print("Augmented basis with initial guess!")
        
        # Track solution history?
        if keep_solution_history:
            solution_history.append( (x, V, Lopinv ) )

        # For cubic dp
        self.beta0 = None

        # Start matlab engine if regparam == matlab_gcv
        _eng = None
        if regparam == "matlab_gcv":
            # If this throws an error, it is because matlab engine is not set up correctly
            _eng = matlab.engine.start_matlab()

        for ii in progress_bar(range(n_iter)):
            
            # Update weights and get new QR factorization of W L V
            Lx = self.L @ x
            tot_matvec_L += 1 # matvec with L
            w = self.update_weights(Lx, lambdah, weight_params)
            W = sps.diags(w)
            
            # Build new Atilde
            Lpinv = self.Lpinv_factory(w)
            if self.K is not None:
                Lopinv = ( jlinops.IdentityOperator( (self.A.shape[1], self.A.shape[1])  ) - self.K @ (AKpinv @ self.A)  ) @ Lpinv
                Atilde = self.A @ Lopinv
            else:
                Atilde = self.A @ Lpinv
                Lopinv = Lpinv

            # Get new QR factorization for Atilde
            AtildeV = np.hstack([ Atilde.matvec(V[:,j])[:,None] for j in range(V.shape[1])  ])
            tot_matvec_A += V.shape[1] # matvecs with A
            tot_matvec_WLpinv += V.shape[1] # matvecs with WLpinv
            AA = AtildeV
            (Q_Atilde, R_Atilde) = la.qr(AA, mode='economic') 

            # What used to be "R_L", is now just the identity matrix
            R_L = np.eye(V.shape[1])
            Q_L = np.eye(V.shape[1])
            LV = V

            # Misc
            eta = 1.01

            ### Check if DP problem solution exists?
            shift = np.linalg.norm(c - (Q_Atilde @ (Q_Atilde.T @ c ) ) )**2
            tmp, _, _, _ = np.linalg.lstsq(R_Atilde, Q_Atilde.T @ c, rcond=-1)
            #tmp = np.linalg.pinv(A) @ b
            lhs = (np.linalg.norm( (R_Atilde @ tmp) - (Q_Atilde.T @ c) )**2)
            dp_ineq_lhs.append(lhs)
            rhs = np.linalg.norm( Q_Atilde.T @ c)**2
            dp_ineq_rhs.append(rhs)
            inside = ((eta**2)*err_norm_sq) - shift
            dp_ineq_chs.append(inside)
            check1 = lhs <= inside
            check2 = inside <= rhs
            if check1 and check2:
                pass
            else:
                warnings.warn("Inequality is NOT satisfied, no solution to the DP problem...", UserWarning)


            # Select new regularization parameter
            proposed_lambdah = self.select_regparam(Q_Atilde, R_Atilde, Q_L, R_L, c, regparam=regparam, eng=_eng, c=c, V=V, x_true=x_true, xker=xker, priorconditioning=True, Lopinv=Lopinv, **regparam_kwargs)
            lambdah = max(lambdah_min, proposed_lambdah)
            lambdah = min(lambdah_max, lambdah)
            print(lambdah)

            # Screen the proposed lambdah somehow? Just accept for now...
            #print(proposed_lambdah)
            #proposed_lambdah = max(proposed_lambdah, 1e-8)
            #lambdah = proposed_lambdah
            lambda_history.append(lambdah)
            try:
                self.beta0 = 1.0/lambdah
            except:
                self.beta0 = 1e10
                
            self.beta0 = None


            y, _ , _, svals = np.linalg.lstsq(np.vstack( (R_Atilde, np.sqrt(lambdah) * np.eye(R_Atilde.shape[1])) ), np.concatenate((Q_Atilde.T @ c, np.zeros((R_L.shape[0],)))),rcond=-1)
            cond_est = np.amax(svals)/np.amin(svals)
            cond_ests.append(cond_est)
            z = V @ y # Project back onto z space
            if self.K is not None:
                x = (Lopinv @ z) + xker # Project back onto x space
            else:
                x = Lpinv @ z
            tot_matvec_WLpinv += 1 # matvec with WLpinv

            if x_true is not None: 
                rre_history.append( np.linalg.norm(x - x_true)/x_true_norm )
                if self.img_shape is None:
                    ssim_history.append( structural_similarity( x, x_true, data_range=(np.amax(x_true) - np.amin(x_true) ), gaussian_weights=True, sigma=1.5, use_sample_covariance=False ) )
                else:
                    ssim_history.append( structural_similarity( x.reshape(self.img_shape), x_true.reshape(self.img_shape), data_range=(np.amax(x_true) - np.amin(x_true) ), gaussian_weights=True, sigma=1.5, use_sample_covariance=False ) )
                if self.img_shape is not None:
                    haar_calc, _, _ = haar_psi(x_true.reshape(self.img_shape), x.reshape(self.img_shape))
                    haar_history.append(haar_calc)

            # Gini and hoyer
            gini = gini_index(self.L @ x)
            gini_indices.append(gini)
            hoyer_index = hoyer_sparsity(self.L @ x)
            hoyer_indices.append(hoyer_index)

            # Compute k-sparsity?
            if ksparsity is not None:
                ksparsity_indices.append( ksparsity_index(self.L @ x, ksparsity) )

            if keep_solution_history:
                solution_history.append( (x, V, Lopinv ) )

            v = AtildeV @ y
            v = v - c # should this be b or c?
            u = LV @ y
            ra = (AtildeV @ y - c) # should this be b or c?
            ra = Atilde.T @ ra
            rb = LV @ y # is this wrong?
            #rb = w*(LV @ y)
            # rb = (LV @ y) * wr
            r = ra + lambdah * rb
            r = r - V @ (V.T @ r)
            r = r - V @ (V.T @ r)
            normed_r = r / la.norm(r) 
            vn = r / np.linalg.norm(r)
            V = np.column_stack((V, vn))
            residual_history.append(la.norm(r))
            dim_size += 1
            tot_matvec_A += 1 # matvecs with A
            tot_matvec_WLpinv += 1 # matvecs with WLpinv
        

            # Restarting
            if (dim_size == restart_dim) and restarting:

                # Reset basis
                tmp = z.copy()
                tmp /= np.linalg.norm(tmp)
                V = tmp.copy()
                V = V[:,None]
                dim_size = 1
                

            # If dim_size == kmax, need to compress the basis
            if (dim_size == max_dim) and recycling:

                # Update weights and get new QR factorization of W L V
                #LV = self.L @ V
                #tot_matvec_L += dim_size
                #WLV = W @ LV

                # Get new QR factorization for Atilde
                #AtildeV = np.hstack([ Atilde.matvec(V[:,j])[:,None] for j in range(V.shape[1])  ])
                AtildeV = np.hstack([ AtildeV, Atilde.matvec(V[:,-1])[:,None] ]) # only the last column has changed, and weights not updated yet!
                tot_matvec_A += 1 # matvecs with A
                tot_matvec_WLpinv += 1 # matvecs with WLpinv
                AA = AtildeV
                (Q_Atilde, R_Atilde) = la.qr(AA, mode='economic') 

                # New basis
                V = self.compress_priorconditioned_basis(V, R_Atilde, np.eye(R_Atilde.shape[1]), z, Q_Atilde, min_dim, lambdah)
                
                # Solve the problem over the new basis
                #dim_size = min_dim
                #AV = self.A @ V 
                #LV = self.L @ V
                #tot_matvec_A += dim_size
                #tot_matvec_L += dim_size
                #Q_A, R_A = qr(AV, mode='economic')


        # Package output
        data = {
            "x": x,
            "w": w,
            "lambdah": lambdah,
            "tot_matvec_A": tot_matvec_A,
            "tot_matvec_L": tot_matvec_L,
            "tot_matvec_WLpinv": tot_matvec_WLpinv,
            "residual_history": np.asarray(residual_history),
            "lambda_history": np.asarray(lambda_history),
            "V": V,
            "cond_ests": np.asarray(cond_ests),
            "dp_ineq_lhs": np.asarray(dp_ineq_lhs),
            "dp_ineq_rhs": np.asarray(dp_ineq_rhs),
            "dp_ineq_chs": np.asarray(dp_ineq_chs),
            "xker": xker,
            "n_completed_iterations": n_iter,
            "solution_history": solution_history,
        }

        data["hoyer_indices"] = np.asarray(hoyer_indices)
        data["gini_indices"] = np.asarray(gini_indices)

        if x_true is not None:
            data["rre_history"] = np.asarray(rre_history)
            data["ssim_history"] = np.asarray(ssim_history)
            data["haar_history"] = np.asarray(haar_history)


        if ksparsity is not None:
            data["ksparsity_indices"] = np.asarray(ksparsity_indices)

        if self.K is not None:
            data["Lopinv"] = Lopinv
        else:
            data["Lopinv"] = Lpinv

        data["Lpinv"] = Lpinv
 
        # Close engine if need to
        if regparam == "matlab_gcv": _eng.close()

        return data
    





class MMGKS(GKSSolver):

    def __init__(self, A, L, b, noise_var=None, Lpinv_factory=None, K=None, img_shape=None):

        super().__init__(A, L, b, noise_var=noise_var, Lpinv_factory=Lpinv_factory, K=K, img_shape=img_shape)


    def update_weights(self, Lx, lambdah, weight_params):
        
        w = np.sqrt(  mm_weights( Lx, epsilon=weight_params["epsilon"], p=weight_params["q"] ) )

        return w
    



def f_tau(x, tau1, tau2):
    
    # Compute the absolute value of the input
    abs_x = np.abs(x)
    
    # Apply the piecewise function
    return np.where(abs_x >= tau1, abs_x, tau2)


class MM2019GKS(GKSSolver):

    """Uses version of MM weights given in Chung 2019.
    q corresponds to the lq norm. 
    tau1 and tau2 correspond to the thresholds.
    """


    def __init__(self, A, L, b, noise_var=None, Lpinv_factory=None, K=None, img_shape=None):

        super().__init__(A, L, b, noise_var=noise_var, Lpinv_factory=Lpinv_factory, K=K, img_shape=img_shape)


    def update_weights(self, Lx, lambdah, weight_params):
        
        w = np.sqrt(np.power(f_tau(Lx, weight_params["tau1"], weight_params["tau2"]), weight_params["q"]-2))

        return w





class IASGKS(GKSSolver):

    def __init__(self, A, L, b, noise_var=None, Lpinv_factory=None, K=None, img_shape=None):

        super().__init__(A, L, b, noise_var=noise_var, Lpinv_factory=Lpinv_factory, K=K, img_shape=img_shape)


    def update_weights(self, Lx, lambdah, weight_params):

        theta = theta_update( Lx/np.sqrt(1.0/lambdah), weight_params["hyperpr_r"], weight_params["hyperpr_beta"], 1.0)
        w = 1.0/np.sqrt(theta)

        return w
    


class VanillaGKS(GKSSolver):

    def __init__(self, A, L, b, noise_var=None, Lpinv_factory=None, K=None, img_shape=None):

        super().__init__(A, L, b, noise_var=noise_var, Lpinv_factory=Lpinv_factory, K=K, img_shape=img_shape)


    def update_weights(self, Lx, lambdah, weight_params):

        return np.ones_like(Lx)
    








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

















## MM methods for adherring to specific papers

class MM1GKS(GKSSolver):

    def __init__(self, A, L, b, noise_var=None, Lpinv_factory=None, K=None, img_shape=None):

        super().__init__(A, L, b, noise_var=noise_var, Lpinv_factory=Lpinv_factory, K=K, img_shape=img_shape)


    def update_weights(self, Lx, lambdah, weight_params):
        
        w = np.sqrt(  mm_weights( Lx, epsilon=1.0, p=weight_params["q"] ) )

        return w
    

class MM2GKS(GKSSolver):

    def __init__(self, A, L, b, noise_var=None, Lpinv_factory=None, K=None, img_shape=None):

        super().__init__(A, L, b, noise_var=noise_var, Lpinv_factory=Lpinv_factory, K=K, img_shape=img_shape)


    def update_weights(self, Lx, lambdah, weight_params):
        
        w = np.sqrt(  mm_weights( Lx, epsilon=0.01, p=weight_params["q"] ) )

        return w
    

class MM3GKS(GKSSolver):

    def __init__(self, A, L, b, noise_var=None, Lpinv_factory=None, K=None, img_shape=None):

        super().__init__(A, L, b, noise_var=noise_var, Lpinv_factory=Lpinv_factory, K=K, img_shape=img_shape)


    def update_weights(self, Lx, lambdah, weight_params):
        
        w = np.sqrt(  mm_weights( Lx, epsilon=1e-10, p=weight_params["q"] ) )

        return w
    


class MM4GKS(GKSSolver):

    def __init__(self, A, L, b, noise_var=None, Lpinv_factory=None, K=None, img_shape=None):

        super().__init__(A, L, b, noise_var=noise_var, Lpinv_factory=Lpinv_factory, K=K, img_shape=img_shape)


    def update_weights(self, Lx, lambdah, weight_params):
        
        w = np.sqrt( np.power(f_tau(Lx, 1e-10, 1e-16), weight_params["q"]-2) )

        return w
    



class MMXGKS(GKSSolver):

    def __init__(self, A, L, b, noise_var=None, Lpinv_factory=None, K=None, img_shape=None):

        super().__init__(A, L, b, noise_var=noise_var, Lpinv_factory=Lpinv_factory, K=K, img_shape=img_shape)


    def update_weights(self, Lx, lambdah, weight_params):
        
        w = np.sqrt(  mm_weights( Lx, epsilon=1e-3, p=weight_params["q"] ) )

        return w











