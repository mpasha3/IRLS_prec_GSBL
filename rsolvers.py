import numpy as np
import matplotlib.pyplot as plt

# Add files up one level to the path
import sys
# sys.path.append("..")

import jlinops

from matplotlib.backends.backend_pdf import PdfPages


from scipy.linalg import qr, qr_insert
from scipy.integrate import odeint
from fastprogress import progress_bar
import scipy.sparse as sps

from trips.utilities.weights import smoothed_holder_weights
from trips.utilities.reg_param.discrepancy_principle import discrepancy_principle
from trips.utilities.reg_param.gcv import generalized_crossvalidation


# Try to import matlab.engine
try:
    import matlab.engine
except:
    pass # assume we don't need it
    # raise Exception("Error with matlab.engine, make sure it is installed correctly.")


import scipy.optimize as op
from scipy.sparse.linalg import svds

from regparam_selection import mgcv, dp_cubic, matlab_gcv




class RMMGKS:
    """Implements a MMGKS solver for problems of the form
        argmin_x  || A x - b ||_2^2 + \lambda || L x ||_q^q.

    If noise_var is provided, then the problem solved is 
        argmin_x  (1/noise_var)*|| A x - b ||_2^2 + \lambda || L x ||_q^q.

    """

    def __init__(self, A, L, b, noise_var=None):

        # Bind
        self._A = A
        self.L = L
        self._b = b
        self.noise_var = noise_var
        self.n = A.shape[1]
        self.m = A.shape[0]
        self.k = L.shape[0]

        # Whiten?
        if self.noise_var is not None:
            self.noise_stdev = np.sqrt(self.noise_var)
            self.A = (1.0/self.noise_stdev)*self._A
            self.b = (1.0/self.noise_stdev)*self._b
        else:
            self.noise_stdev = None
            self.A = self._A
            self.b = self._b


    def build_initial_basis(self, dim):
        """Builds the initial basis.
        """

        Atb = self.A.T @ self.b

        V = np.zeros((self.n, dim))
        V[:,0] = Atb
        tmp = Atb.copy()
        for j in range(1,dim):
            tmp = self.A.T @ (self.A @ tmp)
            V[:,j] = tmp.copy()

        # Orthogonalize
        qr_fac = np.linalg.qr(V)
        V = qr_fac.Q

        return V


    def build_adjusted_initial_basis(self, lambdah, w, dim):
        """Builds the adjusted initial basis.
        """
        Atb = self.A.T @ self.b

        V = np.zeros((self.n, dim))
        V[:,0] = Atb
        tmp = Atb.copy()
        for j in range(1, dim):
            tmp = self.A.T @ (self.A @ tmp) + lambdah*( self.L.T @ ( w * w * (self.L @ tmp) ) )
            V[:,j] = tmp.copy()

        # Orthogonalize
        qr_fac = np.linalg.qr(V)
        V = qr_fac.Q

        return V
    

    def compress_basis(self, V, R_A, R_WL, x, Q_A, kmin, lambdah):
        """Compresses the basis using TSVD approach.
        """

        # Make H matrix
        H = np.vstack([R_A, np.sqrt(lambdah)*R_WL ])

        # Truncated SVD
        _, _, Wh = svds(H, k=kmin-1)
        W = Wh.T

        # Get new basis
        Vtilde = V @ W
        xnew = x -  Vtilde @ (Vtilde.T @ x)
        xnew /= np.linalg.norm(xnew)
        V = np.column_stack((Vtilde, xnew))

        # Orthogonalize
        qr_fac = np.linalg.qr(V)
        V = qr_fac.Q

        return V
    

    def solve(self, n_iter=100, init_dim=10, kmin=10, kmax=40, q=1.0, epsilon=1e-2, regparam="dp", x0=None, x_true=None, **kwargs):
        """Driver for MMGKS solver.
        """
        assert kmin <= (init_dim + 2), "must have kmin >= init_dim+2 in this implementation"

        # Tracking
        tot_matvec_A = 0
        tot_matvec_L = 0 
        lambda_history = []
        residual_history = []
        rre_history = []
        cond_ests = []

        # Initialization
        if x0 is None:
            x = self.A.T @ self.b
        else:
            x = x0
        
        if x_true is not None:
            x_true_norm = np.linalg.norm(x_true)
            rre_history.append( np.linalg.norm(x - x_true)/x_true_norm )

        # Build initial basis and QR factor of AV
        V = self.build_initial_basis(init_dim)
        dim_size = init_dim
        AV = self.A @ V 
        tot_matvec_A += 2*init_dim
        tot_matvec_A += init_dim
        Q_A, R_A = qr(AV, mode='economic')

        # Get initial LV
        LV = self.L @ V
        tot_matvec_L += init_dim

        # Stuff for cubic DP
        beta0 = None
        dp_cubic_n_iters = []

        # Start matlab engine if regparam == matlab_gcv
        if regparam == "matlab_gcv":
            # If this throws an error, it is because matlab engine is not set up correctly
            _eng = matlab.engine.start_matlab()

        for ii in progress_bar(range(n_iter)):

            # Update weights and get new QR factorization of W L V
            Lx = self.L @ x
            tot_matvec_L += 1
            w = smoothed_holder_weights( Lx, epsilon=epsilon, p=q)
            W = sps.diags(w)
            WLV = W @ LV
            Q_WL, R_WL = qr(WLV, mode='economic') 

            # Select new regularization parameter
            if regparam == "mgcv": # Jonathan's MGCV method
                proposed_lambdah = mgcv(Q_A, R_A, R_WL, self.b, **kwargs)
            elif regparam == "dp": # Trips-Py dp
                proposed_lambdah = discrepancy_principle(Q_A, R_A, R_WL, self.b, delta=1.0, **kwargs)
            elif regparam == 'gcv': # Trips-Py gcv
                proposed_lambdah = generalized_crossvalidation(Q_A, R_A, R_WL, self.b, **kwargs)
            elif regparam == "dp3": # Jonathan's 3rd-order DP code
                shift = np.linalg.norm( self.b -  ( Q_A @ (Q_A.T @ self.b) ), ord=2 )**2 # needed for calculation w.r.t. full-scale objective
                # Hacky try-except block since code is not completely debugged for small \beta!
                try:
                    data = dp_cubic(R_A, R_WL, Q_A.T @ self.b, noise_var=1.0, shift=shift, m=len(self.b), beta0=beta0, **kwargs)
                except:
                    pass
                proposed_lambdah = data["lambda"]
                dp_cubic_n_iters.append(data["n_iters"])
            elif regparam == "matlab_gcv": # MATLAB GCV code
                proposed_lambdah = matlab_gcv( R_A, R_WL, Q_A.T @ self.b, eng=_eng,  **kwargs)
            elif np.isscalar(regparam): # hope that regparam is a scalar
                proposed_lambdah = regparam
            else:
                raise NotImplementedError

            # Screen the proposed lambdah somehow? Just accept for now...
            lambdah = proposed_lambdah
            lambda_history.append(lambdah)
            try:
                beta0 = 1.0/lambdah
            except:
                beta0 = 1e10


            # Obtain new solution
            y, _ , _, svals = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_WL)), np.concatenate((Q_A.T @ self.b, np.zeros((R_WL.shape[0],)))), rcond=None)
            cond_est = np.amax(svals)/np.amin(svals)
            cond_ests.append(cond_est)
            x = V @ y

            # Compute new error?
            if x_true is not None: rre_history.append( np.linalg.norm(x - x_true)/x_true_norm )

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
            tot_matvec_A += 2
            tot_matvec_L += 2
            residual_history.append(np.linalg.norm(r))
            dim_size += 1

            # Update QR factor for A
            #Q_A, R_A = qr_insert( Q_A, R_A, u = Avn, k=-1, which="col" )
            Q_A, R_A = qr(AV, mode='economic')             

            # If dim_size == kmax, need to compress the basis
            if dim_size == kmax:

                # Update weights and get new QR factorization of W L V
                LV = self.L @ V
                tot_matvec_L += dim_size
                WLV = W @ LV
                Q_WL, R_WL = qr(WLV, mode='economic') 
                Q_L, R_L = qr(LV, mode='economic') 
                
                # New basis
                V = self.compress_basis(V, R_A, R_WL, x, Q_A, kmin, lambdah)
                
                # Solve the problem over the new basis
                dim_size = kmin
                AV = self.A @ V 
                LV = self.L @ V
                tot_matvec_A += dim_size
                tot_matvec_L += dim_size
                Q_A, R_A = qr(AV, mode='economic')
             
        # Package output
        data = {
            "x": x,
            "w": w,
            "tot_matvec_A": tot_matvec_A,
            "tot_matvec_L": tot_matvec_L,
            "residual_history": np.asarray(residual_history),
            "lambda_history": np.asarray(lambda_history),
            "V": V,
            "cond_ests": np.asarray(cond_ests),
            "dp_cubic_n_iters": np.asarray(dp_cubic_n_iters)
        }

        if x_true is not None:
            data["rre_history"] = np.asarray(rre_history)

        # Close engine if need to
        if regparam == "matlab_gcv": _eng.close()
           
        return data
















def conditional_mode_ode_rhs(varphi, z, r):
        """RHS of the ODE used for updating theta.
        """
        
        dvarphidz = (2*z*varphi)/((2*(r**2)*((varphi)**(r+1))) + (z**2))

        return dvarphidz


def theta_update(x, r, beta, vartheta):
    """Returns the theta-update for fixed x (Rx).
    """

    eta = r*beta - 1.5
    initial_value = (eta/r)**(1/r)

    # Rescale
    z = np.abs(  ( x )  / np.sqrt(vartheta) )

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







class RIASGKS:
    """Implements a IASGKS solver for problems of the form
        argmin_x  || A x - b ||_2^2 + \lambda || L x ||_q^q.

    If noise_var is provided, then the problem solved is 
        argmin_x  (1/noise_var)*|| A x - b ||_2^2 + \lambda || L x ||_q^q.

    """

    def __init__(self, A, L, b, noise_var=None):

        # Bind
        self._A = A
        self.L = L
        self._b = b
        self.noise_var = noise_var
        self.n = A.shape[1]
        self.m = A.shape[0]
        self.k = L.shape[0]

        # Whiten?
        if self.noise_var is not None:
            self.noise_stdev = np.sqrt(self.noise_var)
            self.A = (1.0/self.noise_stdev)*self._A
            self.b = (1.0/self.noise_stdev)*self._b
        else:
            self.noise_stdev = None
            self.A = self._A
            self.b = self._b


    def build_initial_basis(self, dim):
        """Builds the initial basis.
        """

        Atb = self.A.T @ self.b

        V = np.zeros((self.n, dim))
        V[:,0] = Atb
        tmp = Atb.copy()
        for j in range(1,dim):
            tmp = self.A.T @ (self.A @ tmp)
            V[:,j] = tmp.copy()

        # Orthogonalize
        qr_fac = np.linalg.qr(V)
        V = qr_fac.Q

        return V


    def build_adjusted_initial_basis(self, lambdah, w, dim):
        """Builds the adjusted initial basis.
        """
        Atb = self.A.T @ self.b

        V = np.zeros((self.n, dim))
        V[:,0] = Atb
        tmp = Atb.copy()
        for j in range(1, dim):
            tmp = self.A.T @ (self.A @ tmp) + lambdah*( self.L.T @ ( w * w * (self.L @ tmp) ) )
            V[:,j] = tmp.copy()

        # Orthogonalize
        qr_fac = np.linalg.qr(V)
        V = qr_fac.Q

        return V
    

    def compress_basis(self, V, R_A, R_WL, x, Q_A, kmin, lambdah):
        """Compresses the basis using TSVD approach.
        """

        # Make H matrix
        H = np.vstack([R_A, np.sqrt(lambdah)*R_WL ])

        # Truncated SVD
        _, _, Wh = svds(H, k=kmin-1)
        W = Wh.T

        # Get new basis
        Vtilde = V @ W
        xnew = x -  Vtilde @ (Vtilde.T @ x)
        xnew /= np.linalg.norm(xnew)
        V = np.column_stack((Vtilde, xnew))

        # Orthogonalize
        qr_fac = np.linalg.qr(V)
        V = qr_fac.Q

        return V
    

    def solve(self, n_iter=100, init_dim=10, kmin=10, kmax=40, hyperpr_r=-1.0, hyperpr_beta=1.0, regparam="dp", x0=None, x_true=None, **kwargs):
        """Driver for MMGKS solver.
        """

        # Tracking
        tot_matvec_A = 0
        tot_matvec_L = 0 
        lambda_history = []
        vartheta_history = []
        residual_history = []
        rre_history = []
        cond_ests = []

        # Initialization
        if x0 is None:
            x = self.A.T @ self.b
        else:
            x = x0
        
        if x_true is not None:
            x_true_norm = np.linalg.norm(x_true)
            rre_history.append( np.linalg.norm(x - x_true)/x_true_norm )


        # Build initial basis and QR factor of AV
        V = self.build_initial_basis(init_dim)
        dim_size = init_dim
        AV = self.A @ V 
        tot_matvec_A += 2*init_dim
        tot_matvec_A += init_dim
        Q_A, R_A = qr(AV, mode='economic') 

        # Get initial LV
        LV = self.L @ V
        tot_matvec_L += init_dim
        lambdah = 1.0
        vartheta = 1.0/lambdah

        # Stuff for cubic DP
        beta0 = None
        dp_cubic_n_iters = []

        # Start matlab engine if regparam == matlab_gcv
        if regparam == "matlab_gcv":
            # If this throws an error, it is because matlab engine is not set up correctly
            _eng = matlab.engine.start_matlab()


        for ii in progress_bar(range(n_iter)):

            # Update weights and get new QR factorization of W L V
            Lx = self.L @ x
            tot_matvec_L += 1
            theta = theta_update( Lx/np.sqrt(vartheta), hyperpr_r, hyperpr_beta, 1.0)
            w = 1.0/np.sqrt(theta)
            w = np.squeeze(w)
            W = sps.diags(w)
            WLV = W @ LV
            Q_WL, R_WL = qr(WLV, mode='economic') 


            # Select new regularization parameter
            if regparam == "mgcv": # Jonathan's MGCV method
                proposed_lambdah = mgcv(Q_A, R_A, R_WL, self.b, **kwargs)
            elif regparam == "dp": # Trips-Py dp
                proposed_lambdah = discrepancy_principle(Q_A, R_A, R_WL, self.b, delta=1.0, **kwargs)
            elif regparam == 'gcv': # Trips-Py gcv
                proposed_lambdah = generalized_crossvalidation(Q_A, R_A, R_WL, self.b, **kwargs)
            elif regparam == "dp3": # Jonathan's 3rd-order DP code
                shift = np.linalg.norm( self.b -  ( Q_A @ (Q_A.T @ self.b) ), ord=2 )**2 # needed for calculation w.r.t. full-scale objective
                # Hacky try-except block since code is not completely debugged for small \beta!
                try:
                    data = dp_cubic(R_A, R_WL, Q_A.T @ self.b, noise_var=1.0, shift=shift, m=len(self.b), beta0=beta0, **kwargs)
                except:
                    pass
                proposed_lambdah = data["lambda"]
                dp_cubic_n_iters.append(data["n_iters"])
            elif regparam == "matlab_gcv": # MATLAB GCV code
                proposed_lambdah = matlab_gcv( R_A, R_WL, Q_A.T @ self.b, eng=_eng, **kwargs)
            elif np.isscalar(regparam): # hope that regparam is a scalar
                proposed_lambdah = regparam
            else:
                raise NotImplementedError
            
            

            # Screen the proposed lambdah somehow? Just accept for now...
            lambdah = proposed_lambdah
            lambda_history.append(lambdah)
            # Handle vartheta
            try:
                vartheta = 1.0/lambdah
            except:
                vartheta = np.inf
            vartheta_history.append(vartheta)
            try:
                beta0 = 1.0/lambdah
            except:
                beta0 = 1e10


            # Obtain new solution
            y, _ , _, svals = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_WL)), np.concatenate((Q_A.T @ self.b, np.zeros((R_WL.shape[0],)))), rcond=None)
            cond_est = np.amax(svals)/np.amin(svals)
            cond_ests.append(cond_est)
            x = V @ y

            # Compute new error?
            if x_true is not None: rre_history.append( np.linalg.norm(x - x_true)/x_true_norm )

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
            tot_matvec_A += 2
            tot_matvec_L += 2
            residual_history.append(np.linalg.norm(r))
            dim_size += 1

            # Update QR factor for A
            #Q_A, R_A = qr_insert( Q_A, R_A, u = Avn, k=-1, which="col" )
            Q_A, R_A = qr(AV, mode='economic')             


            # If dim_size == kmax, need to compress the basis
            if dim_size == kmax:

                # Update weights and get new QR factorization of W L V
                LV = self.L @ V
                tot_matvec_L += dim_size
                WLV = W @ LV
                Q_WL, R_WL = qr(WLV, mode='economic') 
                Q_L, R_L = qr(LV, mode='economic') 
                
                # New basis
                V = self.compress_basis(V, R_A, R_WL, x, Q_A, kmin, lambdah)
                
                # Solve the problem over the new basis
                dim_size = kmin
                AV = self.A @ V 
                LV = self.L @ V
                tot_matvec_A += dim_size
                tot_matvec_L += dim_size
                Q_A, R_A = qr(AV, mode='economic')
             
        # Package output
        data = {
            "x": x,
            "w": w,
            "theta": vartheta_history[-1]*( (1.0/w)**2 ),
            "tot_matvec_A": tot_matvec_A,
            "vartheta": vartheta_history[-1],
            "vartheta_history": np.asarray(vartheta_history),
            "tot_matvec_L": tot_matvec_L,
            "residual_history": np.asarray(residual_history),
            "lambda_history": np.asarray(lambda_history),
            "V": V,
            "cond_ests": np.asarray(cond_ests),
            "dp_cubic_n_iters": np.asarray(dp_cubic_n_iters),
        }

        if x_true is not None:
            data["rre_history"] = np.asarray(rre_history)

        # Close engine if need to
        if regparam == "matlab_gcv": _eng.close()

        return data
    


