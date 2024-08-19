# Will have priorconditioned solvers here. Work in progress.



# from ..utilities.decompositions import *
# from ..utilities.reg_param.gcv import *
# from ..utilities.reg_param.discrepancy_principle import *
# from ..utilities.utils import *
# from scipy import sparse
# import numpy as np
# from scipy import linalg as la
# from pylops import Identity
# from trips.utilities.weights import *
# from tqdm import tqdm
# from collections.abc import Iterable

from trips.utilities.decompositions import *

from trips.utilities.reg_param.discrepancy_principle import *
from trips.utilities.utils import *

from scipy import sparse
import numpy as np
from scipy import linalg as la
from pylops import Identity
from trips.utilities.weights import *
from tqdm import tqdm
from collections.abc import Iterable

import jlinops

import warnings

from scipy.integrate import odeint

# Use the gcv codes in this repo!
from trips.utilities.reg_param.discrepancy_principle import discrepancy_principle
from trips.utilities.reg_param.gcv import generalized_crossvalidation
from regparam_selection import mgcv, dp_quadratic, dp_cubic, matlab_gcv, dp_cubic_standard, dp_cubic_standard_val, dp_val


# Try to import matlab.engine
try:
    import matlab.engine
except:
    pass # assume we don't need it
    # raise Exception("Error with matlab.engine, make sure it is installed correctly.")



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




def PGKSIAS(A, b, L, W=None, hyperpr_r=-1.0, hyperpr_beta=1.0, shift_scale=1.0, projection_dim=3, n_iter=5, regparam='gcv', regparam_fixed=None, keep_x=True, x_true=None, x0=None, Lpinv_factory=None, err_norm_sq=0.0, **kwargs):

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False
    isoTV_option = kwargs['isoTV'] if ('isoTV' in kwargs) else False
    GS_option = kwargs['GS'] if ('GS' in kwargs) else False
    epsilon = kwargs['epsilon'] if ('epsilon' in kwargs) else 0.1
    prob_dims = kwargs['prob_dims'] if ('prob_dims' in kwargs) else False
    non_neg = kwargs['non_neg'] if ('non_neg' in kwargs) else False
    regparam_sequence = kwargs['regparam_sequence'] if ('regparam_sequence' in kwargs) else [0.1*(0.5**(x)) for x in range(0,n_iter)]
    
    # Initialize vartheta
    if regparam_fixed is None:
         vartheta = 1.0
    else:
         vartheta = 1.0/regparam_fixed

    if x0 is None:
        x = A.T @ b 
    else:
        x = x0

    # Initial setup
    u = L @ x
    theta = theta_update( u/np.sqrt(vartheta), hyperpr_r, hyperpr_beta, 1.0)
    wr = 1.0/np.sqrt(theta)
    wr = np.squeeze(wr)
    # wr = np.ones(L.shape[0])
    Lpinv = Lpinv_factory(wr)
    if W is not None:
        AWpinv = jlinops.QRPinvOperator( jlinops.MatrixLinearOperator(A.matmat(W.A)) )
        Lopinv = ( jlinops.IdentityOperator( (A.shape[0], A.shape[0])  ) - W @ (AWpinv @ A)  ) @ Lpinv
        Atilde = A @ Lopinv
    else:
        Atilde = A @ Lpinv

    if W is not None:
         xker = W @ (AWpinv @ b)
    else:
         xker = np.zeros(A.shape[1])
    Axker = A @ xker
    c = b - Axker

    (U, B, V) = golub_kahan(Atilde, c, projection_dim, dp_stop, **kwargs)

    #(U, B, V) = golub_kahan(A, c, projection_dim, dp_stop, **kwargs)

    # (U, B, V) = golub_kahan(A, b, projection_dim, dp_stop, **kwargs)
    # newV = np.hstack([ L.matvec(V[:,j])[:,None] for j in range(V.shape[1])  ])
    # V = newV
    # qr_fac = np.linalg.qr(V)
    # V = qr_fac.Q

    # Atc = Atilde.T @ c
    # V = np.zeros((L.shape[0], projection_dim))
    # V[:,0] = Atc
    # tmp = Atc.copy()
    # for j in range(1,projection_dim):
    #     tmp = Atilde.T @ (Atilde @ tmp) + regparam_fixed*tmp
    #     V[:,j] = tmp.copy()


    # V = np.random.normal(size=(self.n, 5))

    # Orthogonalize
    qr_fac = np.linalg.qr(V)
    V = qr_fac.Q


    # V, _ = la.qr(V, mode='economic') 
    
    x_history = []
    lambda_history = []
    vartheta_history = []
    residual_history = []
    n_gcv_evals_history = []
    rre_history = []
    e = 1
    m = len(b)

    if x_true is not None:
        x_true_norm = np.linalg.norm(x_true)
        rre_history.append( np.linalg.norm(x - x_true)/x_true_norm )
    

    # Initialize vartheta
    if regparam_fixed is None:
         vartheta = 1.0
    else:
         vartheta = 1.0/regparam_fixed
    
    cond_ests = []

    if regparam == "matlab_gcv":
            # If this throws an error, it is because matlab engine is not set up correctly
            _eng = matlab.engine.start_matlab()


    for ii in tqdm(range(n_iter), desc='running MMGKS...'):

        #print(f"Starting iteration {ii+1}...")

        # Transform to sparse domain
        u = L @ x
     
        # Update the weights
        theta = theta_update( u/np.sqrt(vartheta), hyperpr_r, hyperpr_beta, 1.0)
        wr = 1.0/np.sqrt(theta)
        wr = np.squeeze(wr)
       
        # Build new Atilde
        Lpinv = Lpinv_factory(wr)
        if W is not None:
            Lopinv = ( jlinops.IdentityOperator( (A.shape[0], A.shape[0])  ) - W @ (AWpinv @ A)  ) @ Lpinv
            Atilde = A @ Lopinv
        else:
             Atilde = A @ Lpinv

        # Get new QR factorization for Atilde
        AtildeV = np.hstack([ Atilde.matvec(V[:,j])[:,None] for j in range(V.shape[1])  ])
        AA = AtildeV
        (Q_Atilde, R_Atilde) = la.qr(AA, mode='economic') 

        # What used to be "R_L", is now just the identity matrix.
        R_L = np.eye(V.shape[1])
        LV = V

        # Stuff for cubic DP
        beta0 = None
        dp_cubic_n_iters = []

        # Misc
        eta = 1.01


        ### Check if DP problem solution exists?
        shift = np.linalg.norm(c - (Q_Atilde @ Q_Atilde.T @ c) )**2
        tmp, _, _, _ = np.linalg.lstsq(R_Atilde, Q_Atilde.T @ c, rcond=-1)
        #tmp = np.linalg.pinv(A) @ b
        lhs = (np.linalg.norm( (R_Atilde @ tmp) - (Q_Atilde.T @ c) )**2)
        rhs = np.linalg.norm( Q_Atilde.T @ c)**2
        inside = ((eta**2)*err_norm_sq) - shift
        check1 = lhs <= inside
        check2 = inside <= rhs
        # print(lhs)
        # print(inside)
        # print(rhs)
        if check1 and check2:
            pass
        else:
            warnings.warn("Inequality is NOT satisfied, no solution to the DP problem...", UserWarning)




        # Select new regularization parameter
        if regparam == "mgcv": # Jonathan's MGCV method
            proposed_lambdah = mgcv(Q_Atilde, R_Atilde, R_L, c, **kwargs)
        elif regparam == "dp": # Trips-Py dp
            # Extra terms to account for the priorconditioning
            #shift = np.linalg.norm( c -  ( Q_Atilde @ (Q_Atilde.T @ c) ), ord=2 )**2
            #shift = -2*np.dot(b,Axker) + (np.linalg.norm(Axker, ord=2))**2
            shift = 0
            proposed_lambdah = discrepancy_principle(Q_Atilde, R_Atilde, R_L, c, delta=np.sqrt(Atilde.shape[0]), explicitProj=True, shift=shift, **kwargs)
            #print(f"proposed_lambda: {proposed_lambdah}")
        elif regparam == 'gcv': # Trips-Py gcv
            proposed_lambdah = generalized_crossvalidation(Q_Atilde, R_Atilde, R_L, c, **kwargs)
        elif regparam == "dp2": # Jonathan's 2nd-order DP code. Requires 2 LS solves per iteration.
            shift = np.linalg.norm( c -  ( Q_Atilde @ (Q_Atilde.T @ c) ), ord=2 )**2
            # Extra terms to account for the priorconditioning
            # xker = W @ AWpinv @ b
            # Axker = A @ xker
            #shift += -2*np.dot(b,Axker) + (np.linalg.norm(Axker, ord=2)**2)
            #shift = 0
            # Hacky try-except block since code is not completely debugged for small \beta!
            try:
                data = dp_quadratic(R_Atilde, np.eye(R_Atilde.shape[1]), Q_Atilde.T @ c, noise_var=1.0, shift=shift, eta=eta, m=len(b), beta0=None, **kwargs)
            except:
                # If it didn't work with initializing beta0, try again without initializing it (defaults to beta0 very small). 
                data = dp_quadratic(R_Atilde, np.eye(R_Atilde.shape[1]), Q_Atilde.T @ c, noise_var=1.0, shift=shift, eta=eta, m=len(b), beta0=None, **kwargs)
                #pass
            proposed_lambdah = data["lambda"]
            #print(data["beta_hist"])
        elif regparam == "my_dp": # Jonathan's 3rd-order DP code. Requires 3 LS solves per iteration.
            
            if ii < -2:
                proposed_lambdah = regparam_fixed
                
            else:
                
                # Extra terms to account for the priorconditioning
                #xker = W @ (AWpinv @ b)
                # xker = W.A @ (np.linalg.pinv(A @ W.A) @ b)
                # Axker = A @ xker
                # c = b - Axker
                shift = np.linalg.norm( c -  ( Q_Atilde @ (Q_Atilde.T @ c) ), ord=2 )**2 # needed for calculation w.r.t. full-scale objective
                #print(f"norm of c, squared: {np.linalg.norm(c)**2}")
                #print(f"Unscaled shift: {shift}")
                shift *= shift_scale
                #print(f"Scaled shift: {shift}")
                #eta = 1.05
                #print(f"eta_sq times m: {(eta**2)*len(b)}")
                # shift = 0
                # Hacky try-except block since code is not completely debugged for small \beta!
                try:
                    #data = dp_cubic_standard_val(R_Atilde, Q_Atilde.T @ b, Q_Atilde.T @ c, noise_var=1.0, shift=shift, m=len(b), beta0=beta0, **kwargs)
                    data = dp_cubic(R_Atilde, np.eye(R_Atilde.shape[1]), Q_Atilde.T @ c, 1.0, eta=eta, shift=shift, m=len(b), beta0=None, **kwargs)
                    #data = dp_val(R_Atilde, np.eye(R_Atilde.shape[1]), Q_Atilde.T @ c, 1.0, eta=eta, shift=shift, m=len(b), beta0=beta0, **kwargs)
                except:
                    # If it didn't work with initializing beta0, try again without initializing it (defaults to beta0 very small). 
                    #data = dp_cubic_standard_val(R_Atilde, Q_Atilde.T @ b, Q_Atilde.T @ c, noise_var=1.0, shift=shift, m=len(b), beta0=beta0, **kwargs)
                    data = dp_cubic(R_Atilde, np.eye(R_Atilde.shape[1]), Q_Atilde.T @ c, 1.0, eta=eta, shift=shift, m=len(b), beta0=None, **kwargs)
                    #data = dp_val(R_Atilde, np.eye(R_Atilde.shape[1]), Q_Atilde.T @ c, 1.0, eta=eta, shift=shift, m=len(b), beta0=None, **kwargs)
                    #pass
                solution = data["xbeta"]
                proposed_lambdah = data["lambda"]
                dp_cubic_n_iters.append(data["n_iters"])
                #print(data["beta_hist"])
        elif regparam == "matlab_gcv": # MATLAB GCV code
            proposed_lambdah = matlab_gcv( R_Atilde, R_L, Q_Atilde.T @ c, eng=_eng,  **kwargs)
        elif not isinstance(regparam, str): # hope that regparam is a scalar
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
            beta0 = np.inf


        # Get solution
        #print(lambdah)
        y, _ , _, svals = np.linalg.lstsq(np.vstack( (R_Atilde, np.sqrt(lambdah) * np.eye(R_Atilde.shape[1])) ), np.concatenate((Q_Atilde.T @ c, np.zeros((R_L.shape[0],)))),rcond=-1)
        cond_est = np.amax(svals)/np.amin(svals)
        cond_ests.append(cond_est)
        z = V @ y # Project back onto z space
        if W is not None:
            x = (Lopinv @ z) + W @ (AWpinv @ b) # Project back onto x space
        else:
             x = Lpinv @ z

        if keep_x: 
             x_history.append(x)

        if x_true is not None:
             x_true_norm = la.norm(x_true)
             #rel_err = la.norm(x - x_true) /x_true_norm
             rre_history.append( la.norm(x - x_true)/x_true_norm )
             
        if ii >= R_L.shape[0]:
            print("broke")
            break

        v = AtildeV @ y
        v = v - b
        u = LV @ y
        ra =  (AtildeV @ y - b)
        ra = Atilde.T @ ra
        rb = wr*(LV @ y)
        # rb = (LV @ y) * wr
        rb = rb
        r = ra + lambdah * rb
        r = r - V @ (V.T @ r)
        r = r - V @ (V.T @ r)
        normed_r = r / la.norm(r) 
        vn = r / np.linalg.norm(r)
        V = np.column_stack((V, vn))

        residual_history.append(la.norm(r))


    if x_true is not None:
        
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'relError': rre_history, 'Residual': residual_history, 'its': ii}
        info["vartheta_history"] = np.asarray(vartheta_history)
    else:
        info = {'xHistory': x_history, 'regParam': lambdah, 'regParam_history': lambda_history, 'Residual': residual_history, 'its': ii}
        info["vartheta_history"] = np.asarray(vartheta_history)
    
    info["V"] = V # also give out V
    info["wr"] = wr
    info["theta"] = vartheta_history[-1]*( (1.0/wr)**2 ) # Need to multiply by vartheta to undo change of variables to get true theta
    info["cond_ests"] = np.asarray(cond_ests)
    info["n_gcv_evals"] = np.asarray(n_gcv_evals_history)

    if W is not None:
        LopinvV = np.zeros((len(x), V.shape[1]))
        for j in range(V.shape[1]):
            LopinvV[:,j] = Lopinv @ V[:,j]
        info["LopinvV"] = LopinvV
    else:
        LpinvV = np.zeros((len(x), V.shape[1]))
        for j in range(V.shape[1]):
            LpinvV[:,j] = Lpinv @ V[:,j]
        info["LopinvV"] = LpinvV


    # Close engine if need to
    if regparam == "matlab_gcv": _eng.close()

    return (x, info)