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

from scipy.integrate import odeint

# Use the gcv codes in this repo!
from modified_gcv import modified_gcv, generalized_crossvalidation



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




def PGKSIAS(A, b, L, W=None, hyperpr_r=-1.0, hyperpr_beta=1.0, projection_dim=3, n_iter=5, regparam='gcv', regparam_fixed=None, keep_x=True, x_true=None, Lpinv_factory=None, **kwargs):

    dp_stop = kwargs['dp_stop'] if ('dp_stop' in kwargs) else False
    isoTV_option = kwargs['isoTV'] if ('isoTV' in kwargs) else False
    GS_option = kwargs['GS'] if ('GS' in kwargs) else False
    epsilon = kwargs['epsilon'] if ('epsilon' in kwargs) else 0.1
    prob_dims = kwargs['prob_dims'] if ('prob_dims' in kwargs) else False
    non_neg = kwargs['non_neg'] if ('non_neg' in kwargs) else False
    regparam_sequence = kwargs['regparam_sequence'] if ('regparam_sequence' in kwargs) else [0.1*(0.5**(x)) for x in range(0,n_iter)]
    
    # Initial setup
    wr = np.ones(L.shape[0])
    Lpinv = Lpinv_factory(wr)
    AWpinv = jlinops.QRPinvOperator( jlinops.MatrixLinearOperator(A.matmat(W.A)) )
    Lopinv = ( jlinops.IdentityOperator( (A.shape[0], A.shape[0])  ) - W @ (AWpinv @ A)  ) @ Lpinv
    Atilde = A @ Lopinv
    
    (U, B, V) = golub_kahan(Atilde, b, projection_dim, dp_stop, **kwargs)
    
    x_history = []
    lambda_history = []
    vartheta_history = []
    residual_history = []
    n_gcv_evals_history = []
    rre_history = []
    e = 1
    x = A.T @ b 

    # Initialize vartheta
    if regparam_fixed is None:
         vartheta = 1.0
    else:
         vartheta = 1.0/regparam_fixed
    
    cond_ests = []

    for ii in tqdm(range(n_iter), desc='running MMGKS...'):


        # Transform to sparse domain
        u = L @ x
     
        # Update the weights
        theta = theta_update( u/np.sqrt(vartheta), hyperpr_r, hyperpr_beta, 1.0)
        wr = 1.0/np.sqrt(theta)
        wr = np.squeeze(wr)
       
        # Build new Atilde
        Lpinv = Lpinv_factory(wr)
        Lopinv = ( jlinops.IdentityOperator( (A.shape[0], A.shape[0])  ) - W @ (AWpinv @ A)  ) @ Lpinv
        Atilde = A @ Lopinv

        # Get new QR factorization for Atilde
        AtildeV = np.hstack([ Atilde.matvec(V[:,j])[:,None] for j in range(V.shape[1])  ])
        AA = AtildeV
        (Q_Atilde, R_Atilde) = la.qr(AA, mode='economic') 

        # What used to be "R_L", is now just the identity matrix.
        R_L = np.eye(V.shape[1])
        LV = V

        # Update lambda and vartheta
        if regparam_fixed is None:
            if regparam == 'gcv':
                lambdah, num_gcv_evals  = generalized_crossvalidation(Q_Atilde, R_Atilde, R_L, b, **kwargs)
                n_gcv_evals_history.append(num_gcv_evals)
            elif regparam == 'mgcv':
                 lambdah = modified_gcv(Q_Atilde, R_Atilde, R_L, b, **kwargs)
            elif regparam == 'dp':
                lambdah = discrepancy_principle(Q_Atilde, R_Atilde, R_L, b, **kwargs)
            else:
                lambdah = regparam
        else:
             lambdah = regparam_fixed
        lambda_history.append(lambdah)
 
        
        # Handle vartheta
        try:
            vartheta = 1.0/lambdah
        except:
            vartheta = np.inf
        vartheta_history.append(vartheta)

        # Get solution
        y, _ , _, svals = np.linalg.lstsq(np.concatenate((R_Atilde, np.sqrt(lambdah) * R_L)), np.concatenate((Q_Atilde.T@ b, np.zeros((R_L.shape[0],)))),rcond=None)
        cond_est = np.amax(svals)/np.amin(svals)
        cond_ests.append(cond_est)
        z = V @ y # Project back onto z space
        x = (Lopinv @ z) + W @ AWpinv @ b # Project back onto x space

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

    return (x, info)