import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import warnings

from modified_gcv import gcv_numerator, gcv_denominator

from scipy.misc import derivative
from scipy.optimize import root_scalar

import matplotlib.pyplot as plt

import os
import sys

# Try to import matlab.engine
try:
    import matlab.engine
    
    # Add the directory containing gcv.m to the MATLAB path
    SCRIPT_DIR = os.path.join( os.path.dirname(os.path.abspath(__file__)), "mscripts" )

except:
    pass # assume we don't need it
    # raise Exception("Error with matlab.engine, make sure it is installed correctly.")




def find_last_local_min(arr):
    """Helper function for finding the rightmost local minimima of GCV/MGCV.
    """
    # Ensure the input is a NumPy array
    arr = np.asarray(arr)
    
    # Iterate from the end of the array to the beginning
    for i in range(len(arr) - 2, 0, -1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            return i
    
    # Check the edge cases for the first and last elements
    if len(arr) > 1 and arr[0] < arr[1]:
        return 0
    if len(arr) > 1 and arr[-1] < arr[-2]:
        return len(arr) - 1
    
    # Return -1 if no local minimum is found
    return -1



def mgcv(Q_A, R_A, R_WL, b, gcv_gamma=0.1):
    """Chooses a new lambdah parameter using cross validation. If plotting=True is passed,
    also makes a plot of the GCV functional.
    """

    # Build objective function (in terms of log lambda)
    def gcv_func(log_reg_param):
        inverted, _ , _, svals = np.linalg.lstsq(np.concatenate((R_A, np.sqrt( np.power(10, log_reg_param) ) * R_WL)), np.vstack( (  np.eye( R_A.T.shape[1] )  , np.zeros((R_WL.shape[0],R_A.T.shape[1]) ) ) ),rcond=-1)
        Z = R_A @ inverted
        fac = gcv_gamma + (1.0 - gcv_gamma)*(1.0/R_A.shape[0])*np.trace( Z @ Z )
        cond_est = np.amax(svals)/np.amin(svals)

        numerator = gcv_numerator(np.power(10, log_reg_param), Q_A, R_A, R_WL, b)
        denominator = gcv_denominator(np.power(10,log_reg_param), R_A, R_WL, b ) 

        return np.log(fac) + np.log( numerator ) - np.log( denominator )

    # Compute a solution
    log_lambdah_star = op.fminbound(func = gcv_func, x1 = -12, x2 = 12, args=(), xtol=1e-12, maxfun=1000, full_output=0, disp=0) ## should there be tol here?
    log_lambdah_star_eval = gcv_func(log_lambdah_star)
    lambdah_star = np.power(10,log_lambdah_star)

    # Let's not use the rightmost local minimizer below; we will not even use global minimization, just take the solution found above.

    # log_lambdahs = np.linspace(-12, 12, 100)
    # evals = []
    # for log_lambdah in log_lambdahs:
    #     evals.append( gcv_func(log_lambdah) )

    # Using the rightmost local minimizer criterion?
    # idx = find_last_local_min(evals)
    # log_lambdah_star = log_lambdahs[idx]
    # log_lambdah_star_eval = evals[idx]
    # lambdah_star = np.power(10, log_lambdah_star)

    return lambdah_star






def dp_cubic(A, L, b, noise_var, eta=1.05, maxiter=50, beta0=None, beta_min=1e-8, rtol=1e-6, atol=1e-16, early_stopping=True, shift=0.0, m=None, warmstarting=True, err_norm_sq=None, xker=None):
    """Implements an efficient cubically-convergent root-finder for finding a 
    regularization parameter satisfying the discrepancy principle, for a problem
    of the form

        argmin_x (1.0/noise_var)*|| A x - b ||_2^2 + lambda*|| L x ||_2^2
        
    Each iteration requires the solution of 3 least-squares problems. We assume 
    that A and L are passed as dense matrices. The method is guaranteed to converge 
    under weak assumptions.

    The optimization is done over beta in lambda = 1/beta, instead of lambda itself.
    This guarantees monotonicity/convexity properties of the objective.

    eq_norm_sq will be "whitened" according to noise_var.
    """

    # Whiten the variables
    A = (1.0/np.sqrt(noise_var))*A
    b = (1.0/np.sqrt(noise_var))*b
    if err_norm_sq is not None: err_norm_sq = err_norm_sq/noise_var

    # Misc
    Atb = A.T @ b
    if m is not None:
        pass
    else:
        m = len(b)

    # If err_norm_sq is passed, print whether or not we satisfy || x_0 - b ||_2^2 < (eta**2)*err_norm_sq - shift < || b ||_2^2.
    if err_norm_sq is not None:
        # Get x_0 
        tmp, _, _, _ = np.linalg.lstsq(A, b, rcond=-1)
        #tmp = np.linalg.pinv(A) @ b
        lhs = (np.linalg.norm( (A @ tmp) - b)**2)
        rhs = np.linalg.norm(b - A @ xker)**2
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


    if (beta0 is None) or (not warmstarting):
        # Initialize at beta = 0, use analytic formulas for phi and its derivatives
        beta = 0.0
        phi = (np.linalg.norm(b)**2) - (eta**2)*m + shift
        tmp, _, _, _ = np.linalg.lstsq(L.T, Atb, rcond=-1)
        phi_prime = -2*(np.linalg.norm(tmp)**2)
        tmp2, _, _, _ = np.linalg.lstsq(L, tmp, rcond=-1)
        phi_prime_prime = 2*(np.linalg.norm(tmp2)**2)
    else:
        # Otherwise begin the iteration at the provided beta0
        beta = beta0

        ### Update phi and derivatives
        xbeta, _, _, _ = np.linalg.lstsq( np.vstack([ A, np.sqrt(1.0/beta)*L ]), np.hstack([ b, np.zeros(L.shape[0]) ]) , rcond=-1  )
        Axbeta = A @ xbeta
        Lxbeta = L @ xbeta
        phi = (np.linalg.norm(Axbeta - b)**2) - (eta**2)*m + shift

        # phi prime
        term1 = - (2/(beta**2))*(np.linalg.norm(Lxbeta)**2)
        QinvLtLxbeta, _, _, _ = np.linalg.lstsq( np.vstack([ A, np.sqrt(1.0/beta)*L ]), np.hstack([ np.zeros(A.shape[0]),  np.sqrt(beta)*Lxbeta  ]) , rcond=-1  )
        AtAxbeta= A.T @ (Axbeta)
        term2 = (2/(beta**2))*np.dot(QinvLtLxbeta, AtAxbeta )
        phi_prime = term1 + term2
        
        # phi prime prime
        term1 = -(2/(beta**4))*np.dot( QinvLtLxbeta, L.T @ Lxbeta )
        AtAQinvLtLxbeta = A.T @ (A @ QinvLtLxbeta)
        term2 = (2/(beta**4))*np.dot(QinvLtLxbeta, AtAQinvLtLxbeta )
        QinvAtAxbeta, _, _, _ = np.linalg.lstsq( np.vstack([ A, np.sqrt(1.0/beta)*L ]), np.hstack([ Axbeta, np.zeros(L.shape[0]) ]) , rcond=-1  )
        term3 =  (4/(beta**4))*np.dot( L @ QinvAtAxbeta, L @ QinvLtLxbeta )
        term4 = -(2/(beta**4))*np.dot( L @ xbeta, L @ QinvLtLxbeta )
        term5 = -(4/(beta**3))*np.dot(QinvAtAxbeta, L.T @ Lxbeta)
        term6 = +(4/(beta**3))*np.dot(Lxbeta, Lxbeta)
        phi_prime_prime = term1 + term2 + term3 + term4 + term5 + term6


    # Setup for iteration
    beta_hist = [beta]
    phi_hist = [phi]
    n_iters = 0

    # Check if we already satisfy the absolute tolerance criteria?
    if np.abs(phi) < atol:
        pass
    else: # otherwise do the iteration

        for j in range(maxiter):

            # Convergence criteria based on relative change in root
            if j < 2:
                pass
            else:
                if early_stopping:

                    if ((beta_hist[-1] - beta_hist[-2])/beta_hist[-2]) < rtol:
                        break

                    if np.abs(phi) < atol:
                        break

            ### Take step
            mu = beta + phi_prime/(2*phi_prime_prime)
            alpha = 2*phi_prime*np.sqrt(beta - mu)
            gamma = phi - alpha*np.sqrt(beta - mu)
            beta = mu + ((gamma/alpha)**2)
            beta = max(beta, beta_min)
            #print(beta)
            beta_hist.append(beta)

            ### Update phi and derivatives
            # print(beta)
            # print(np.linalg.matrix_rank(A))
            # print(np.linalg.matrix_rank(np.sqrt(1.0/beta)*L))
            # print(np.linalg.matrix_rank(np.vstack([ A, np.sqrt(1.0/beta)*L ])))
            # print()
            xbeta, _, _, _ = np.linalg.lstsq( np.vstack([ A, np.sqrt(1.0/beta)*L ]), np.hstack([ b, np.zeros(L.shape[0]) ]) , rcond=-1  )
            Axbeta = A @ xbeta
            Lxbeta = L @ xbeta
            phi = (np.linalg.norm(Axbeta - b)**2) - (eta**2)*m + shift
            phi_hist.append(phi)

            # phi prime
            term1 = - (2/(beta**2))*(np.linalg.norm(Lxbeta)**2)
            QinvLtLxbeta, _, _, _ = np.linalg.lstsq( np.vstack([ A, np.sqrt(1.0/beta)*L ]), np.hstack([ np.zeros(A.shape[0]),  np.sqrt(beta)*Lxbeta  ]) , rcond=-1  )
            #term2 = (2/(beta**2))*np.dot( QinvLtLxbeta, A.T @ Axbeta  )
            AtAxbeta = A.T @ (Axbeta)
            term2 = (2/(beta**2))*np.dot(QinvLtLxbeta, AtAxbeta )
            phi_prime = term1 + term2
            
            # phi prime prime
            term1 = -(2/(beta**4))*np.dot( QinvLtLxbeta, L.T @ Lxbeta )
            AtAQinvLtLxbeta = A.T @ (A @ QinvLtLxbeta)
            term2 = (2/(beta**4))*np.dot(QinvLtLxbeta, AtAQinvLtLxbeta )
            QinvAtAxbeta, _, _, _ = np.linalg.lstsq( np.vstack([ A, np.sqrt(1.0/beta)*L ]), np.hstack([ Axbeta, np.zeros(L.shape[0]) ]), rcond=-1  )
            term3 =  (4/(beta**4))*np.dot( L @ QinvAtAxbeta, L @ QinvLtLxbeta )
            term4 = -(2/(beta**4))*np.dot( L @ xbeta, L @ QinvLtLxbeta )
            term5 = -(4/(beta**3))*np.dot(QinvAtAxbeta, L.T @ Lxbeta)
            term6 = +(4/(beta**3))*np.dot(Lxbeta, Lxbeta)
            phi_prime_prime = term1 + term2 + term3 + term4 + term5 + term6

            # Increase counter
            n_iters += 1

  
    data = {
        "beta": beta,
        "xbeta": xbeta,
        "lambda": 1.0/beta,
        "beta_hist": np.asarray(beta_hist),
        "phi_hist": np.asarray(phi_hist),
        "n_iters": n_iters,
    }
    
    return data






def dp_val(A, L, b, noise_var, eta=1.05, maxiter=25, beta0=None, rtol=1e-5, atol=1e-10, early_stopping=True, shift=0.0, m=None, warmstarting=True):
    """Implements an efficient cubically-convergent root-finder for finding a 
    regularization parameter satisfying the discrepancy principle, for a problem
    of the form

        argmin_x (1.0/noise_var)*|| A x - b ||_2^2 + lambda*|| L x ||_2^2
        
    Each iteration requires the solution of 3 least-squares problems. We assume 
    that A and L are passed as dense matrices. The method is guaranteed to converge 
    under weak assumptions.

    The optimization is done over beta in lambda = 1/beta, instead of lambda itself.
    This guarantees monotonicity/convexity properties of the objective.
    """

    # print(eta)
    # print(noise_var)
    # print(m)
    # print(shift)

    # Whiten the variables
    A = (1.0/np.sqrt(noise_var))*A
    b = (1.0/np.sqrt(noise_var))*b

    # Misc
    Atb = A.T @ b
    if m is not None:
        pass
    else:
        m = len(b)


    def _phi(beta):
        xbeta, _, _, _ = np.linalg.lstsq( np.vstack([ A, np.sqrt(1.0/beta)*L ]), np.hstack([ b, np.zeros(L.shape[0]) ]) , rcond=-1  )
        Axbeta = A @ xbeta
        Lxbeta = L @ xbeta
        phi = (np.linalg.norm(Axbeta - b)**2) - (eta**2)*m + shift
        return phi
    
    result = root_scalar(_phi, bracket=(1e-12, 1e10))
    beta = result.root
    xbeta, _, _, _ = np.linalg.lstsq( np.vstack([ A, np.sqrt(1.0/beta)*L ]), np.hstack([ b, np.zeros(L.shape[0]) ]) , rcond=-1  )
    
    data = {
        "beta": beta,
        "xbeta": xbeta,
        "lambda": 1.0/beta,
        "beta_hist": None,
        "phi_hist": None,
        "n_iters": None,
    }
    
    return data



























































def dp_cubic_standard(A, b, c, noise_var, eta=1.1, maxiter=25, beta0=None, rtol=1e-5, atol=1e-10, early_stopping=True, shift=0.0, m=None, warmstarting=True):
    """
    """

    # Whiten the variables
    A = (1.0/np.sqrt(noise_var))*A
    b = (1.0/np.sqrt(noise_var))*b
    c = (1.0/np.sqrt(noise_var))*c
    
    # Misc
    Atb = A.T @ b
    AAtb = A @ Atb
    Atc = A.T @ c
    I = np.eye(A.shape[1])
    if m is not None:
        pass
    else:
        m = len(b)

    if (beta0 is None) or (not warmstarting):
        # Initialize at beta = 0, use analytic formulas for phi and its derivatives
        beta = 0.0
        phi = (np.linalg.norm(c)**2) - (eta**2)*m + shift
        phi_prime = -2*np.dot( Atb, Atc )
        phi_prime_prime = 2*np.dot(AAtb, AAtb)
        print(f"Starting with beta = {beta}")
    else:
        # Otherwise begin the iteration at the provided beta0
        beta = beta0
        print(f"Starting with beta = {beta}")

        ### Update phi and derivatives
        xbeta, _, _, _ = np.linalg.lstsq( np.vstack([ A, np.sqrt(1.0/beta)*I ]), np.hstack([ b, np.zeros(I.shape[0]) ]) , rcond=-1  )
        Axbeta = A @ xbeta
        phi = (np.linalg.norm( Axbeta - c )**2) - (eta**2)*m + shift

        # phi prime
        Qinvxbeta, _, _, _ = np.linalg.lstsq( np.vstack([ A, np.sqrt(1.0/beta)*I ]), np.hstack([ np.zeros(A.shape[0]),  np.sqrt(beta)*xbeta  ]) , rcond=-1  )
        diff = Axbeta - c
        AQinvxbeta = A @ Qinvxbeta
        phi_prime = (2/(beta**2))*np.dot( AQinvxbeta, diff)

        # phi prime prime
        QinvQinvxbeta, _, _, _ = np.linalg.lstsq( np.vstack([ A, np.sqrt(1.0/beta)*I ]), np.hstack([ np.zeros(A.shape[0]),  np.sqrt(beta)*Qinvxbeta  ]) , rcond=-1  )
        AQinvQinvxbeta = A @ QinvQinvxbeta
        term1 = (4/(beta**4))*np.dot(AQinvQinvxbeta, Axbeta)
        term2 = -(4/(beta**3))*np.dot( AQinvxbeta, Axbeta )
        term3 = (2/(beta**4))*np.dot(AQinvxbeta, AQinvxbeta)
        term4 = -(4/(beta**4))*np.dot( QinvQinvxbeta, Atc )
        term5 = (4/(beta**3))*np.dot( Qinvxbeta, Atc )
        phi_prime_prime = term1 + term2 + term3 + term4 + term5


    # Setup for iteration
    beta_hist = [beta]
    phi_hist = [phi]
    n_iters = 0

    # Check if we already satisfy the absolute tolerance criteria?
    if np.abs(phi) < atol:
        pass
    else: # otherwise do the iteration

        for j in range(maxiter):

            # Convergence criteria based on relative change in root
            if j < 2:
                pass
            else:
                if early_stopping:

                    if ((beta_hist[-1] - beta_hist[-2])/beta_hist[-2]) < rtol:
                        break

                    if np.abs(phi) < atol:
                        break

            ### Take step
            mu = beta + phi_prime/(2*phi_prime_prime)
            alpha = 2*phi_prime*np.sqrt(beta - mu)
            gamma = phi - alpha*np.sqrt(beta - mu)
            beta = mu + ((gamma/alpha)**2)
            print(beta)
            beta_hist.append(beta)

            ### Update phi and derivatives
            xbeta, _, _, _ = np.linalg.lstsq( np.vstack([ A, np.sqrt(1.0/beta)*I ]), np.hstack([ b, np.zeros(I.shape[0]) ]) , rcond=-1  )
            Axbeta = A @ xbeta
            phi = (np.linalg.norm( Axbeta - c )**2) - (eta**2)*m + shift

            # phi prime
            Qinvxbeta, _, _, _ = np.linalg.lstsq( np.vstack([ A, np.sqrt(1.0/beta)*I ]), np.hstack([ np.zeros(A.shape[0]),  np.sqrt(beta)*xbeta  ]) , rcond=-1  )
            diff = Axbeta - c
            AQinvxbeta = A @ Qinvxbeta
            phi_prime = (2/(beta**2))*np.dot( AQinvxbeta, diff)

            # phi prime prime
            QinvQinvxbeta, _, _, _ = np.linalg.lstsq( np.vstack([ A, np.sqrt(1.0/beta)*I ]), np.hstack([ np.zeros(A.shape[0]),  np.sqrt(beta)*Qinvxbeta  ]) , rcond=-1  )
            AQinvQinvxbeta = A @ QinvQinvxbeta
            term1 = (4/(beta**4))*np.dot(AQinvQinvxbeta, Axbeta)
            term2 = -(4/(beta**3))*np.dot( AQinvxbeta, Axbeta )
            term3 = (2/(beta**4))*np.dot(AQinvxbeta, AQinvxbeta)
            term4 = -(4/(beta**4))*np.dot( QinvQinvxbeta, Atc )
            term5 = (4/(beta**3))*np.dot( Qinvxbeta, Atc )
            phi_prime_prime = term1 + term2 + term3 + term4 + term5

            # Increase counter
            n_iters += 1

  
    data = {
        "beta": beta,
        "xbeta": xbeta,
        "lambda": 1.0/beta,
        "beta_hist": np.asarray(beta_hist),
        "phi_hist": np.asarray(phi_hist),
        "n_iters": n_iters,
    }
    
    return data







def dp_cubic_standard_val(A, b, c, noise_var, eta=1.1, order=3, dx=1e-3, maxiter=25, beta0=None, rtol=1e-5, atol=1e-10, early_stopping=True, shift=0.0, m=None, warmstarting=True):
    """
    """

    # Whiten the variables
    A = (1.0/np.sqrt(noise_var))*A
    b = (1.0/np.sqrt(noise_var))*b
    c = (1.0/np.sqrt(noise_var))*c
    b = c

    # Misc
    Atb = A.T @ b
    AAtb = A @ Atb
    Atc = A.T @ c
    I = np.eye(A.shape[1])
    if m is not None:
        pass
    else:
        m = len(b)

    # Build func
    def _phi(beta):
        xbeta, _, _, _ = np.linalg.lstsq( np.vstack([ A, np.sqrt(1.0/beta)*I ]), np.hstack([ b, np.zeros(I.shape[0]) ]) , rcond=-1  )
        Axbeta = A @ xbeta
        return (np.linalg.norm(Axbeta - b)**2) - (eta**2)*m + shift

    result = root_scalar(_phi, bracket=(1e-10, 1e10))
    beta = result.root

    data = {
        "beta": beta,
        "lambda": 1.0/beta,
        "beta_hist": np.asarray([1]),
        "phi_hist": np.asarray([1]),
        "n_iters": 1,
    }

    return data
    
        

    # if (beta0 is None) or (not warmstarting):
    #     # Initialize at beta = 0, use analytic formulas for phi and its derivatives
    #     beta = 0.0
    #     phi = (np.linalg.norm(c)**2) - (eta**2)*m + shift
    #     phi_prime = -2*np.dot( Atb, Atc )
    #     phi_prime_prime = 2*np.dot(AAtb, AAtb)
    #     print(f"Starting with beta = {beta}")
    # else:
    #     # Otherwise begin the iteration at the provided beta0
    #     beta = beta0
    #     print(f"Starting with beta = {beta}")

    #     ### Update phi and derivatives
    #     xbeta, _, _, _ = np.linalg.lstsq( np.vstack([ A, np.sqrt(1.0/beta)*I ]), np.hstack([ b, np.zeros(I.shape[0]) ]) , rcond=None  )
    #     Axbeta = A @ xbeta
    #     phi = (np.linalg.norm( Axbeta - c )**2) - (eta**2)*m + shift

    #     # phi prime
    #     phi_prime = derivative(_phi, x0=beta, order=order, dx=dx, n=1)

    #     # phi prime prime
    #     phi_prime_prime = derivative(_phi, x0=beta, order=order, dx=dx, n=2)


    # # Setup for iteration
    # beta_hist = [beta]
    # phi_hist = [phi]
    # n_iters = 0

    # # Check if we already satisfy the absolute tolerance criteria?
    # if np.abs(phi) < atol:
    #     pass
    # else: # otherwise do the iteration

    #     for j in range(maxiter):

    #         # Convergence criteria based on relative change in root
    #         if j < 2:
    #             pass
    #         else:
    #             if early_stopping:

    #                 if ((beta_hist[-1] - beta_hist[-2])/beta_hist[-2]) < rtol:
    #                     break

    #                 if np.abs(phi) < atol:
    #                     break

    #         ### Take step
    #         mu = beta + phi_prime/(2*phi_prime_prime)
    #         alpha = 2*phi_prime*np.sqrt(beta - mu)
    #         gamma = phi - alpha*np.sqrt(beta - mu)
    #         beta = mu + ((gamma/alpha)**2)
    #         print(beta)
    #         beta_hist.append(beta)

    #         ### Update phi and derivatives
    #         xbeta, _, _, _ = np.linalg.lstsq( np.vstack([ A, np.sqrt(1.0/beta)*I ]), np.hstack([ b, np.zeros(I.shape[0]) ]) , rcond=None  )
    #         Axbeta = A @ xbeta
    #         phi = (np.linalg.norm( Axbeta - c )**2) - (eta**2)*m + shift

    #         # phi prime
    #         phi_prime = derivative(_phi, x0=beta, order=order, dx=dx, n=1)

    #         # phi prime prime
    #         phi_prime_prime = derivative(_phi, x0=beta, order=order, dx=dx, n=2)

    #         # Increase counter
    #         n_iters += 1

  
    # data = {
    #     "beta": beta,
    #     "lambda": 1.0/beta,
    #     "beta_hist": np.asarray(beta_hist),
    #     "phi_hist": np.asarray(phi_hist),
    #     "n_iters": n_iters,
    # }
    
    # return data












def matlab_gcv(A, L, b, eng=None, opt_upper_bound=200):
    """Solves the GCV problem using MATLAB fminbnd.

    eng: matlabengine instance, if have already started one. Otherwise starts and closes engine for computation.
    """

    # Start engine?
    if eng is None:
        eng_passed = False
        eng = matlab.engine.start_matlab()
    else:
        eng_passed = True

    # Add path to gcv function
    eng.addpath( eng.genpath(SCRIPT_DIR), nargout=0)

    # Cast to MATLAB
    _A = matlab.double(A)
    _L = matlab.double(L)
    _b = matlab.double( b.reshape( (len(b), 1) ) )
    _opt_upper_bound = matlab.double(opt_upper_bound)

    # Call MATLAB GSVD
    mu = eng.gcv(_A, _L, _b, _opt_upper_bound, nargout=1)

    if eng_passed:
        pass
    else:
        eng.close()


    return mu




























##### OLD code


# def dp_cubic_old(A, L, b, noise_var, eta=1.025, maxiter=100, beta0=None, tol=1e-3, shift=0.0, m=None):
#     """Implements an efficient cubically-convergent root-finder for finding a 
#     regularization parameter satisfying the discrepancy principle, for a problem
#     of the form

#         argmin_x (1.0/noise_var)*|| A x - b ||_2^2 + lambda*|| L x ||_2^2
        
#     Each iteration requires the solution of 3 least-squares problems. We assume 
#     that A and L are passed as dense matrices. The method is guaranteed to converge 
#     under weak assumptions.

#     The optimization is done over beta in lambda = 1/beta, instead of lambda itself.
#     This guarantees convexity of the objective.

#     shift: adds +shift to function that root-finding is applied to.
#     m: 2-norm of error is assumed to be m*noise_var. Defaults to m=len(b).
#     """

#     # Whiten the variables
#     A = (1.0/np.sqrt(noise_var))*A
#     b = (1.0/np.sqrt(noise_var))*b

#     # Misc
#     Atb = A.T @ b
#     if m is not None:
#         pass
#     else:
#         m = len(b)

#     if beta0 is None:
#         # Initialize at beta = 0, use analytic formulas for phi and its derivatives
#         beta = 0.0
#         phi = (np.linalg.norm(b)**2) - (eta**2)*m + shift
#         tmp, _, _, _ = np.linalg.lstsq(L.T, Atb, rcond=None)
#         phi_prime = -2*(np.linalg.norm(tmp)**2)
#         tmp2, _, _, _ = np.linalg.lstsq(L, tmp, rcond=None)
#         phi_prime_prime = 2*(np.linalg.norm(tmp2)**2)
#     else:
#         # Otherwise begin the iteration at the provided beta0
#         beta = beta0

#         ### Update phi and derivatives
#         xbeta, _, _, _ = np.linalg.lstsq( np.vstack([ A, np.sqrt(1.0/beta)*L ]), np.hstack([ b, np.zeros(L.shape[0]) ]) , rcond=None  )
#         Axbeta = A @ xbeta
#         Lxbeta = L @ xbeta
#         phi = (np.linalg.norm(Axbeta - b)**2) - (eta**2)*m + shift

#         # phi prime
#         term1 = - (2/(beta**2))*(np.linalg.norm(Lxbeta)**2)
#         QinvLtLxbeta, _, _, _ = np.linalg.lstsq( np.vstack([ A, np.sqrt(1.0/beta)*L ]), np.hstack([ np.zeros(A.shape[0]),  Lxbeta  ]) , rcond=None  )
#         AtAQinvLtLxbeta = A.T @ (A @ QinvLtLxbeta)
#         term2 = (2/(beta**2))*np.dot(QinvLtLxbeta, AtAQinvLtLxbeta )
#         phi_prime = term1 + term2
        
#         # phi prime prime
#         term1 = -(2/(beta**4))*np.dot( QinvLtLxbeta, L.T @ Lxbeta )
#         term2 = (2/(beta**4))*np.dot(QinvLtLxbeta, AtAQinvLtLxbeta )
#         QinvAtAxbeta, _, _, _ = np.linalg.lstsq( np.vstack([ A, np.sqrt(1.0/beta)*L ]), np.hstack([ Axbeta, np.zeros(L.shape[0]) ]) , rcond=None  )
#         term3 =  (4/(beta**4))*np.dot( L @ QinvAtAxbeta, L @ QinvLtLxbeta )
#         term4 = -(2/(beta**4))*np.dot( L @ xbeta, L @ QinvLtLxbeta )
#         term5 = -(4/(beta**3))*np.dot(QinvAtAxbeta, L.T @ Lxbeta)
#         term6 = +(4/(beta**3))*np.dot(Lxbeta, Lxbeta)
#         phi_prime_prime = term1 + term2 + term3 + term4 + term5 + term6

#     # Begin the iteration
#     beta_hist = [beta]
#     phi_hist = [phi]
#     n_iters = 0
#     for j in range(maxiter):

#         # Convergence criteria based on relative change in root
#         if j < 2:
#             pass
#         else:
#             if ((beta_hist[-1] - beta_hist[-2])/beta_hist[-2]) < tol:
#                 break

#         ### Take step
#         mu = beta + phi_prime/(2*phi_prime_prime)
#         alpha = 2*phi_prime*np.sqrt(beta - mu)
#         gamma = phi - alpha*np.sqrt(beta - mu)
#         beta = mu + ((gamma/alpha)**2)
#         beta_hist.append(beta)
#         print(beta)

#         ### Update phi and derivatives
#         xbeta, _, _, _ = np.linalg.lstsq( np.vstack([ A, np.sqrt(1.0/beta)*L ]), np.hstack([ b, np.zeros(L.shape[0]) ]) , rcond=None  )
#         Axbeta = A @ xbeta
#         Lxbeta = L @ xbeta
#         phi = (np.linalg.norm(Axbeta - b)**2) - (eta**2)*m + shift
#         phi_hist.append(phi)

#         # phi prime
#         term1 = - (2/(beta**2))*(np.linalg.norm(Lxbeta)**2)
#         QinvLtLxbeta, _, _, _ = np.linalg.lstsq( np.vstack([ A, np.sqrt(1.0/beta)*L ]), np.hstack([ np.zeros(A.shape[0]),  Lxbeta  ]) , rcond=None  )
#         AtAQinvLtLxbeta = A.T @ (A @ QinvLtLxbeta)
#         term2 = (2/(beta**2))*np.dot(QinvLtLxbeta, AtAQinvLtLxbeta )
#         phi_prime = term1 + term2
        
#         # phi prime prime
#         term1 = -(2/(beta**4))*np.dot( QinvLtLxbeta, L.T @ Lxbeta )
#         term2 = (2/(beta**4))*np.dot(QinvLtLxbeta, AtAQinvLtLxbeta )
#         QinvAtAxbeta, _, _, _ = np.linalg.lstsq( np.vstack([ A, np.sqrt(1.0/beta)*L ]), np.hstack([ Axbeta, np.zeros(L.shape[0]) ]), rcond=None  )
#         term3 =  (4/(beta**4))*np.dot( L @ QinvAtAxbeta, L @ QinvLtLxbeta )
#         term4 = -(2/(beta**4))*np.dot( L @ xbeta, L @ QinvLtLxbeta )
#         term5 = -(4/(beta**3))*np.dot(QinvAtAxbeta, L.T @ Lxbeta)
#         term6 = +(4/(beta**3))*np.dot(Lxbeta, Lxbeta)
#         phi_prime_prime = term1 + term2 + term3 + term4 + term5 + term6

#         # Increase counter
#         n_iters += 1

#     data = {
#         "beta": beta,
#         "lambda": 1.0/beta,
#         "beta_hist": np.asarray(beta_hist),
#         "phi_hist": np.asarray(phi_hist),
#         "n_iters": n_iters,
#     }
    
#     return data









#### OLD plotting codes for future reference




# def discrepancy_principle(Q_A, R_A, R_WL, b, eta=1.01, last_lpoint=None):
#     """Chooses a new lambdah parameter using discrepancy principle.
#     """

#     # We have already whitened Q_A and R_A!

#     def dp_func(log_reg_param):

#         try:
#             # Evaluate tikhonov solution
#             y, _ , _, svals = np.linalg.lstsq(np.concatenate((R_A, np.sqrt( np.power(10, log_reg_param) ) * R_WL)), np.concatenate((Q_A.T @ b, np.zeros((R_WL.shape[0],)))), rcond=None)

#             #return np.linalg.norm( Q_A @ (R_A @ y) -  b, ord=2 )**2 - eta*Q_A.shape[0]
#             #return (np.linalg.norm( (R_A @ y) -  (Q_A.T @ b), ord=2 )**2) - (eta*R_A.shape[0])
            
#             # Residual for the full-scale problem?
#             term1 = np.linalg.norm( (R_A @ y) -  (Q_A.T @ b), ord=2 )**2
#             term2 = np.linalg.norm( b -  ( Q_A @ (Q_A.T @ b) ), ord=2 )**2
#             #term2 = 0.0

#             return term1 + term2 - eta*len(b)
        
#         except:

#             return 1e10



#     # Compute a solution
#     #root_res = op.root_scalar(f = dp_func, x0=-10, x1=12, args=(), xtol=1e-6, maxiter=1000)
#     root_res = op.bisect(f = dp_func, a=-15, b=15, args=(), xtol=1e-6, maxiter=1000)
#     #root_res = dp_cubic(R_A, R_WL, Q_A.T @ b, noise_var=1.0, tol=1e-3, )

#     # log_lambdah_star = root_res.root
#     log_lambdah_star = root_res
#     log_lambdah_star_eval = dp_func(log_lambdah_star)
#     lambdah_star = np.power(10,log_lambdah_star)

#     # If plotting?
#     fig = None

#     #log_lambdahs = np.linspace(-12, 12, 100)
#     log_lambdahs = np.linspace( log_lambdah_star - 0.02, log_lambdah_star + 0.02, 100 )
#     evals = []
#     for log_lambdah in log_lambdahs:
#         evals.append( dp_func(log_lambdah) )

    # fig, axs = plt.subplots()

    # axs.plot(log_lambdahs, evals, linestyle='--', color='b') 
    # axs.scatter(log_lambdahs, evals, marker='x', color='b', label="Sample points") 
    # axs.scatter(log_lambdah_star, log_lambdah_star_eval, marker="o", color="red", label="Optimized $\\lambda$")
    # axs.set_xlabel("$\\log_{10} \\lambda$")
    # axs.set_ylabel("$\\text{DP}(\\lambda)$")
    # axs.set_title(f"DP curve, eta = {eta}")

    # # Also plot 0's
    # tmp = np.zeros_like(log_lambdahs)
    # axs.plot(log_lambdahs, tmp, color="black", ls="--")

    # #axs.set_yscale("log")
    
    # # min_idx = np.argmin(evals)
    # # min_eval = np.amin(evals)
    # # grid_opt_log_lambdah = log_lambdahs[min_idx]
    # # if log_lambdah_star_eval > min_eval:
    # #     axs.set_title(f"Modified GCV curve, gamma = {gcv_gamma}, overwrote!")
    # #     log_lambdah_star = grid_opt_log_lambdah
    # #     lambdah_star = np.power(10, log_lambdah_star)
    # #     log_lambdah_star_eval = min_eval

    # # Using the rightmost local minimizer criterion?
    # # idx = find_last_local_min(evals)
    # # log_lambdah_star = log_lambdahs[idx]
    # # log_lambdah_star_eval = evals[idx]
    # # lambdah_star = np.power(10, log_lambdah_star)
    # # axs.scatter(log_lambdah_star, log_lambdah_star_eval, marker="o", color="orange", label="Helped $\\lambda$")

    # #axs.scatter(log_lambdah_star, log_lambdah_star_eval, marker="o", color="red", label="Optimized $\\lambda$")
    # axs.legend()

    # Finally, don't let lambdah move TOO much across the iterations

    #return lambdah_star, fig









    # fig, axs = plt.subplots()

    # axs.plot(log_lambdahs, evals, linestyle='--', color='b') 
    # axs.scatter(log_lambdahs, evals, marker='x', color='b', label="Sample points") 
    # axs.scatter(log_lambdah_star, log_lambdah_star_eval, marker="o", color="red", label="Optimized $\\lambda$")
    # axs.set_xlabel("$\\log_{10} \\lambda$")
    # axs.set_ylabel("$\\ln \\text{GCV}(\\lambda)$")
    # axs.set_title(f"Modified GCV curve, gamma = {gcv_gamma}")
    # axs.legend()
    
    # min_idx = np.argmin(evals)
    # min_eval = np.amin(evals)
    # grid_opt_log_lambdah = log_lambdahs[min_idx]
    # if log_lambdah_star_eval > min_eval:
    #     axs.set_title(f"Modified GCV curve, gamma = {gcv_gamma}, overwrote!")
    #     log_lambdah_star = grid_opt_log_lambdah
    #     lambdah_star = np.power(10, log_lambdah_star)
    #     log_lambdah_star_eval = min_eval

    # Using the rightmost local minimizer criterion?
    # idx = find_last_local_min(evals)
    # log_lambdah_star = log_lambdahs[idx]
    # log_lambdah_star_eval = evals[idx]
    # lambdah_star = np.power(10, log_lambdah_star)
    # axs.scatter(log_lambdah_star, log_lambdah_star_eval, marker="o", color="orange", label="Helped $\\lambda$")


    #axs.scatter(log_lambdah_star, log_lambdah_star_eval, marker="o", color="red", label="Optimized $\\lambda$")
    #axs.legend()

    # Finally, don't let lambdah move TOO much across the iterations

    #return lambdah_star, fig
