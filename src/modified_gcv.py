

import numpy as np 
import matplotlib.pyplot as plt
# from scipy.optimize import newton, minimize
import scipy.linalg as la
import scipy.optimize as op
#from pylops import LinearOperator


from scipy.optimize import minimize



def gcv_numerator(reg_param, Q_A, R_A, R_L, b, **kwargs):

    variant = kwargs['variant'] if ('variant' in kwargs) else 'standard'

    inverted, _ , _, svals = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(reg_param) * R_L)), np.concatenate((Q_A.T @ b, np.zeros((R_L.shape[0],)))),rcond=None)
    cond_est = np.amax(svals)/np.amin(svals)

    # return np.sqrt((np.linalg.norm( R_A @ inverted - Q_A.T @ b ))**2 + np.linalg.norm(b - Q_A@(Q_A.T@b))**2)
    if variant == 'modified':
        return ((np.linalg.norm( R_A @ inverted - Q_A.T @ b ))**2 + np.linalg.norm(b - Q_A@(Q_A.T@b))**2)
    else:
        return np.linalg.norm( R_A @ inverted - Q_A.T @ b )


def gcv_denominator(reg_param, R_A, R_L, b, condlim=1e5, **kwargs):

    variant = kwargs['variant'] if ('variant' in kwargs) else 'standard'
  
    inverted, _ , _, svals = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(reg_param) * R_L)), np.vstack( (  np.eye( R_A.T.shape[1] )  , np.zeros((R_L.shape[0],R_A.T.shape[1]) ) ) ),rcond=None)
    cond_est = np.amax(svals)/np.amin(svals)

    if variant == 'modified':
       m = kwargs['fullsize'] # probably this can be b.size -- NOT FOR HYBRID SOLVERS!
       # trace_term = (m - R_A.shape[1]) - np.trace(R_A @ inverted) # b.size - np.trace(R_A @ inverted) # this is defined with respect to the projected quantities 
       trace_term = m - np.trace(R_A @ inverted) # b.size - np.trace(R_A @ inverted) # this is defined with respect to the projected quantities 
    else:
        # in this way works even if we revert to the fully projected pb (call with Q_A.T@b)
        # trace_term = b.size - np.trace(R_A @ inverted) # this is defined with respect to the projected quantities
        trace_term = R_A.shape[0] - np.trace(R_A @ inverted) # this is defined with respect to the projected quantities
    
    return trace_term




def generalized_crossvalidation(Q_A, R_A, R_L, b, last_lambdah=None, **kwargs):

    if 'tol' in kwargs:
        tol = kwargs['tol']
    else:
        tol = 10**(-12)

    if 'gcvtype' in kwargs:
        gcvtype = kwargs['gcvtype']
    else:
        gcvtype = 'tikhonov'
    
    # function to minimize
    if gcvtype == 'tikhonov':    
        
        #gcv_func = lambda log_reg_param: gcv_numerator(np.exp(log_reg_param), Q_A, R_A, R_L, b) / gcv_denominator(np.exp(log_reg_param), R_A, R_L, b, **kwargs)
        gcv_func = lambda log_reg_param: np.log( gcv_numerator(np.exp(log_reg_param), Q_A, R_A, R_L, b) ) - np.log( np.abs( gcv_denominator(np.exp(log_reg_param), R_A, R_L, b, **kwargs) ) )
        
        log_lambdah = op.fminbound(func = gcv_func, x1 = -12, x2 = 12, args=(), xtol=1e-12, maxfun=1000, full_output=0, disp=0) ## should there be tol here?
        lambdah = np.exp(log_lambdah)

    elif gcvtype == 'tsvd':
        m = Q_A.shape[0]
        n = R_L.shape[1]
        gcv_vals = []
        bhat = Q_A.T@b
        f = np.ones((m,1))
        for i in range(n):
            f[n-(i+1),] = 0
            fvar = np.concatenate((1 - f[:n,], f[n:,]))
            coeff = (fvar*bhat)**2
            gcv_numerator_tsvd = np.sum(coeff)
            gcv_denominator_tsvd = (m - (n-(i+1)))**2
            gcv_vals.append(gcv_numerator_tsvd/gcv_denominator_tsvd)   
        lambdah = n - (gcv_vals.index(min(gcv_vals))+1)
    elif gcvtype == 'tgsvd':
        m = Q_A.shape[0]
        n = R_L.shape[1]
        p = R_L.shape[0]
        gcv_vals = []
        bhat = Q_A.T@b
        coeff = np.square(bhat)
        for i in range(n):
            coeff[n-(i+1),] = 0
            gcv_numerator_tgsvd = np.sum(coeff)
            gcv_denominator_tgsvd = (n-(i+1) - (n-p))**2
            gcv_vals.append(gcv_numerator_tgsvd/gcv_denominator_tgsvd)   
        lambdah = gcv_vals.index(min(gcv_vals))
    
    return lambdah 






def modified_gcv(Q_A, R_A, R_L, b, gamma=0.1, **kwargs):

    if 'tol' in kwargs:
        tol = kwargs['tol']
    else:
        tol = 10**(-12)

    
    # function to minimize
    def _modified_gcv_func(reg_param):
        numerator = gcv_numerator(reg_param, Q_A, R_A, R_L, b) 
        denominator = gcv_denominator(reg_param, R_A, R_L, b, **kwargs)

        # Data term
        R_A_2 = R_A.T @ R_A
        R_A_2 = R_A_2.todense() if isinstance(R_A_2, LinearOperator) else R_A_2

        # The regularizer term:
        R_L_2 = (R_L.T @ R_L)
        R_L_2 = R_L_2.todense() if isinstance(R_L_2, LinearOperator) else R_L_2
        inverted = la.solve( ( R_A_2 + reg_param * R_L_2), R_A.T )

        Z = R_A @ inverted
        fac = gamma + (1.0 - gamma)*(1.0/R_A.shape[0])*np.trace( Z @ Z )

        return (numerator/denominator)*fac

    lambdah = op.fminbound(func = _modified_gcv_func, x1 = 1e-09, x2 = 1e12, args=(), xtol=1e-12, maxfun=1000, full_output=0, disp=0) ## should there be tol here?
    
    return lambdah