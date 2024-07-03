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
from modified_gcv import generalized_crossvalidation

from modified_gcv import gcv_numerator, gcv_denominator
import scipy.optimize as op








############### MMGKS

class MMGKS:
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



    def solve(self, n_iter=25, init_dim=5, q=1.0, epsilon=1e-2, regparam="gcv", x0=None, x_true=None, gcv_gamma=0.1, plotting=False, pdf_name="demo.pdf", helping=False, **kwargs):
        """Driver for MMGKS solver.
        """

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

        # Plotting?
        if plotting:
            pdf = PdfPages(pdf_name)
        
        # Build initial basis and QR factor of AV
        V = self.build_initial_basis(init_dim)
        AV = self.A @ V 
        tot_matvec_A += 2*init_dim
        tot_matvec_A += init_dim
        Q_A, R_A = qr(AV, mode='economic') 

        # Get initial LV
        LV = self.L @ V
        tot_matvec_L += init_dim

        for ii in progress_bar(range(n_iter)):

            # Update weights and get new QR factorization of W L V
            Lx = self.L @ x
            tot_matvec_L += 1
            w = smoothed_holder_weights( Lx, epsilon=epsilon, p=q)
            W = sps.diags(w)
            WLV = W @ LV
            Q_WL, R_WL = qr(WLV, mode='economic') 
            

            lambdah, fig = self.choose_lambdah_gcv(Q_A, R_A, R_WL, self.b, plotting=plotting, gcv_gamma=gcv_gamma, helping=helping)
            if plotting:
                pdf.savefig(fig)
                plt.close(fig)

            # # Select new regularization parameter
            # if regparam == "gcv":
            #     lambdah = generalized_crossvalidation(Q_A, R_A, R_WL, self.b, **kwargs)
            # elif regparam == "dp":
            #     lambdah = discrepancy_principle(Q_A, R_A, R_WL, self.b, delta=1.0, **kwargs)
            #     # fix delta to be 1, since we have already whitened in the begining
            # else:
            #     # Otherwise hope regparam is a value and hold equal to this
            #     lambdah = regparam

            lambda_history.append(lambdah)

            # Obtain new solution
            y, _ , _, svals = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_WL)), np.concatenate((Q_A.T @ self.b, np.zeros((R_WL.shape[0],)))), rcond=None)
            cond_est = np.amax(svals)/np.amin(svals)
            cond_ests.append(cond_est)
            x = V @ y

            if x_true is not None: rre_history.append( np.linalg.norm(x - x_true)/x_true_norm )

            if ii >= R_WL.shape[0]:
                break
            
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
            #r = r - V @ (V.T @ r)
            #normed_r = r / np.linalg.norm(r) 
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

            # Update QR factor for A
            #Q_A, R_A = qr_insert( Q_A, R_A, u = Avn, k=-1, which="col" )
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
        }

        if x_true is not None:
            data["rre_history"] = np.asarray(rre_history)

        if plotting:
            pdf.close()

        return data
    


    def choose_lambdah_gcv(self,Q_A, R_A, R_WL, b, gcv_gamma=0.1, plotting=False, helping=False):
        """Chooses a new lambdah parameter using cross validation. If plotting=True is passed,
        also makes a plot of the GCV functional.
        """

        # Build objective function (in terms of log lambda)
        #gcv_func = lambda log_reg_param: np.log( gcv_numerator(np.power(10, log_reg_param), Q_A, R_A, R_WL, b) ) - np.log( gcv_denominator(np.power(10,log_reg_param), R_A, R_WL, b ) ) 
        #gcv_func = lambda log_reg_param: gcv_numerator(np.exp(log_reg_param), Q_A, R_A, R_WL, b) / gcv_denominator(np.exp(log_reg_param), R_A, R_WL, b)

        def gcv_func(log_reg_param):
            inverted, _ , _, svals = np.linalg.lstsq(np.concatenate((R_A, np.sqrt( np.power(10, log_reg_param) ) * R_WL)), np.vstack( (  np.eye( R_A.T.shape[1] )  , np.zeros((R_WL.shape[0],R_A.T.shape[1]) ) ) ),rcond=None)
            Z = R_A @ inverted
            fac = gcv_gamma + (1.0 - gcv_gamma)*(1.0/R_A.shape[0])*np.trace( Z @ Z )
            return np.log(fac) + np.log( gcv_numerator(np.power(10, log_reg_param), Q_A, R_A, R_WL, b) ) - np.log( np.abs(gcv_denominator(np.power(10,log_reg_param), R_A, R_WL, b ) ) ) 

        # Compute a solution
        log_lambdah_star = op.fminbound(func = gcv_func, x1 = -12, x2 = 12, args=(), xtol=1e-12, maxfun=1000, full_output=0, disp=0) ## should there be tol here?
        log_lambdah_star_eval = gcv_func(log_lambdah_star)
        lambdah_star = np.power(10,log_lambdah_star)
        
        # If plotting?
        fig = None
        if plotting:
            log_lambdahs = np.linspace(-12, 12, 100)
            evals = []
            for log_lambdah in log_lambdahs:
                evals.append( gcv_func(log_lambdah) )

            fig, axs = plt.subplots()

            axs.plot(log_lambdahs, evals, linestyle='--', color='b') 
            axs.scatter(log_lambdahs, evals, marker='x', color='b', label="Sample points") 
            axs.scatter(log_lambdah_star, log_lambdah_star_eval, marker="o", color="red", label="Optimized $\\lambda$")
            axs.set_xlabel("$\\log_{10} \\lambda$")
            axs.set_ylabel("$\\ln \\text{GCV}(\\lambda)$")
            axs.set_title(f"Modified GCV curve, gamma = {gcv_gamma}")
            axs.legend()

           # Overwrite if a grid point is better?
            if helping:
                
                min_idx = np.argmin(evals)
                min_eval = np.amin(evals)
                grid_opt_log_lambdah = log_lambdahs[min_idx]
                if log_lambdah_star_eval > min_eval:
                    axs.set_title(f"Modified GCV curve, gamma = {gcv_gamma}, overwrote!")
                    log_lambdah_star = grid_opt_log_lambdah
                    lambdah_star = np.power(10, log_lambdah_star)
                    log_lambdah_star_eval = min_eval

            axs.scatter(log_lambdah_star, log_lambdah_star_eval, marker="o", color="red", label="Optimized $\\lambda$")
            axs.legend()

        return lambdah_star, fig
    





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



class GKSIAS:
    """Implements GKSIAS solver.

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



    def solve(self, n_iter=25, init_dim=5, hyperpr_r=-1.0, hyperpr_beta=1.0, x0=None, x_true=None, gcv_gamma=0.1, plotting=False, pdf_name="demo.pdf", helping=False, **kwargs):
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

        # Plotting?
        if plotting:
            pdf = PdfPages(pdf_name)
        
        # Build initial basis and QR factor of AV
        V = self.build_initial_basis(init_dim)
        AV = self.A @ V 
        tot_matvec_A += 2*init_dim
        tot_matvec_A += init_dim
        Q_A, R_A = qr(AV, mode='economic') 

        # Get initial LV
        LV = self.L @ V
        tot_matvec_L += init_dim
        lambdah = 1.0
        vartheta = 1.0/lambdah

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
            
            lambdah, fig = self.choose_lambdah_gcv(Q_A, R_A, R_WL, self.b, plotting=plotting, gcv_gamma=gcv_gamma, helping=helping)
            if plotting:
                pdf.savefig(fig)
                plt.close(fig)

            # # Select new regularization parameter
            # if regparam == "gcv":
            #     lambdah = generalized_crossvalidation(Q_A, R_A, R_WL, self.b, **kwargs)
            # elif regparam == "dp":
            #     lambdah = discrepancy_principle(Q_A, R_A, R_WL, self.b, delta=1.0, **kwargs)
            #     # fix delta to be 1, since we have already whitened in the begining
            # else:
            #     # Otherwise hope regparam is a value and hold equal to this
            #     lambdah = regparam

            lambda_history.append(lambdah)
            # Handle vartheta
            try:
                vartheta = 1.0/lambdah
            except:
                vartheta = np.inf
            vartheta_history.append(vartheta)

            # Obtain new solution
            y, _ , _, svals = np.linalg.lstsq(np.concatenate((R_A, np.sqrt(lambdah) * R_WL)), np.concatenate((Q_A.T @ self.b, np.zeros((R_WL.shape[0],)))), rcond=None)
            cond_est = np.amax(svals)/np.amin(svals)
            cond_ests.append(cond_est)
            x = V @ y

            if x_true is not None: rre_history.append( np.linalg.norm(x - x_true)/x_true_norm )

            if ii >= R_WL.shape[0]:
                break
            
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
            #r = r - V @ (V.T @ r)
            #normed_r = r / np.linalg.norm(r) 
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

            # Update QR factor for A
            #Q_A, R_A = qr_insert( Q_A, R_A, u = Avn, k=-1, which="col" )
            Q_A, R_A = qr(AV, mode='economic') 
            

        # Package output
        data = {
            "x": x,
            "w": w,
            "theta": vartheta_history[-1]*( (1.0/w)**2 ),
            "vartheta": vartheta_history[-1],
            "vartheta_history": np.asarray(vartheta_history),
            "tot_matvec_A": tot_matvec_A,
            "tot_matvec_L": tot_matvec_L,
            "residual_history": np.asarray(residual_history),
            "lambda_history": np.asarray(lambda_history),
            "V": V,
            "cond_ests": np.asarray(cond_ests),
        }

        if x_true is not None:
            data["rre_history"] = np.asarray(rre_history)

        if plotting:
            pdf.close()

        return data
    


    def choose_lambdah_gcv(self,Q_A, R_A, R_WL, b, gcv_gamma=0.1, plotting=False, helping=False, last_lambdah=None):
        """Chooses a new lambdah parameter using cross validation. If plotting=True is passed,
        also makes a plot of the GCV functional.
        """

        # Build objective function (in terms of log lambda)
        #gcv_func = lambda log_reg_param: np.log( gcv_numerator(np.power(10, log_reg_param), Q_A, R_A, R_WL, b) ) - np.log( gcv_denominator(np.power(10,log_reg_param), R_A, R_WL, b ) ) 
        #gcv_func = lambda log_reg_param: gcv_numerator(np.exp(log_reg_param), Q_A, R_A, R_WL, b) / gcv_denominator(np.exp(log_reg_param), R_A, R_WL, b)

        def gcv_func(log_reg_param):
            inverted, _ , _, svals = np.linalg.lstsq(np.concatenate((R_A, np.sqrt( np.power(10, log_reg_param) ) * R_WL)), np.vstack( (  np.eye( R_A.T.shape[1] )  , np.zeros((R_WL.shape[0],R_A.T.shape[1]) ) ) ),rcond=None)
            Z = R_A @ inverted
            fac = gcv_gamma + (1.0 - gcv_gamma)*(1.0/R_A.shape[0])*np.trace( Z @ Z )
            return np.log(fac) + np.log( gcv_numerator(np.power(10, log_reg_param), Q_A, R_A, R_WL, b) ) - np.log( np.abs( gcv_denominator(np.power(10,log_reg_param), R_A, R_WL, b ) ) ) 

        # Compute a solution
        log_lambdah_star = op.fminbound(func = gcv_func, x1 = -12, x2 = 12, args=(), xtol=1e-12, maxfun=1000, full_output=0, disp=0) ## should there be tol here?
        log_lambdah_star_eval = gcv_func(log_lambdah_star)
        lambdah_star = np.power(10,log_lambdah_star)
        
        # If plotting?
        fig = None
        if plotting:
            log_lambdahs = np.linspace(-12, 12, 100)
            evals = []
            for log_lambdah in log_lambdahs:
                evals.append( gcv_func(log_lambdah) )

            fig, axs = plt.subplots()

            axs.plot(log_lambdahs, evals, linestyle='--', color='b') 
            axs.scatter(log_lambdahs, evals, marker='x', color='b', label="Sample points") 
            axs.scatter(log_lambdah_star, log_lambdah_star_eval, marker="o", color="red", label="Optimized $\\lambda$")
            axs.set_xlabel("$\\log_{10} \\lambda$")
            axs.set_ylabel("$\\ln \\text{GCV}(\\lambda)$")
            axs.set_title(f"Modified GCV curve, gamma = {gcv_gamma}")
            axs.legend()

            # Overwrite if a grid point is better?
            if helping:
                
                min_idx = np.argmin(evals)
                min_eval = np.amin(evals)
                grid_opt_log_lambdah = log_lambdahs[min_idx]
                if log_lambdah_star_eval > min_eval:
                    axs.set_title(f"Modified GCV curve, gamma = {gcv_gamma}, overwrote!")
                    log_lambdah_star = grid_opt_log_lambdah
                    lambdah_star = np.power(10, log_lambdah_star)
                    log_lambdah_star_eval = min_eval

            axs.scatter(log_lambdah_star, log_lambdah_star_eval, marker="o", color="red", label="Optimized $\\lambda$")
            axs.legend()

        # Finally, don't let lambdah move TOO much across the iterations

        return lambdah_star, fig
    
    























