import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from scipy.sparse.linalg import aslinearoperator
import scipy.sparse as sps

from fastprogress import progress_bar


import jlinops

from jlinops import CUPY_INSTALLED
if CUPY_INSTALLED:
    import cupy as cp



class IASSolver:

    def __init__(self, F, R, y, hyperparams, noise_var=None, pdata={}):

        # Check that everything is on the correct device
     
        # Bind
        self.F = F
        self.y = y
        self.noise_var = noise_var
        self.R = R
        self.hyperparams = hyperparams
    
        # Some checks
        assert self.F.shape[1] == self.R.shape[1], "Shapes of forward and regularization operators do not agree!"
       
        if not np.isscalar(self.hyperparams["prior"]["vartheta"]):
            assert len(self.hyperparams["prior"]["vartheta"]) == self.R.shape[0], "If vartheta is a vector, must be same size as output size of regularization operator."

        assert not ( ( "noise_var" in self.hyperparams.keys() ) and ( noise_var is not None ) ), "Leave noise_var=None if specifying a hyperprior for the noise variance."
        
        # Handle pdata
        self._pdata = pdata
        self.pdata = {
            "W": None,
            "Rpinv_factory": None,
            "Rinv": None,
            "FWpinv": None,
        }
        for key in self._pdata.keys():
            if key in ["W", "Rpinv_factory", "Rinv", "FWpinv", "Mpinv_factory"]:
                self.pdata[key] = self._pdata[key]
            else:
                raise ValueError
        
        # Some others
        self.n = self.F.shape[1] # dimension of the unknown
        self.m = self.F.shape[0] # output size of forward operator
        self.k = self.R.shape[0] # output size of regularization operator
        
        if noise_var is None:
            self.updating_noise_var = True
        else:
            self.updating_noise_var = False

        if (self.pdata["W"] is not None) and (self.pdata["FWpinv"] is None):
            tmp = jlinops.QRPinvOperator( jlinops.MatrixLinearOperator(self.F.matmat(self.pdata["W"].A)) )
            self.pdata["FWpinv"] = tmp
            self.pdata["Wpinv"] = jlinops.QRPinvOperator(self.pdata["W"].A)
      
        # Misc
        self.last_x_solve = None
        self.last_xbar = None



    def solve(self, max_iters=100, x0=None, priorconditioning=False, x_update_solver_params={}, eps=1e-2, early_stopping=True, warmstarting=True, x_update_device="cpu", x_true=None):
        """Driver for the IAS solver.
        """

        # Handle noise var
        if self.updating_noise_var:
            noise_var_curr = 1.0
        else:
            noise_var_curr = self.noise_var
        
        # Handle initialization
        if x0 is None:
            x = np.ones(self.n)
        else:
            x = x0
        
        # Handle warmstarting
        if warmstarting:
            x_prev = x.copy()
        else:
            x_prev = None

        # Other
        self.Rpinv_prev_eval = None
        self.Rpinv_prev_eval_t = None

        # Tracking
        n_cg_its_per_iteration = []
        n_Rpinv_matvec_cg_its_per_iteration = []
        n_Rpinv_rmatvec_cg_its_per_iteration = []
        theta_curr = np.ones(self.R.shape[0])
        obj_vals = [self.objective(x, theta_curr, noise_var_curr)]
        obj_vals_each_update = [self.objective(x, theta_curr, noise_var_curr)]
        noise_vars = []
        converged = False
        n_iters = 0
        rel_errs = []
        
        # Iterate
        for j in progress_bar(range(max_iters)):

            if (j == 0) or ( (j % 10) == 0 ): 
                # Update noise variance
                if self.updating_noise_var:
                    noise_var_curr = self.noise_var_update(x)
                noise_vars.append(noise_var_curr)
                obj_vals_each_update.append(self.objective(x, theta_curr, noise_var_curr))

            # Update prior thetas
            theta_curr = self.theta_update(x)
            obj_vals_each_update.append(self.objective(x, theta_curr, noise_var_curr))

            # Update x
            x, n_cg_its, Rpinv_cg_matvecs, Rpinv_cg_rmatvecs = self.x_update(theta_curr, noise_var_curr, priorconditioning, x_update_solver_params, x_prev, x_update_device)
            n_Rpinv_matvec_cg_its_per_iteration.append(Rpinv_cg_matvecs)
            n_Rpinv_rmatvec_cg_its_per_iteration.append(Rpinv_cg_rmatvecs)
            
            if warmstarting:
                x_prev = x.copy()
            n_cg_its_per_iteration.append(n_cg_its)
            
            # Compute objective value
            obj_vals_each_update.append(self.objective(x, theta_curr, noise_var_curr))
            obj_val = self.objective(x, theta_curr, noise_var_curr)
            obj_vals.append(obj_val)

            # Compute relative error if applicable
            if x_true is not None:
                x_true_norm = np.linalg.norm(x_true)
                rre = np.linalg.norm(x - x_true)/x_true_norm
                rel_errs.append(rre)

            if (j > 0) and (early_stopping):
                converged = self.check_stopping_criterion(theta_prev, theta_curr, noise_var_prev, noise_var_curr, eps=eps)
                if converged:
                    break
                else:
                    pass
                
            # Advance
            theta_prev = theta_curr
            noise_var_prev = noise_var_curr
            n_iters += 1

        data = {
            "x": x,
            "theta": theta_curr,
            "n_cg_its_per_iteration": np.asarray(n_cg_its_per_iteration),
            "converged": converged,
            "obj_vals": np.asarray(obj_vals),
            "noise_var": noise_var_curr,
            "noise_vars": np.asarray(noise_vars),
            "n_iters": n_iters,
            "n_Rpinv_matvec_cg_its_per_iteration": np.asarray(n_Rpinv_matvec_cg_its_per_iteration),
            "n_Rpinv_rmatvec_cg_its_per_iteration": np.asarray(n_Rpinv_rmatvec_cg_its_per_iteration),
            "obj_vals_each_update": np.asarray(obj_vals_each_update),
        }

        if x_true is not None:
            data["rel_errs"] = np.asarray(rel_errs)

        return data



    def x_update(self, theta, noise_var, priorconditioning=False, x_update_solver_params={}, x_prev=None, x_update_device="cpu"):
        """Returns the x-update for fixed local variance parameters theta.
        """
        
        # Whiten forward operator and data
        noise_stdev = np.sqrt(noise_var)
        Ftilde = (1.0/noise_stdev)*self.F
        ytilde = (1.0/noise_stdev)*self.y
        
        # Build Rtilde
        Rtilde = jlinops.DiagonalOperator(1.0/np.sqrt(theta)) @ self.R
        
        # If not using priorconditioning, solve original problem using CGLS without standardizing
        if not priorconditioning:

            # Solve using cgls

            if x_update_device == "cpu":
            
                cgls_solve = jlinops.trlstsq(Ftilde, Rtilde, ytilde, lam=1.0, initialization=x_prev, **x_update_solver_params)
                
                return cgls_solve["x"], cgls_solve["n_iters"], 0, 0

            else:

                Ftilde = Ftilde.to_gpu()
                Rtilde = Rtilde.to_gpu()
                ytilde = cp.asarray(ytilde)
                
                x_prev =  cp.asarray(x_prev)

                cgls_solve = jlinops.trlstsq(Ftilde, Rtilde, ytilde, lam=1.0, initialization=x_prev, **x_update_solver_params)
                
                x = cp.asnumpy(cgls_solve["x"])
                
                return x, cgls_solve["n_iters"], 0, 0

        
        # If using priorconditioning
        else:

            if x_update_device == "cpu":

                # If we have Mpinv, use it
                if "Mpinv_factory" in self.pdata.keys():
                    
                    print("using Mpinv")
                    # Build current Rpinv operator
                    Mpinv = self.pdata["Mpinv_factory"](theta)
                    cgls_solve = jlinops.epcgls(Ftilde, Rtilde, Mpinv, self.pdata["W"], ytilde, shift=None, FWpinv=None, x0=x_prev, **x_update_solver_params)
                    self.last_x_solve = cgls_solve["x_perp_F"]
                    # # Solve using transformed CGLS
                    # cgls_solve = jlinops.trlstsq_standard_form(Ftilde, ytilde, Rpinv=Rpinv, R=Rtilde,
                    #                                         AWpinv=noise_stdev*self.pdata["FWpinv"], lam=1.0, shift=None, W=self.pdata["W"], initialization=x_prev, **x_update_solver_params)

                    # # Only if it has this attribute
                    Rpinv_cg_matvecs = 0
                    Rpinv_cg_rmatvecs = 0
                    # if hasattr(Rpinv, 'matvec_cg_hist'):
                    #     Rpinv_cg_matvecs = sum(Rpinv.matvec_cg_hist)
                    #     Rpinv_cg_rmatvecs = sum(Rpinv.rmatvec_cg_hist)
                    
                    return cgls_solve["x"], cgls_solve["n_iters"], Rpinv_cg_matvecs, Rpinv_cg_rmatvecs

                
                # Otherwise, use Rpinv
                else:
            
                    # Build current Rpinv operator
                    Rpinv = self.pdata["Rpinv_factory"](theta)
                    
                    # Solve using transformed CGLS
                    cgls_solve = jlinops.trlstsq_standard_form(Ftilde, ytilde, Rpinv=Rpinv, R=Rtilde,
                                                            AWpinv=noise_stdev*self.pdata["FWpinv"], lam=1.0, shift=None, W=self.pdata["W"], initialization=x_prev, **x_update_solver_params)

                    # Only if it has this attribute
                    Rpinv_cg_matvecs = 0
                    Rpinv_cg_rmatvecs = 0
                    if hasattr(Rpinv, 'matvec_cg_hist'):
                        Rpinv_cg_matvecs = sum(Rpinv.matvec_cg_hist)
                        Rpinv_cg_rmatvecs = sum(Rpinv.rmatvec_cg_hist)
                    
                    return cgls_solve["x"], cgls_solve["n_iters"], Rpinv_cg_matvecs, Rpinv_cg_rmatvecs

            else:

                # # Build current Rpinv operator
                # if (self.Rpinv_prev_eval is not None) and (not np.all([a is None for a in self.Rpinv_prev_eval.flat])):
                #     #print(self.Rpinv_prev_eval)
                #     # print(type())
                #     prev_eval = cp.asarray(self.Rpinv_prev_eval)
                #     prev_eval_t = cp.asarray(self.Rpinv_prev_eval_t)
                #     print("not all none")
                # else:
                #     prev_eval = None
                #     prev_eval_t = None

                prev_eval = None
                prev_eval_t = None
                    
                Rpinv = self.pdata["Rpinv_factory"](theta)
                #Rpinv = self.pdata["Rpinv_factory"](theta)

                print(Rpinv.Rwpinv.prev_eval)
                if Rpinv.device == "gpu":
                    pass
                else:
                    Rpinv = Rpinv.to_gpu()

                # # Give Rpinv the last solution, if it is an attribute
                # if hasattr(Rpinv, 'prev_eval'):
                #     if self.Rpinv_prev_eval is None:
                #         pass
                #     else:
                #         if Rpinv.device == "cpu":
                #             Rpinv.Rwpinv.prev_eval = self.Rpinv_prev_eval
                #             Rpinv.Rwpinv.prev_eval_t = self.Rpinv_prev_eval_t
                #         else:
                #             Rpinv.Rwpinv.prev_eval = cp.asarray(self.Rpinv_prev_eval)
                #             Rpinv.Rwpinv.prev_eval_t = cp.asarray(self.Rpinv_prev_eval_t)
                
                # # Give Rpinv the last solution, if it is an attribute
                # if hasattr(Rpinv, 'prev_eval'):
                #     if self.Rpinv_prev_eval is None:
                #         pass
                #     else:
                #         if Rpinv.device == "cpu":
                #             Rpinv.Rwpinv.prev_eval = self.Rpinv_prev_eval
                #             Rpinv.Rwpinv.prev_eval_t = self.Rpinv_prev_eval_t
                #         else:
                #             Rpinv.Rwpinv.prev_eval = cp.asarray(self.Rpinv_prev_eval)
                #             Rpinv.Rwpinv.prev_eval_t = cp.asarray(self.Rpinv_prev_eval_t)

                Ftilde = Ftilde.to_gpu()
                Rtilde = Rtilde.to_gpu()
                ytilde = cp.asarray(ytilde)
                x_prev =  cp.asarray(x_prev)
                Awpinv = self.pdata["FWpinv"].to_gpu()
                W = self.pdata["W"].to_gpu()

                print(Rpinv.Rwpinv.prev_eval)
                # Solve using transformed CGLS
                cgls_solve = jlinops.trlstsq_standard_form(Ftilde, ytilde, Rpinv=Rpinv, R=Rtilde,
                                                           AWpinv=noise_stdev*Awpinv, lam=1.0, shift=None, W=W, initialization=x_prev, **x_update_solver_params)

                x = cp.asnumpy(cgls_solve["x"])

             
                # Only if it has this attribute
                Rpinv_cg_matvecs = None
                Rpinv_cg_rmatvecs = None
                if hasattr(Rpinv, 'matvec_cg_hist'):
                    Rpinv_cg_matvecs = sum(Rpinv.matvec_cg_hist)
                    Rpinv_cg_rmatvecs = sum(Rpinv.rmatvec_cg_hist)

                # Give Rpinv the last solution, if it is an attribute
                # tmp = cp.asnumpy(Rpinv.prev_eval)
                # self.Rpinv_prev_eval = tmp - (self.pdata["W"] @ (self.pdata["Wpinv"] @ tmp ) )
                # tmp = cp.asnumpy(Rpinv.prev_eval_t)
                # self.Rpinv_prev_eval_t = tmp - (self.pdata["W"] @ (self.pdata["Wpinv"] @ tmp ) )
                
                # if hasattr(Rpinv, 'prev_eval'):
                #     if Rpinv.device == "cpu":
                #         self.Rpinv_prev_eval = Rpinv.Rwpinv.prev_eval
                #         self.Rpinv_prev_eval_t = Rpinv.Rwpinv.prev_eval_t
                #     else:
                #         self.Rpinv_prev_eval = cp.asnumpy(Rpinv.Rwpinv.prev_eval)
                #         self.Rpinv_prev_eval_t = cp.asnumpy(Rpinv.Rwpinv.prev_eval_t)
                        
                 
                return x, cgls_solve["n_iters"], Rpinv_cg_matvecs, Rpinv_cg_rmatvecs
            
        
        
    def theta_update(self, x):
        """Returns the theta-update for fixed x.
        """

        r, beta, vartheta = self.hyperparams["prior"]["r"], self.hyperparams["prior"]["beta"], self.hyperparams["prior"]["vartheta"]
        eta = r*beta - 1.5
        initial_value = (eta/r)**(1/r)

        # Rescale
        z = np.abs(  ( self.R @ (x) )  / np.sqrt(vartheta) )

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
            ode_sol = odeint(self._conditional_mode_ode_rhs,
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


    def noise_var_update(self, x):
        """Returns the theta-update for fixed x.
        """

        r, beta, vartheta = self.hyperparams["noise_var"]["r"], self.hyperparams["noise_var"]["beta"], self.hyperparams["noise_var"]["vartheta"]
        eta = r*beta - ((self.F.shape[0]+2)/2)
        initial_value = (eta/r)**(1/r)

        # Rescale
        z = np.abs(  np.linalg.norm(self.F @ x - self.y) / np.sqrt(vartheta) )

        # If r = 1, use exact formula
        if abs(r - 1) < 1e-5:    
                # print("Using exact formula for r = 1")
                xi = 0.5*( eta + np.sqrt( (eta**2) +  2*(z**2)  ) )
                new_theta = vartheta*xi
                
                return new_theta

        elif abs(r + 1) < 1e-5:
                
                # print("Using exact formula for r = -1")
                
                k = beta + ((self.F.shape[0]+2)/2)
                xi = (1/(2*k)) * ( (z**2) + 2 )
                new_theta = vartheta*xi
                
                return new_theta

        # Otherwise, solve using the ODE method
        else:
            
            #print("Using ODE method")
            
            # final_times = z

            # # Sort the final times
            # argsort = final_times.argsort()
            # final_times_sorted = final_times[argsort]

            final_times_sorted = [z]

            # We need to prepend zero to this for the solver
            final_times_sorted = np.insert(final_times_sorted, 0, 0)

            # Now solve the ODE
            ode_sol = odeint(self._conditional_mode_ode_rhs,
                                np.atleast_1d(initial_value), 
                                final_times_sorted,
                                args=(r,) # r parameter
                            )

            #print(f"ODE sol: {ode_sol}")

            # Reshape and drop the first value corresponding to the dummy initial value
            ode_sol = ode_sol[1:,0]
            xi = ode_sol[0]
            new_noise_var = vartheta*xi

            return new_noise_var




    def _conditional_mode_ode_rhs(self, varphi, z, r):
        """RHS of the ODE used for updating theta.
        """
        
        dvarphidz = (2*z*varphi)/((2*(r**2)*((varphi)**(r+1))) + (z**2))

        return dvarphidz
    

    
    
    def objective(self, x, theta, noise_var):
        """Evaluates the objective function.
        """
        
        r, beta, vartheta = self.hyperparams["prior"]["r"], self.hyperparams["prior"]["beta"], self.hyperparams["prior"]["vartheta"]
        eta = (r*beta) - 1.5

        # Assemble terms
        gauss_like = (0.5/noise_var)*(np.linalg.norm( self.F.matvec(x) - self.y )**2)
        cond_pr = 0.5*((np.linalg.norm( (1.0/np.sqrt(theta))*(self.R @ x)  ))**2)
        hyperpr = ((theta/vartheta)**(r)).sum() - eta*(np.log(theta)).sum()
        
        # Sum together
        obj = gauss_like + cond_pr + hyperpr
        
        # Add part for noise var if needed
        if self.updating_noise_var:
            r, beta, vartheta = self.hyperparams["noise_var"]["r"], self.hyperparams["noise_var"]["beta"], self.hyperparams["noise_var"]["vartheta"]
            eta = (r*beta) - ((self.m + 2)/2)
            noise_contrib = ((noise_var/vartheta)**r) - eta*np.log(noise_var)
            obj += noise_contrib
        
        return obj


    
    
    def check_stopping_criterion(self, theta_prev, theta_curr, noise_var_prev, noise_var_curr, eps=1e-3):
        """Our stopping criterion is whenever the relative norm change in thetas (and noise var, if applicable)
        falls below eps, but each separately (unanimous vote).
        """
        
        theta_check = (np.linalg.norm(theta_curr - theta_prev)/np.linalg.norm(theta_prev)) < eps
        noise_var_check = True
        if self.updating_noise_var:
            noise_var_check = (np.abs(noise_var_prev - noise_var_curr)/noise_var_prev) < eps
        
        if theta_check and noise_var_check:
            return True
        else:
            return False
        




