"""
JAX-based spectrum fitting module
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.optimize import minimize as scipy_minimize
import warnings


@dataclass
class FitResult:
    """Container for fit results"""
    success: bool
    parameters: Dict[str, float]
    parameter_errors: Optional[Dict[str, float]]
    statistic: float
    reduced_statistic: float
    dof: int
    nfev: int
    message: str
    covariance: Optional[np.ndarray] = None
    free_param_names: Optional[List[str]] = None
    

class Fitter:
    """JAX-based spectrum fitter"""
    
    def __init__(self,
                 response_obj=None,
                 model_obj=None,
                 spectrum_obj=None):
        """
        Initialize fitter with Response, Model, and Spectrum objects
        """
        self.response_obj = response_obj
        self.model_obj = model_obj
        self.spectrum_obj = spectrum_obj
        self._mask = np.ones(self.response_obj.ebounds_centroid.size, dtype=bool)
        
        # Use JAX arrays from response object
        self._drm_jax = self.response_obj.drm_jax
        self._energy_jax = self.response_obj.energy_centroid_jax
        
        # Prepare data
        self._prepare_data()
        
        # Initialize mask from spectrum quality flags
        if self.spectrum_obj.quality.size > 0:
            self._mask = self.spectrum_obj.get_good_channels()
        else:
            self._mask = np.ones(self.response_obj.n_channels, dtype=bool)
        
        # Create JIT-compiled functions
        self._create_jit_functions()

    def _prepare_data(self):
        """Prepare data for fitting"""
        # Use JAX arrays from spectrum object
        self._net_counts = self.spectrum_obj.net_counts_jax / self.spectrum_obj.exposure
        self._net_errors = self.spectrum_obj.net_error_jax / self.spectrum_obj.exposure
        
        # Precompute variance for chi-square
        self._variance = self._net_errors ** 2
        
        # Apply any grouping if present
        if self.spectrum_obj.grouping.size > 0 and np.any(self.spectrum_obj.grouping != 1):
            self._apply_grouping()
    def _apply_grouping(self):
        """Apply spectrum grouping for fitting"""
        grouped = self.spectrum_obj.apply_grouping()
        self._grouped_counts = jnp.array(grouped['counts'] / self.spectrum_obj.exposure)
        self._grouped_errors = jnp.array(grouped['error'] / self.spectrum_obj.exposure)
        self._grouped_variance = self._grouped_errors ** 2
        
        # Need to also group the response matrix
        # This is a simplified approach - full implementation would properly
        # handle the response matrix grouping
        warnings.warn("Grouped fitting not fully implemented for response matrix")
    
    def set_energy_range(self, energy_ranges):
        """Set energy range for fitting (convenience method)"""
        self.mask = energy_ranges
        
    def _create_jit_functions(self):
        """Create JIT-compiled functions for fitting"""
        
        @jax.jit
        def forward_fold_model(params: jnp.ndarray) -> jnp.ndarray:
            """Forward fold model through response matrix"""
            # Evaluate model at photon energies
            photon_flux = self.model_obj.function_free_params(params, self._energy_jax)
            # Fold through response matrix
            count_flux = jnp.dot(photon_flux, self._drm_jax)
            return count_flux
        
        @jax.jit
        def chi_square_statistic(params: jnp.ndarray) -> float:
            """Chi-square statistic"""
            model = forward_fold_model(params)
            
            # Apply mask
            model_masked = model[self._mask]
            data_masked = self._net_counts[self._mask]
            variance_masked = self._variance[self._mask]
            
            chi2 = jnp.sum((data_masked - model_masked)**2 / variance_masked)
            return chi2
        
        @jax.jit
        def pgstat_statistic(params: jnp.ndarray) -> float:
            """Profile Gaussian (pgstat) statistic for Poisson data with Gaussian background"""
            model = forward_fold_model(params)
            
            # Apply mask
            model_masked = model[self._mask]
            data_masked = self._net_counts[self._mask]
            
            # For now, simplified pgstat (can be expanded with full implementation later)
            # This is a simplified version - full pgstat would need background terms
            stat = 2 * jnp.sum(model_masked - data_masked * jnp.log(model_masked + 1e-20))
            return stat
        
        # Store JIT-compiled functions
        self._forward_fold_model = forward_fold_model
        self._chi_square_statistic = chi_square_statistic
        self._pgstat_statistic = pgstat_statistic
        
        # Create gradient functions
        self._chi_square_gradient = jax.grad(chi_square_statistic)
        self._pgstat_gradient = jax.grad(pgstat_statistic)
        """Create JIT-compiled functions for fitting"""
        
        @jax.jit
        def forward_fold_model(params: jnp.ndarray) -> jnp.ndarray:
            """Forward fold model through response matrix"""
            # Evaluate model at photon energies
            photon_flux = self.model_obj.function_free_params(params, self._energy_jax)
            # Fold through response matrix
            count_flux = jnp.dot(photon_flux, self._drm_jax)
            return count_flux
        
        @jax.jit
        def chi_square_statistic(params: jnp.ndarray) -> float:
            """Chi-square statistic"""
            model = forward_fold_model(params)
            
            # Apply mask
            model_masked = model[self._mask]
            data_masked = self._net_counts[self._mask]
            variance_masked = self._variance[self._mask]
            
            chi2 = jnp.sum((data_masked - model_masked)**2 / variance_masked)
            return chi2
        
        @jax.jit
        def pgstat_statistic(params: jnp.ndarray) -> float:
            """Profile Gaussian (pgstat) statistic for Poisson data with Gaussian background"""
            model = forward_fold_model(params)
            
            # Apply mask
            model_masked = model[self._mask]
            data_masked = self._net_counts[self._mask]
            
            # For now, simplified pgstat (can be expanded with full implementation later)
            # This is a simplified version - full pgstat would need background terms
            stat = 2 * jnp.sum(model_masked - data_masked * jnp.log(model_masked + 1e-20))
            return stat
        
        # Store JIT-compiled functions
        self._forward_fold_model = forward_fold_model
        self._chi_square_statistic = chi_square_statistic
        self._pgstat_statistic = pgstat_statistic
        
        # Create gradient functions
        self._chi_square_gradient = jax.grad(chi_square_statistic)
        self._pgstat_gradient = jax.grad(pgstat_statistic)

    def __str__(self):
        return f"""Prepared to fit the spectrum:
        Spectrum: {self.spectrum_obj}
        Model: {self.model_obj}
        Response: {self.response_obj}"""

    @property
    def response_obj(self):
        return self._response_obj
    
    @response_obj.setter
    def response_obj(self, Response):
        self._response_obj = Response

    @property
    def model_obj(self):
        return self._model_obj
    
    @model_obj.setter
    def model_obj(self, Model):
        self._model_obj = Model

    @property
    def spectrum_obj(self):
        return self._spectrum_obj
    
    @spectrum_obj.setter
    def spectrum_obj(self, Spectrum):
        self._spectrum_obj = Spectrum

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, energy_list):
        """Set energy mask
        energy_list = [(10, 20), (30, 100)]
        --> select 10keV -- 20 keV, and 30--100 keV"""
        mask = np.zeros(self.response_obj.ebounds_centroid.size, dtype=bool)
        for energy_range in energy_list:
            mask = mask | ((self.response_obj.ebounds_centroid >= energy_range[0]) & 
                          (self.response_obj.ebounds_centroid <= energy_range[1]))
        self._mask = mask

    def fit(self, stat='chi2', method='L-BFGS-B', options=None, use_bounds=True):
        """
        Perform fitting using JAX-based optimization
        
        Parameters
        ----------
        stat : str
            Statistic to use ('chi2' or 'pgstat')
        method : str
            Optimization method:
            - 'L-BFGS-B': Limited-memory BFGS with bounds (scipy)
            - 'adam': Adam optimizer (JAX/optax)
            - 'sgd': Stochastic gradient descent (JAX/optax)
            - 'BFGS': BFGS without bounds (scipy)
        options : dict
            Options for optimizer
        use_bounds : bool
            Whether to use parameter bounds
            
        Returns
        -------
        FitResult
            Object containing fit results
        """
        # Select statistic
        if stat == 'chi2':
            loss_fn = self._chi_square_statistic
            grad_fn = self._chi_square_gradient
        elif stat == 'pgstat':
            loss_fn = self._pgstat_statistic
            grad_fn = self._pgstat_gradient
        else:
            raise ValueError(f"Unknown statistic: {stat}")
        
        # Get initial parameters
        initial_params = self.model_obj.get_free_param_values()
        param_names = self.model_obj.get_free_param_names()
        
        # Get bounds if requested
        if use_bounds:
            lower_bounds, upper_bounds = self.model_obj.get_free_param_bounds()
        
        print(f"Performing {stat} fitting with {method} optimizer...")
        
        if method in ['L-BFGS-B', 'BFGS']:
            # Use scipy for traditional optimization
            result = self._fit_scipy(loss_fn, grad_fn, initial_params, 
                                   lower_bounds if use_bounds else None,
                                   upper_bounds if use_bounds else None,
                                   method=method, options=options)
        else:
            # Use JAX/optax for gradient-based optimization
            result = self._fit_jax(loss_fn, grad_fn, initial_params,
                                 lower_bounds if use_bounds else None,
                                 upper_bounds if use_bounds else None,
                                 method=method, options=options)
        
        # Add parameter names to result
        result.free_param_names = param_names
        
        return result

    def _fit_scipy(self, loss_fn, grad_fn, initial_params, 
                   lower_bounds=None, upper_bounds=None, 
                   method='L-BFGS-B', options=None):
        """Fit using scipy optimizers"""
        
        # Convert to numpy for scipy
        def loss_np(params):
            return float(loss_fn(jnp.array(params)))
        
        def grad_np(params):
            return np.array(grad_fn(jnp.array(params)))
        
        # Setup bounds
        if lower_bounds is not None and upper_bounds is not None:
            bounds = list(zip(lower_bounds, upper_bounds))
        else:
            bounds = None
        
        # Default options
        if options is None:
            options = {'maxiter': 1000, 'ftol': 1e-8}
        
        # Optimize
        result = scipy_minimize(
            loss_np,
            initial_params,
            method=method,
            jac=grad_np,
            bounds=bounds,
            options=options
        )
        
        # Update model with best-fit parameters
        self.model_obj.update_free_params(result.x)
        
        # Calculate degrees of freedom
        n_data = np.sum(self._mask)
        n_free_params = len(initial_params)
        dof = n_data - n_free_params
        
        # Create FitResult
        fit_result = FitResult(
            success=result.success,
            parameters=self.model_obj.get_param_dict(),
            parameter_errors=None,  # Will be calculated if needed
            statistic=float(result.fun),
            reduced_statistic=float(result.fun) / dof if dof > 0 else np.inf,
            dof=dof,
            nfev=result.nfev,
            message=result.message,
            covariance=None,
            free_param_names=self.model_obj.get_free_param_names()
        )
        
        # Calculate parameter errors if requested
        if result.success:
            fit_result = self._calculate_errors(fit_result, loss_fn, result.x)
        
        return fit_result

    def _fit_jax(self, loss_fn, grad_fn, initial_params,
                 lower_bounds=None, upper_bounds=None,
                 method='adam', options=None):
        """Fit using JAX/optax optimizers"""
        
        # Default options
        if options is None:
            options = {
                'learning_rate': 0.01,
                'max_steps': 10000,
                'tolerance': 1e-6
            }
        
        # Select optimizer
        if method == 'adam':
            optimizer = optax.adam(options.get('learning_rate', 0.01))
        elif method == 'sgd':
            optimizer = optax.sgd(options.get('learning_rate', 0.01))
        else:
            raise ValueError(f"Unknown JAX optimizer: {method}")
        
        # Initialize optimizer state
        opt_state = optimizer.init(initial_params)
        params = initial_params.copy()
        
        # Training loop
        max_steps = options.get('max_steps', 10000)
        tolerance = options.get('tolerance', 1e-6)
        prev_loss = np.inf
        
        for step in range(max_steps):
            # Compute gradient
            grads = grad_fn(params)
            
            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            # Apply bounds if specified
            if lower_bounds is not None and upper_bounds is not None:
                params = jnp.clip(params, lower_bounds, upper_bounds)
            
            # Check convergence
            if step % 100 == 0:
                current_loss = float(loss_fn(params))
                if abs(prev_loss - current_loss) < tolerance:
                    print(f"Converged at step {step}")
                    break
                prev_loss = current_loss
        
        # Update model with best-fit parameters
        self.model_obj.update_free_params(params)
        
        # Calculate degrees of freedom
        n_data = np.sum(self._mask)
        n_free_params = len(initial_params)
        dof = n_data - n_free_params
        
        # Create FitResult
        final_statistic = float(loss_fn(params))
        fit_result = FitResult(
            success=True,
            parameters=self.model_obj.get_param_dict(),
            parameter_errors=None,
            statistic=final_statistic,
            reduced_statistic=final_statistic / dof if dof > 0 else np.inf,
            dof=dof,
            nfev=step,
            message=f"JAX optimization completed in {step} steps",
            covariance=None,
            free_param_names=self.model_obj.get_free_param_names()
        )
        
        # Calculate parameter errors
        fit_result = self._calculate_errors(fit_result, loss_fn, params)
        
        return fit_result

    def _calculate_errors(self, fit_result: FitResult, loss_fn: Callable, 
                         best_params: jnp.ndarray) -> FitResult:
        """Calculate parameter errors using Hessian"""
        
        # Compute Hessian
        hessian_fn = jax.hessian(loss_fn)
        hessian = hessian_fn(best_params)
        
        # Convert to numpy and ensure it's symmetric
        hessian_np = np.array(hessian)
        hessian_np = 0.5 * (hessian_np + hessian_np.T)
        
        try:
            # Invert to get covariance matrix
            # For chi-square, covariance = 2 * inverse(Hessian)
            covariance = 2 * np.linalg.inv(hessian_np)
            
            # Extract diagonal for parameter errors
            param_errors = np.sqrt(np.diag(covariance))
            
            # Create error dictionary
            param_names = fit_result.free_param_names
            error_dict = {}
            all_params = self.model_obj.get_param_dict()
            
            # Initialize all errors to 0
            for param_name in all_params:
                error_dict[param_name] = 0.0
            
            # Fill in errors for free parameters
            for i, param_name in enumerate(param_names):
                error_dict[param_name] = float(param_errors[i])
            
            fit_result.parameter_errors = error_dict
            fit_result.covariance = covariance
            
        except np.linalg.LinAlgError:
            warnings.warn("Could not compute parameter errors: Hessian is singular")
            fit_result.parameter_errors = None
            fit_result.covariance = None
        
        return fit_result

    def plot_fit(self, ax=None, plot_data=True, plot_model=True, 
                 plot_residuals=True, energy_unit='keV'):
        """Plot the fit results"""
        if ax is None:
            if plot_residuals:
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, 
                                               gridspec_kw={'height_ratios': [3, 1]})
            else:
                fig, ax1 = plt.subplots(1, 1)
                ax2 = None
        else:
            ax1 = ax
            ax2 = None
        
        # Get energy bins
        energy = self.response_obj.ebounds_centroid
        energy_low = self.response_obj.ebounds_low
        energy_high = self.response_obj.ebounds_high
        
        # Get data
        data = self.spectrum_obj.net_cspec / self.spectrum_obj.exposure
        errors = self.spectrum_obj.net_error / self.spectrum_obj.exposure
        
        # Get model
        params = self.model_obj.get_free_param_values()
        model = self._forward_fold_model(params)
        
        # Plot data
        if plot_data:
            ax1.errorbar(energy[self._mask], data[self._mask], 
                        yerr=errors[self._mask],
                        xerr=[energy[self._mask] - energy_low[self._mask],
                              energy_high[self._mask] - energy[self._mask]],
                        fmt='o', label='Data', capsize=0)
        
        # Plot model
        if plot_model:
            ax1.plot(energy[self._mask], model[self._mask], 
                    'r-', label='Model', linewidth=2)
        
        ax1.set_ylabel(f'Counts/s/{energy_unit}')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot residuals
        if plot_residuals and ax2 is not None:
            residuals = (data[self._mask] - model[self._mask]) / errors[self._mask]
            ax2.errorbar(energy[self._mask], residuals, yerr=1,
                        xerr=[energy[self._mask] - energy_low[self._mask],
                              energy_high[self._mask] - energy[self._mask]],
                        fmt='o', capsize=0)
            ax2.axhline(0, color='r', linestyle='--', alpha=0.5)
            ax2.set_ylabel('Residuals (σ)')
            ax2.set_xlabel(f'Energy ({energy_unit})')
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale('log')
        
        if ax is None:
            plt.tight_layout()
            return fig
        
    def confidence_intervals(self, param_names=None, confidence_level=0.68, 
                           method='profile'):
        """
        Calculate confidence intervals for parameters
        
        Parameters
        ----------
        param_names : list of str, optional
            Parameters to calculate intervals for (default: all free parameters)
        confidence_level : float
            Confidence level (default: 0.68 for 1-sigma)
        method : str
            Method to use ('profile' or 'hessian')
        """
        # Implementation would go here
        raise NotImplementedError("Confidence intervals not yet implemented")
