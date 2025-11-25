"""Fitting functions for detector model parameters using iminuit"""
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares


def mean_model(E0, lambda_param, Delta):
    """
    Linear model for mean energy.
    
    μ_E = λ * E_0 + Δ
    
    Parameters
    ----------
    E0 : array-like
        True energy values
    lambda_param : float
        Scale factor
    Delta : float
        Offset
        
    Returns
    -------
    mu : array-like
        Predicted mean values
    """
    return lambda_param * E0 + Delta


def resolution_model(E0, a, b, c):
    """
    Resolution model for detector.
    
    σ_E / E_0 = sqrt(a²/E_0 + b²/E_0² + c²)
    
    Parameters
    ----------
    E0 : array-like
        True energy values
    a : float
        Stochastic term
    b : float
        Noise term
    c : float
        Constant term
        
    Returns
    -------
    sigma_over_E : array-like
        Predicted σ/E_0 values
    """
    term1 = (a / np.sqrt(E0))**2
    term2 = (b / E0)**2
    term3 = c**2
    return np.sqrt(term1 + term2 + term3)


def fit_mean_parameters(E0_values, means, mean_errors):
    """
    Fit linear model to mean values using iminuit.
    
    Parameters
    ----------
    E0_values : array-like
        True energy values
    means : array-like
        Sample mean values
    mean_errors : array-like
        Errors on means
        
    Returns
    -------
    minuit : Minuit object
        Fitted Minuit object with results
    params : dict
        Dictionary with parameter values
    errors : dict
        Dictionary with parameter errors
    """
    # Create least squares cost function
    least_squares = LeastSquares(E0_values, means, mean_errors, mean_model)
    
    # Create Minuit object
    m = Minuit(least_squares, lambda_param=1.0, Delta=0.0)
    
    # Set parameter limits (optional but good practice)
    m.limits['lambda_param'] = (0.5, 1.5)  # reasonable range
    m.limits['Delta'] = (-5, 5)  # reasonable range
    
    # Run minimization
    m.migrad()  # Find minimum
    m.hesse()   # Calculate parabolic errors
    
    # Extract results
    params = {
        'lambda': m.values['lambda_param'],
        'Delta': m.values['Delta']
    }
    
    errors = {
        'lambda': m.errors['lambda_param'],
        'Delta': m.errors['Delta']
    }
    print(m)
    return m, params, errors


def fit_resolution_parameters(E0_values, stds, std_errors):
    """
    Fit resolution model to standard deviation values using iminuit.
    
    Parameters
    ----------
    E0_values : array-like
        True energy values
    stds : array-like
        Sample std values
    std_errors : array-like
        Errors on stds
        
    Returns
    -------
    minuit : Minuit object
        Fitted Minuit object with results
    params : dict
        Dictionary with parameter values
    errors : dict
        Dictionary with parameter errors
    """
    # Convert to σ/E_0
    sigma_over_E = np.array(stds) / np.array(E0_values)
    sigma_over_E_err = np.array(std_errors) / np.array(E0_values)
    
    # Create least squares cost function
    least_squares = LeastSquares(
        E0_values, 
        sigma_over_E, 
        sigma_over_E_err, 
        resolution_model
    )
    
    # Create Minuit object with initial guesses
    m = Minuit(least_squares, a=0.15, b=0.5, c=0.01)
    
    # Set limits - all parameters must be positive
    m.limits['a'] = (0, None)
    m.limits['b'] = (0, None)
    m.limits['c'] = (0, None)
    
    # Run minimization
    m.migrad()  # Find minimum
    m.hesse()   # Calculate parabolic errors
    print(m)
    # Extract results
    params = {
        'a': m.values['a'],
        'b': m.values['b'],
        'c': m.values['c']
    }
    
    errors = {
        'a': m.errors['a'],
        'b': m.errors['b'],
        'c': m.errors['c']
    }
    
    return m, params, errors


def bootstrap_fit(grouped_data, n_bootstrap=100):
    """
    Bootstrap analysis for fitting uncertainty.
    
    Parameters
    ----------
    grouped_data : dict
        Dictionary from group_by_energy()
    n_bootstrap : int
        Number of bootstrap samples
        
    Returns
    -------
    results : dict
        Bootstrap results with parameter distributions
    """
    lambda_vals = []
    Delta_vals = []
    a_vals = []
    b_vals = []
    c_vals = []
    
    E0_list = sorted(grouped_data.keys())
    
    for i in range(n_bootstrap):
        # Resample each energy group
        boot_means = []
        boot_stds = []
        
        for E0 in E0_list:
            E_rec = grouped_data[E0]['E_rec']
            # Resample with replacement
            boot_sample = np.random.choice(E_rec, size=len(E_rec), replace=True)
            boot_means.append(np.mean(boot_sample))
            boot_stds.append(np.std(boot_sample, ddof=1))
        
        # Fit to bootstrap sample
        try:
            # Calculate errors for this bootstrap sample
            mean_err = [np.std(grouped_data[E0]['E_rec'], ddof=1) / np.sqrt(len(grouped_data[E0]['E_rec'])) 
                       for E0 in E0_list]
            std_err = [np.std(grouped_data[E0]['E_rec'], ddof=1) / np.sqrt(2*(len(grouped_data[E0]['E_rec'])-1))
                      for E0 in E0_list]
            
            # Fit mean using iminuit
            _, mean_params, _ = fit_mean_parameters(E0_list, boot_means, mean_err)
            
            # Fit resolution using iminuit
            _, res_params, _ = fit_resolution_parameters(E0_list, boot_stds, std_err)
            
            lambda_vals.append(mean_params['lambda'])
            Delta_vals.append(mean_params['Delta'])
            a_vals.append(res_params['a'])
            b_vals.append(res_params['b'])
            c_vals.append(res_params['c'])
        except Exception as e:
            # Skip failed fits
            print(f"Bootstrap sample {i} failed: {e}")
            continue
    
    return {
        'lambda': np.array(lambda_vals),
        'Delta': np.array(Delta_vals),
        'a': np.array(a_vals),
        'b': np.array(b_vals),
        'c': np.array(c_vals)
    }


def plot_fits_with_bands(E0_values, means, mean_errors, stds, std_errors,
                         mean_params, resolution_params, bootstrap_results=None):
    """
    Create Figure 1.4/2.2 with fitted curves and error bands.
    
    Parameters
    ----------
    E0_values : array-like
        True energy values
    means, mean_errors : array-like
        Sample means and errors
    stds, std_errors : array-like
        Sample stds and errors
    mean_params : dict
        {'lambda': value, 'Delta': value} from fit
    resolution_params : dict
        {'a': value, 'b': value, 'c': value} from fit
    bootstrap_results : dict, optional
        Bootstrap parameter distributions
        
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
    
    # Create fine grid for smooth curves
    E0_fine = np.linspace(min(E0_values), max(E0_values), 200)
    
    # --- Left plot: Mean - E0 ---
    lambda_param = mean_params['lambda']
    Delta = mean_params['Delta']
    
    residuals_mean = np.array(means) - np.array(E0_values)
    predicted_residuals = mean_model(E0_fine, lambda_param, Delta) - E0_fine
    
    ax[0].errorbar(E0_values, residuals_mean, yerr=mean_errors,
                   fmt='o', capsize=5, label='Data', markersize=8, color='#1f77b4')
    ax[0].plot(E0_fine, predicted_residuals, 'r-', linewidth=2, 
               label=f'Fit: λ={lambda_param:.4f}, Δ={Delta:.3f}')
    
    # Correct Error Band Calculation
    if bootstrap_results is not None:
        # Calculate curves for ALL bootstrap samples
        boot_curves = []
        for i in range(len(bootstrap_results['lambda'])):
            l_i = bootstrap_results['lambda'][i]
            d_i = bootstrap_results['Delta'][i]
            curve = mean_model(E0_fine, l_i, d_i) - E0_fine
            boot_curves.append(curve)
        
        boot_curves = np.array(boot_curves)
        # Calculate std dev of the curves at each point
        curve_std = np.std(boot_curves, axis=0)
        
        # Band is mean prediction +/- curve std
        upper = predicted_residuals + curve_std
        lower = predicted_residuals - curve_std
        
        ax[0].fill_between(E0_fine, lower, upper, alpha=0.3, color='red', label='±1σ band')
    
    ax[0].set_xlabel(r'$E_0$ [GeV]', fontsize=12)
    ax[0].set_ylabel(r'$\hat{\mu} - E_0$ [GeV]', fontsize=12)
    ax[0].set_title('Mean Residuals vs True Energy', fontsize=13, fontweight='bold')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(fontsize=9, loc='best')
    
    # --- Right plot: σ/E0 ---
    a = resolution_params['a']
    b = resolution_params['b']
    c = resolution_params['c']
    
    sigma_over_E = np.array(stds) / np.array(E0_values)
    sigma_over_E_err = np.array(std_errors) / np.array(E0_values)
    predicted_resolution = resolution_model(E0_fine, a, b, c)
    
    ax[1].errorbar(E0_values, sigma_over_E, yerr=sigma_over_E_err,
                   fmt='s', capsize=5, color='orange', label='Data', markersize=8)
    ax[1].plot(E0_fine, predicted_resolution, 'r-', linewidth=2,
               label=f'Fit: a={a:.3f}, b={b:.3f}, c={c:.4f}')
    
    # Correct Error Band Calculation
    if bootstrap_results is not None:
        boot_curves = []
        for i in range(len(bootstrap_results['a'])):
            a_i = bootstrap_results['a'][i]
            b_i = bootstrap_results['b'][i]
            c_i = bootstrap_results['c'][i]
            curve = resolution_model(E0_fine, a_i, b_i, c_i)
            boot_curves.append(curve)
            
        boot_curves = np.array(boot_curves)
        curve_std = np.std(boot_curves, axis=0)
        
        upper = predicted_resolution + curve_std
        lower = predicted_resolution - curve_std
        
        ax[1].fill_between(E0_fine, lower, upper, alpha=0.3, color='red', label='±1σ band')
    
    ax[1].set_xlabel(r'$E_0$ [GeV]', fontsize=12)
    ax[1].set_ylabel(r'$\hat{\sigma} / E_0$', fontsize=12)
    ax[1].set_title('Relative Resolution vs True Energy', fontsize=13, fontweight='bold')
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(fontsize=9, loc='best')
    
    fig.tight_layout()
    return fig, ax


def print_fit_results(mean_minuit, resolution_minuit):
    """
    Print detailed fit results from Minuit objects.
    
    Parameters
    ----------
    mean_minuit : Minuit
        Fitted Minuit object for mean
    resolution_minuit : Minuit
        Fitted Minuit object for resolution
    """
    print("\n" + "="*70)
    print("FIT RESULTS SUMMARY (using iminuit)")
    print("="*70)
    
    print("\n--- MEAN FIT (μ = λ·E₀ + Δ) ---")
    print(mean_minuit)
    print(f"\nχ²/ndf = {mean_minuit.fval:.2f}/{mean_minuit.ndof}")
    print(f"Fit quality: {'GOOD' if mean_minuit.valid else 'POOR'}")
    
    print("\n--- RESOLUTION FIT (σ/E₀ = √(a²/E₀ + b²/E₀² + c²)) ---")
    print(resolution_minuit)
    print(f"\nχ²/ndf = {resolution_minuit.fval:.2f}/{resolution_minuit.ndof}")
    print(f"Fit quality: {'GOOD' if resolution_minuit.valid else 'POOR'}")
    
    print("\n" + "="*70)


def bootstrap_mle_trends(grouped_data, n_bootstrap=100):
    """
    Perform full bootstrap analysis for MLE trends (Exercise 2ii).
    
    1. Resample data for each E0
    2. Fit Gaussian to resampled data (get mu, sigma)
    3. Fit trends to these mu, sigma
    4. Repeat
    
    Parameters
    ----------
    grouped_data : dict
        Data dictionary
    n_bootstrap : int
        Number of iterations
        
    Returns
    -------
    boot_results : dict
        Distributions of parameters lambda, Delta, a, b, c
    """
    from s1_sol import mle_fits
    
    boot_results = {'lambda': [], 'Delta': [], 'a': [], 'b': [], 'c': []}
    E0_arr = np.array(sorted(grouped_data.keys()))
    
    print(f"Running bootstrap with {n_bootstrap} iterations...")
    
    for i in range(n_bootstrap):
        # Temporary lists for this iteration
        b_means, b_mean_errs = [], []
        b_stds, b_std_errs = [], []
        
        try:
            for E0 in E0_arr:
                # 1. Resample
                data = grouped_data[E0]['E_rec']
                resample = np.random.choice(data, size=len(data), replace=True)
                
                # 2. Fit Gaussian (MLE)
                # We suppress warnings/prints here for speed
                _, p, e = mle_fits.fit_gaussian_for_energy(resample, E0)
                
                b_means.append(p['mu'])
                b_mean_errs.append(e['mu'])
                b_stds.append(p['sigma'])
                b_std_errs.append(e['sigma'])
            
            # 3. Fit Trends
            _, mp, _ = fit_mean_parameters(E0_arr, b_means, b_mean_errs)
            _, rp, _ = fit_resolution_parameters(E0_arr, b_stds, b_std_errs)
            
            boot_results['lambda'].append(mp['lambda'])
            boot_results['Delta'].append(mp['Delta'])
            boot_results['a'].append(rp['a'])
            boot_results['b'].append(rp['b'])
            boot_results['c'].append(rp['c'])
            
        except Exception:
            continue
            
    return boot_results

