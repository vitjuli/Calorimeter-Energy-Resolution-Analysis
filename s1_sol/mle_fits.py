"""
Individual Maximum Likelihood Estimation fits.
"""
import numpy as np
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL

def gaussian_pdf(x, mu, sigma):
    """
    Gaussian probability density function.
    """
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

def fit_gaussian_for_energy(E_rec, E0_true):
    """
    Fit Gaussian to E_rec data for a single E0 value using UnbinnedNLL.
    
    Parameters
    ----------
    E_rec : array-like
        Reconstructed energy measurements
    E0_true : float
        True energy value (used for initial guess)
        
    Returns
    -------
    minuit : Minuit
        Fitted Minuit object
    params : dict
        {'mu': val, 'sigma': val}
    errors : dict
        {'mu': err, 'sigma': err}
    """
    # Create cost function
    nll = UnbinnedNLL(E_rec, gaussian_pdf)
    
    # Initial guesses
    # Use sample statistics for robust initial values
    init_mu = np.mean(E_rec)
    init_sigma = np.std(E_rec, ddof=1)
    
    m = Minuit(nll, mu=init_mu, sigma=init_sigma)
    
    # Limits
    m.limits['sigma'] = (0, None)
    
    # Minimize
    m.migrad()
    m.hesse()
    
    params = {
        'mu': m.values['mu'],
        'sigma': m.values['sigma']
    }
    
    errors = {
        'mu': m.errors['mu'],
        'sigma': m.errors['sigma']
    }
    
    return m, params, errors

def run_mle_fits(grouped_data):
    """
    Run MLE fits for all energy groups.
    
    Parameters
    ----------
    grouped_data : dict
        Dictionary of grouped data
        
    Returns
    -------
    results_mle : dict
        Dictionary with results for each E0
    """
    results_mle = {}
    
    for E0 in sorted(grouped_data.keys()):
        E_rec = grouped_data[E0]['E_rec']
        m, params, errors = fit_gaussian_for_energy(E_rec, E0)
        
        results_mle[E0] = {
            'mu': params['mu'],
            'sigma': params['sigma'],
            'mu_err': errors['mu'],
            'sigma_err': errors['sigma']
        }
        
    return results_mle
