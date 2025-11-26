"""Sample estimators for mean and standard deviation"""
import numpy as np


def calculate_sample_mean(data):
    """
    Calculate sample mean.
    
    Parameters
    ----------
    data : ndarray
        Data array
        
    Returns
    -------
    mean : float
        Sample mean
    """
    return np.mean(data)


def calculate_sample_std(data):
    """
    Calculate sample standard deviation.
    
    Parameters
    ----------
    data : ndarray
        Data array
        
    Returns
    -------
    std : float
        Sample standard deviation
    """
    return np.std(data, ddof=1)  # Use ddof=1 for sample std


def calculate_standard_error_mean(data):
    """
    Calculate standard error of the mean.
    
    Parameters
    ----------
    data : ndarray
        Data array
        
    Returns
    -------
    se_mean : float
        Standard error of the mean
    """
    n = len(data)
    return calculate_sample_std(data) / np.sqrt(n)


def calculate_standard_error_std(data):
    """
    Calculate standard error of the standard deviation.
    
    Parameters
    ----------
    data : ndarray
        Data array
        
    Returns
    -------
    se_std : float
        Standard error of the standard deviation
    """
    n = len(data)
    return calculate_sample_std(data) / np.sqrt(2 * (n - 1))


def estimate_parameters_by_energy(grouped_data):
    """
    Calculate sample estimates for each energy group.
    
    Parameters
    ----------
    grouped_data : dict
        Dictionary from group_by_energy()
        
    Returns
    -------
    results : dict
        Dictionary with E0 as keys and estimates as values
    """
    results = {}
    
    for E0, data in sorted(grouped_data.items()):
        E_rec = data['E_rec']
        
        mean = calculate_sample_mean(E_rec)
        std = calculate_sample_std(E_rec)
        se_mean = calculate_standard_error_mean(E_rec)
        se_std = calculate_standard_error_std(E_rec)
        
        results[E0] = {
            'mean': mean,
            'std': std,
            'se_mean': se_mean,
            'se_std': se_std,
            'n_events': len(E_rec)
        }
    
    return results


def calculate_jackknife_stats(data):
    """
    Calculate Jackknife statistics (bias-corrected mean/std and their variances).
    
    Parameters
    ----------
    data : array-like
        Input data
        
    Returns
    -------
    stats : dict
        Dictionary containing:
        - mean_jk: Bias-corrected mean
        - mean_var: Variance of the mean estimate
        - std_jk: Bias-corrected standard deviation
        - std_var: Variance of the std estimate
    """
    from resample import jackknife
    
    # Define statistic functions
    # Note: for std, we want sample std (ddof=1)
    def sample_std(x):
        return np.std(x, ddof=1)
        
    # Calculate for Mean
    # jackknife.bias_corrected returns the bias-corrected estimate
    mean_jk = jackknife.bias_corrected(np.mean, data)
    mean_var = jackknife.variance(np.mean, data)
    
    # Calculate for Std
    std_jk = jackknife.bias_corrected(sample_std, data)
    std_var = jackknife.variance(sample_std, data)
    
    return {
        'mean_jk': mean_jk,
        'mean_var': mean_var,
        'mean_err': np.sqrt(mean_var), # Standard error is sqrt(variance)
        'std_jk': std_jk,
        'std_var': std_var,
        'std_err': np.sqrt(std_var)
    }
