"""Data loading and preprocessing utilities"""
import numpy as np
from pathlib import Path

def load_data(filepath='sample.csv'):
    """
    Load the sample data from CSV.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file
        
    Returns
    -------
    E_true : ndarray
        True energy values
    E_rec : ndarray
        Reconstructed energy values
    """
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    E_true = data[:, 0]
    E_rec = data[:, 1]
    return E_true, E_rec


def group_by_energy(E_true, E_rec):
    """
    Group data by E_true values.
    
    Parameters
    ----------
    E_true : ndarray
        True energy values
    E_rec : ndarray
        Reconstructed energy values
        
    Returns
    -------
    grouped_data : dict
        Dictionary with E_true values as keys and (E_true_array, E_rec_array) as values
    """
    unique_energies = np.unique(E_true)
    grouped = {}
    
    for E0 in unique_energies:
        mask = E_true == E0
        grouped[E0] = {
            'E_true': E_true[mask],
            'E_rec': E_rec[mask],
            'n_events': np.sum(mask)
        }
    
    return grouped


def compute_residuals(E_true, E_rec):
    """
    Compute residuals (E_rec - E_true).
    
    Parameters
    ----------
    E_true : ndarray
        True energy values
    E_rec : ndarray
        Reconstructed energy values
        
    Returns
    -------
    residuals : ndarray
        E_rec - E_true
    """
    return E_rec - E_true