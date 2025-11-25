"""Plotting utilities for the S1 coursework"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set default style
plt.style.use('seaborn-v0_8-darkgrid')  # or 'ggplot', 'bmh', etc.

def setup_figure(figsize=(10, 6)):
    """Create a standard figure with nice defaults."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def save_figure(fig, filename, dpi=300):
    """
    Save figure to the figs directory.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Filename (e.g., 'Figure1.1.pdf')
    dpi : int
        Resolution for rasterized elements
    """
    filepath = Path('figs') / filename
    filepath.parent.mkdir(exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {filepath}")


def plot_residual_histogram(residuals, bins=50, ax=None, **kwargs):
    """
    Plot histogram of residuals.
    
    Parameters
    ----------
    residuals : ndarray
        Residual values (E_rec - E_true)
    bins : int or array-like
        Number of bins or bin edges
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    **kwargs : dict
        Additional arguments for plt.hist
        
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = setup_figure()
    
    ax.hist(residuals, bins=bins, alpha=0.7, edgecolor='black', **kwargs)
    ax.set_xlabel(r'$E_{\rm rec} - E_{\rm true}$ [GeV]')
    ax.set_ylabel('Counts')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_grouped_residuals(grouped_data, bins=50, figsize=(12, 8)):
    """
    Plot residuals for each E_true group.
    
    Parameters
    ----------
    grouped_data : dict
        Dictionary from group_by_energy()
    bins : int
        Number of bins
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = setup_figure(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(grouped_data)))
    
    for i, (E0, data) in enumerate(sorted(grouped_data.items())):
        residuals = data['E_rec'] - data['E_true']
        ax.hist(residuals, bins=bins, alpha=0.5, 
                label=f'$E_{{\\rm true}} = {E0:.0f}$ GeV',
                color=colors[i], edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel(r'$E_{\rm rec} - E_{\rm true}$ [GeV]')
    ax.set_ylabel('Counts')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax


def plot_mle_histograms(grouped_data, results_mle):
    """
    Plot Figure 2.1: Individual fits and total distribution.
    
    Parameters
    ----------
    grouped_data : dict
        Data dictionary
    results_mle : dict
        Results from run_mle_fits
        
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    from s1_sol import mle_fits
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    E0_list = sorted(grouped_data.keys())
    
    # --- Left Plot: Individual Fits ---
    colors = plt.cm.viridis(np.linspace(0, 1, len(E0_list)))

    for i, E0 in enumerate(E0_list):
        E_rec = grouped_data[E0]['E_rec']
        residuals = E_rec - E0
        
        # Data histogram
        ax[0].hist(residuals, bins=40, density=True, alpha=0.3, color=colors[i], label=f'{E0} GeV')
        
        # Fitted curve
        x = np.linspace(residuals.min(), residuals.max(), 100)
        mu_fit = results_mle[E0]['mu']
        sigma_fit = results_mle[E0]['sigma']
        
        # Shift x back to absolute energy for PDF calculation
        y = mle_fits.gaussian_pdf(x + E0, mu_fit, sigma_fit)
        ax[0].plot(x, y, color=colors[i], linewidth=2)

    ax[0].set_title("Individual Fits (Normalized)")
    ax[0].set_xlabel(r"$E_{rec} - E_0$ [GeV]")
    ax[0].legend(ncol=2, fontsize='small')
    ax[0].grid(True, alpha=0.3)

    # --- Right Plot: Total Distribution ---
    all_residuals = []
    for E0 in E0_list:
        all_residuals.extend(grouped_data[E0]['E_rec'] - E0)

    ax[1].hist(all_residuals, bins=50, density=True, color='gray', alpha=0.6, label='All Data')
    
    # Plot sum of fits (approximate)
    x_total = np.linspace(min(all_residuals), max(all_residuals), 200)
    y_total = np.zeros_like(x_total)
    total_events = len(all_residuals)
    
    for E0 in E0_list:
        mu = results_mle[E0]['mu']
        sigma = results_mle[E0]['sigma']
        n_events = len(grouped_data[E0]['E_rec'])
        weight = n_events / total_events
        
        y_total += weight * mle_fits.gaussian_pdf(x_total + E0, mu, sigma)
        
    ax[1].plot(x_total, y_total, 'r--', linewidth=2, label='Sum of Fits')
    
    ax[1].set_title("Total Residual Distribution")
    ax[1].set_xlabel(r"$E_{rec} - E_0$ [GeV]")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig, ax