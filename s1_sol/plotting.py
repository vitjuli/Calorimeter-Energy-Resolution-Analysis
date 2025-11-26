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


def plot_parameter_comparison(results_dict):
    """
    Plot comparison of parameters from different methods (Figure 3.2).
    
    Parameters
    ----------
    results_dict : dict
        Dictionary containing results from all 3 methods
        
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    methods = ['sample_ests', 'individual_fits', 'simultaneous_fit']
    method_labels = ['Sample Ests', 'Indiv Fits', 'Simult Fit']
    colors = ['#1f77b4', '#2ca02c', '#d62728'] # Blue, Green, Red
    
    params = ['lb', 'dE', 'a', 'b', 'c']
    param_labels = [r'$\lambda$', r'$\Delta$ [GeV]', r'$a$ [GeV$^{1/2}$]', r'$b$ [GeV]', r'$c$']
    
    # Create figure with 5 subplots in a row
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    
    for i, param in enumerate(params):
        ax = axes[i]
        
        # Collect values and errors for this parameter
        vals = []
        errs = []
        
        for method in methods:
            vals.append(results_dict[method]['values'][param])
            errs.append(results_dict[method]['errors'][param])
            
        # Plot points with error bars
        for j, (val, err) in enumerate(zip(vals, errs)):
            ax.errorbar(j, val, yerr=err, fmt='o', capsize=5, 
                        color=colors[j], label=method_labels[j], markersize=8)
            
        # Styling
        ax.set_title(param_labels[i], fontsize=14)
        ax.set_xticks([]) # Hide x ticks as they are meaningless
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adjust y-limits to show error bars clearly
        # Add some padding (e.g. 10% of range)
        y_min = min([v - e for v, e in zip(vals, errs)])
        y_max = max([v + e for v, e in zip(vals, errs)])
        y_range = y_max - y_min
        if y_range == 0: y_range = 1.0 # Prevent singular matrix if errors are 0
        
        ax.set_ylim(y_min - 0.5*y_range, y_max + 0.5*y_range)

    # Add legend to the last plot (or global)
    # We create a dummy legend for the whole figure
    handles = [plt.Line2D([0], [0], marker='o', color=c, label=l, linestyle='') 
               for c, l in zip(colors, method_labels)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=12)
    
    plt.tight_layout()
    return fig, axes


def plot_bootstrap_histograms(boot_results):
    """
    Plot histograms of bootstrapped parameters (Figure 4.1).
    
    Parameters
    ----------
    boot_results : dict
        Results from run_full_bootstrap
        
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    methods = ['sample_ests', 'individual_fits', 'simultaneous_fit']
    method_labels = ['Sample Ests', 'Indiv Fits', 'Simult Fit']
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    params = ['lambda', 'Delta', 'a', 'b', 'c']
    param_labels = [r'$\lambda$', r'$\Delta$', r'$a$', r'$b$', r'$c$']
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for i, param in enumerate(params):
        ax = axes[i]
        
        for j, method in enumerate(methods):
            values = boot_results[method][param]
            # Plot histogram
            ax.hist(values, bins=30, alpha=0.4, color=colors[j], 
                    label=method_labels[j] if i==0 else "", density=True)
            
        ax.set_title(param_labels[i])
        ax.grid(True, alpha=0.3)
        
    # Add legend to first plot
    axes[0].legend(loc='upper right', fontsize='small')
    
    fig.tight_layout()
    return fig, axes


def plot_bootstrap_comparison(final_results, boot_results):
    """
    Plot comparison of estimates/errors: Hesse vs Bootstrap (Figure 4.2).
    
    Parameters
    ----------
    final_results : dict
        Original results (values + Hesse errors)
    boot_results : dict
        Bootstrap results (distributions)
        
    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    methods = ['sample_ests', 'individual_fits', 'simultaneous_fit']
    method_labels = ['Sample Ests', 'Indiv Fits', 'Simult Fit']
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    params = ['lb', 'dE', 'a', 'b', 'c'] # Keys in final_results
    boot_params = ['lambda', 'Delta', 'a', 'b', 'c'] # Keys in boot_results
    param_labels = [r'$\lambda$', r'$\Delta$', r'$a$', r'$b$', r'$c$']
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    
    for i, (param, b_param) in enumerate(zip(params, boot_params)):
        ax = axes[i]
        
        for j, method in enumerate(methods):
            # 1. Original Result (Value + Hesse Error)
            val = final_results[method]['values'][param]
            err = final_results[method]['errors'][param]
            
            # Plot slightly offset to left
            ax.errorbar(j - 0.15, val, yerr=err, fmt='o', capsize=5, 
                        color=colors[j], markerfacecolor='white', 
                        label=f'{method_labels[j]} (Hesse)' if i==0 else "")
            
            # 2. Bootstrap Result (Mean + Std of distribution)
            b_vals = boot_results[method][b_param]
            b_mean = np.mean(b_vals)
            b_std = np.std(b_vals)
            
            # Plot slightly offset to right
            ax.errorbar(j + 0.15, b_mean, yerr=b_std, fmt='s', capsize=5, 
                        color=colors[j], alpha=0.7,
                        label=f'{method_labels[j]} (Boot)' if i==0 else "")
            
        ax.set_title(param_labels[i])
        ax.set_xticks(range(3))
        ax.set_xticklabels(method_labels, rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adjust limits
        # ... (logic to set reasonable limits)

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    # Filter duplicate labels if any
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='upper center', 
               bbox_to_anchor=(0.5, 1.1), ncol=3)
    
    plt.tight_layout()
    return fig, axes