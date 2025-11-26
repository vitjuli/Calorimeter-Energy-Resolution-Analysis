"""
Full bootstrap analysis for all three methods.
"""
import numpy as np
from s1_sol import estimators, fitting, mle_fits, simultaneous_fit

def run_full_bootstrap(grouped_data, n_samples=2500):
    """
    Run bootstrap on the entire analysis chain for all 3 methods using resample package.
    
    Parameters
    ----------
    grouped_data : dict
        Original grouped data
    n_samples : int
        Number of bootstrap samples (default 2500)
        
    Returns
    -------
    results : dict
        Dictionary containing lists of parameters for each method
        Structure:
        {
            'sample_ests': {'lambda': [], 'Delta': [], ...},
            'individual_fits': {'lambda': [], ...},
            'simultaneous_fit': {'lambda': [], ...}
        }
    """
    from resample import bootstrap
    
    # Initialize storage
    params = ['lambda', 'Delta', 'a', 'b', 'c']
    methods = ['sample_ests', 'individual_fits', 'simultaneous_fit']
    
    results = {m: {p: [] for p in params} for m in methods}
    
    # Prepare flat arrays for simultaneous fit
    all_E_true = []
    all_E_rec = []
    for E0 in sorted(grouped_data.keys()):
        all_E_true.extend(grouped_data[E0]['E_true'])
        all_E_rec.extend(grouped_data[E0]['E_rec'])
    all_E_true = np.array(all_E_true)
    all_E_rec = np.array(all_E_rec)
    n_total = len(all_E_true)
    
    E0_list = sorted(grouped_data.keys())
    E0_arr = np.array(E0_list)
    
    print(f"Starting full bootstrap with {n_samples} samples...")
    
    # Create generators for each energy group
    group_generators = {E0: bootstrap.resample(grouped_data[E0]['E_rec'], size=n_samples) for E0 in E0_list}
    
    for i in range(n_samples):
        if (i+1) % 100 == 0:
            print(f"  Sample {i+1}/{n_samples}...")
            
        # --- 1. RESAMPLING ---
        # We resample per energy group to maintain N_events per energy
        # This is standard for fixed-energy beam tests
        resampled_groups = {}
        flat_E_true = []
        flat_E_rec = []
        
        for E0 in E0_list:
            # Get next sample from generator
            resample = next(group_generators[E0])
            
            resampled_groups[E0] = {
                'E_true': grouped_data[E0]['E_true'], # True energy is fixed
                'E_rec': resample,
                'n_events': len(resample)
            }
            
            flat_E_true.extend([E0]*len(resample))
            flat_E_rec.extend(resample)
            
        flat_E_true = np.array(flat_E_true)
        flat_E_rec = np.array(flat_E_rec)
        
        try:
            # --- METHOD 1: SAMPLE ESTIMATES ---
            # Calculate stats
            means, mean_errs = [], []
            stds, std_errs = [], []
            
            for E0 in E0_list:
                rec = resampled_groups[E0]['E_rec']
                means.append(np.mean(rec))
                mean_errs.append(np.std(rec, ddof=1)/np.sqrt(len(rec)))
                stds.append(np.std(rec, ddof=1))
                std_errs.append(np.std(rec, ddof=1)/np.sqrt(2*(len(rec)-1)))
                
            # Fit trends
            # Suppress prints by not capturing return values we don't need
            _, mp1, _ = fitting.fit_mean_parameters(E0_arr, means, mean_errs)
            _, rp1, _ = fitting.fit_resolution_parameters(E0_arr, stds, std_errs)
            
            # Store
            results['sample_ests']['lambda'].append(mp1['lambda'])
            results['sample_ests']['Delta'].append(mp1['Delta'])
            results['sample_ests']['a'].append(rp1['a'])
            results['sample_ests']['b'].append(rp1['b'])
            results['sample_ests']['c'].append(rp1['c'])
            
            # --- METHOD 2: INDIVIDUAL MLE FITS ---
            means_mle, mean_errs_mle = [], []
            stds_mle, std_errs_mle = [], []
            
            for E0 in E0_list:
                rec = resampled_groups[E0]['E_rec']
                # Fast MLE fit (using sample stats as init guesses)
                _, p, e = mle_fits.fit_gaussian_for_energy(rec, E0)
                means_mle.append(p['mu'])
                mean_errs_mle.append(e['mu'])
                stds_mle.append(p['sigma'])
                std_errs_mle.append(e['sigma'])
                
            # Fit trends
            _, mp2, _ = fitting.fit_mean_parameters(E0_arr, means_mle, mean_errs_mle)
            _, rp2, _ = fitting.fit_resolution_parameters(E0_arr, stds_mle, std_errs_mle)
            
            # Store
            results['individual_fits']['lambda'].append(mp2['lambda'])
            results['individual_fits']['Delta'].append(mp2['Delta'])
            results['individual_fits']['a'].append(rp2['a'])
            results['individual_fits']['b'].append(rp2['b'])
            results['individual_fits']['c'].append(rp2['c'])
            
            # --- METHOD 3: SIMULTANEOUS FIT ---
            # We use a simplified call to avoid overhead
            nll = simultaneous_fit.SimultaneousNLL(flat_E_true, flat_E_rec)
            # Use results from Method 1 as smart initial guesses!
            m = simultaneous_fit.Minuit(nll, 
                                      lambda_param=mp1['lambda'], 
                                      Delta=mp1['Delta'], 
                                      a=rp1['a'], 
                                      b=rp1['b'], 
                                      c=rp1['c'])
            m.limits['a'] = (0, None)
            m.limits['b'] = (0, None)
            m.limits['c'] = (0, None)
            m.errordef = simultaneous_fit.Minuit.LIKELIHOOD
            m.migrad()
            
            if m.valid:
                results['simultaneous_fit']['lambda'].append(m.values['lambda_param'])
                results['simultaneous_fit']['Delta'].append(m.values['Delta'])
                results['simultaneous_fit']['a'].append(m.values['a'])
                results['simultaneous_fit']['b'].append(m.values['b'])
                results['simultaneous_fit']['c'].append(m.values['c'])
            
        except Exception:
            continue
            
    return results
