"""SIP helper functions imported by SIP_ML_Tool.ipynb.

This file contains the helper/KK section that used to live inline in the
notebook. The notebook remains the main workflow.
"""

# Helper functions used by the main workflow.
# They are used to read SIP data, work with log-frequency values, and calculate KK-based features.

# ------------------------------------------------------------------------------
# Helper functions for KK and log-frequency calculations
# ------------------------------------------------------------------------------

import numpy as np
import os.path
import csv
from datetime import datetime
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
def create_impedance_data(frequencies, impedances, mask=None, path="", loop=None, date_time=None):
    
    # Convert to numpy arrays
    frequencies = np.array(frequencies, dtype = float)
    impedances = np.array(impedances, dtype = complex)
    
    # Basic validation
    if frequencies.shape != impedances.shape or len(frequencies.shape) != 1 or len(frequencies) == 0:
        raise ValueError("Invalid frequency or impedances data")
    
    # Sort in descending order of frequency
    if frequencies[-1] > frequencies[0]:
        frequencies = np.flip(frequencies)
        impedances = np.flip(impedances)
    
    # Initialize mask 
    data_mask = {i: False for i in range(len(frequencies))}
    
    # Apply any provided mask
    if mask:
        for i, value in mask.items():
            if 0<= i < len(frequencies):
                data_mask[i] = bool(value)
                
    label = os.path.splitext(os.path.basename(path))
    if loop is not None:
        label = f"{label}_loop{loop}"

    return {
        "frequencies": frequencies,
        "impedances": impedances,
        "mask": data_mask,
        "path": path,
        "label": label,
        "loop": loop,
        "date_time": date_time
    }
#------------------------------------------------------------------------
def get_masked_data(data, masked=True):

    if masked is None:
        return data["frequencies"], data["impedances"]
    
    indices = [i for i, is_masked in data["mask"].items() if is_masked != masked]
    return data["frequencies"][indices], data["impedances"][indices]
#-------------------------------------------------------------------------
def filter_data(data, low_cutoff=None, high_cutoff=None):
    
    result = dict(data)
    new_mask = dict(data["mask"])
    
    for i, f in enumerate(data["frequencies"]):
        # Apply low-pass filter
        if low_cutoff is not None and f < low_cutoff:
            new_mask[i] = True
            
        # Apply high-pass filter
        if high_cutoff is not None and f > high_cutoff:
            new_mask[i] = True
            
    result["mask"] = new_mask
    return result
#---------------------------------------------------------------------------
def read_SIP_data(path, geometric_factor, separator=","):
    # Geometric_factor: unit cm 

    # Check if file exists
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    # Read file to extract date and time from header
    date_time = None
    with open(path, 'r') as file:
        # Look for date and time in the first 20 lines
        for _ in range(20):
            line = file.readline()
            if not line:
                break
            if "Date" in line or "Time" in line:
                # Extract date/time information
                date_time = line.strip()
    
    # Initialize collections for each loop
    data_by_loop = {}
    # Read data starting from row 29
    with open(path, 'r') as file:
        # Skip first 28 lines
        for _ in range(28):
            next(file, None)
        
        # Read header row (line 29)
        headers = next(file).strip().split(separator)
        
        # Find column indices
        freq_idx = -1
        phase_idx = -1
        imped_idx = -1
        loop_idx = -1
        
        for i, header in enumerate(headers):
            header_lower = header.lower()
            if 'freq' in header_lower:
                freq_idx = i
            elif 'phase' in header_lower:
                phase_idx = i
            elif 'imped' in header_lower or 'z[' in header_lower:
                imped_idx = i
            elif 'loop' in header_lower:
                loop_idx = i
        
        # Read data rows
        reader = csv.reader(file, delimiter=separator)
        for row in reader:
            if not row or len(row) <= max(freq_idx, imped_idx, phase_idx, loop_idx):
                continue
            
            try:
                frequency = float(row[freq_idx])
                impedance_magnitude = float(row[imped_idx])
                phase = float(row[phase_idx])
                loop_id = int(row[loop_idx]) if loop_idx >= 0 and loop_idx < len(row) else 1
                
                # Calculate complex impedance
                impedance = impedance_magnitude * geometric_factor/100 * np.exp(1j * phase)
                
                # Add to the appropriate loop data
                if loop_id not in data_by_loop:
                    data_by_loop[loop_id] = {
                        'frequencies': [],
                        'impedances': []
                    }
                
                data_by_loop[loop_id]['frequencies'].append(frequency)
                data_by_loop[loop_id]['impedances'].append(impedance)

            except (ValueError, IndexError):
                # Skip rows that can't be parsed
                continue
    
    # Create a dataset for each loop
    result = []
    for loop_id in sorted(data_by_loop.keys()):
        loop_data = data_by_loop[loop_id]
        dataset = create_impedance_data(
            frequencies=loop_data['frequencies'],
            impedances=loop_data['impedances'],
            path=path,
            loop=loop_id,
            date_time=date_time
        )
        result.append(dataset)
    
    return result   
#-----------------------------------------------------------------------  
def plot_eis_data(data, title_prefix=""):
    freqs_raw, Z_raw = get_masked_data(data, masked=False)
    freqs_filtered, Z_filtered = get_masked_data(data, masked=True)

    fig1, ax1 = plt.subplots()
    ax1.plot(Z_raw.real, -Z_raw.imag, 'o-', label='Raw', alpha=0.6)
    ax1.plot(Z_filtered.real, -Z_filtered.imag, 'o-', label='Filtered', alpha=0.9)
    ax1.set_xlabel('Z\' (Ohm)')
    ax1.set_ylabel('-Z\'\' (Ohm)')
    ax1.set_title(f'{title_prefix} Nyquist Plot')
    ax1.grid(True)
    ax1.axis('equal')
    ax1.legend()

    fig2, (ax2, ax3) = plt.subplots(2, 1, sharex=True)
    ax2.semilogx(freqs_raw, np.abs(Z_raw), 'o-', label='Raw', alpha=0.6)
    ax2.semilogx(freqs_filtered, np.abs(Z_filtered), 'o-', label='Filtered', alpha=0.9)
    ax2.set_ylabel('|Z| (Ohm)')
    ax2.set_title(f'{title_prefix} Bode Magnitude')
    ax2.grid(True)
    ax2.legend()

    ax3.semilogx(freqs_raw, np.angle(Z_raw, deg=True), 'o-', label='Raw', alpha=0.6)
    ax3.semilogx(freqs_filtered, np.angle(Z_filtered, deg=True), 'o-', label='Filtered', alpha=0.9)
    ax3.set_ylabel('Phase (deg)')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_title(f'{title_prefix} Bode Phase')
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.show()
 #-----------------------------------------------------------------------
def plot_all_loops(datasets, show=True, save_path=None):
    """
    Plot Nyquist and Bode plots for all loops in the dataset.
    
    Parameters:
        datasets (list): List of impedance data dicts (from create_impedance_data).
        show (bool): Whether to show the plots immediately.
        save_path (str or None): Folder to save plots. If None, plots are not saved.
    """
    for ds in datasets:
        title_prefix = f"Loop {ds['loop']}"
        plot_eis_data(ds, title_prefix=title_prefix)

        if save_path:
            fig_name = f"{title_prefix.replace(' ', '_')}.png"
            plt.savefig(os.path.join(save_path, fig_name), dpi=300)
        
        if show:
            plt.show()
        else:
            plt.close()
#----------------------------------------------------------------------------
def process_eis_file(path, geometric_factor, apply_filter=False, low_cutoff=None, high_cutoff=None):
    """
    Main function to load, optionally filter, and return processed EIS datasets.
    
    Parameters:
        path (str): Path to the CSV data file.
        geometric_factor (float): Geometric factor in cm.
        apply_filter (bool): Whether to filter data.
        low_cutoff (float or None): Low frequency cutoff (Hz).
        high_cutoff (float or None): High frequency cutoff (Hz).
        
    Returns:
        List of processed impedance data dictionaries (one per loop).
    """
    datasets = read_SIP_data(path=path, geometric_factor=geometric_factor)
    
    if apply_filter:
        datasets = [filter_data(ds, low_cutoff, high_cutoff) for ds in datasets]

    return datasets
#------------------------------------------------------------------------------
def analyze_eis_data(
    path: str,
    geometric_factor: float,
    apply_filter: bool = False,
    low_cutoff: float = None,
    high_cutoff: float = None,
    do_plot: bool = True,
    mode: str = "raw"  # "raw", "filtered", or "both"
) -> str:
    datasets = process_eis_file(
        path=path,
        geometric_factor=geometric_factor,
        apply_filter=(mode != "raw"),
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff
    )

    if do_plot:
        plot_all_loops(datasets)

    summary = []
    for ds in datasets:
        summary.append(
            f"Loop {ds['loop']}: {len(ds['frequencies'])} points | Date: {ds['date_time']}"
        )
    return "\n".join(summary)

from numpy import zeros, pi, logspace, log10
from numpy.linalg import lstsq
#------------------------------------------------------------------------------
# Weighting based on Boukamp (1995)
def _boukamp_weight(Z, admittance):
    if admittance:
        Y = 1 / Z
        return (Y.real**2 + Y.imag**2) ** -1
    return (Z.real**2 + Z.imag**2) ** -1
#------------------------------------------------------------------------------
# Generate time constants based on frequency range and log extension
def _generate_time_constants(w, num_RC, log_F_ext):
    min_tau = 1 / w.max()
    max_tau = 1 / w.min()
    factor = 10 ** log_F_ext
    return logspace(log10(min_tau / factor), log10(max_tau * factor), num_RC)
#------------------------------------------------------------------------------
# A-matrix construction helpers
def _initialize_A_matrix(test, w, taus, add_cap, add_ind):
    m = len(w) * (2 if test == "complex" else 1)
    n = len(taus) + 1 + int(add_cap) + int(add_ind)
    return zeros((m, n))
#------------------------------------------------------------------------------
def _add_resistance_to_A_matrix(A, test):
    m = A.shape[0]
    if test == "complex":
        A[:m//2, 0] = 1
#        A[m//2:, 0] = 1
    else:
        A[:m, 0] = 1
#------------------------------------------------------------------------------
def _calculate_kth_term(w, tau, admittance):
    return (w/(w*tau - 1j)) if admittance else (1/(1 + 1j*w*tau))
#------------------------------------------------------------------------------
def _add_kth_to_A(A, test, w, tau, idx, admittance):
    c = _calculate_kth_term(w, tau, admittance)
    m = A.shape[0]
    if test == "complex":
        A[:m//2, idx] = c.real
        A[m//2:, idx] = c.imag
    elif test == "real":
        A[:m, idx] = c.real
    else:
        A[:m, idx] = c.imag
#------------------------------------------------------------------------------
def _add_cap_to_A(A, test, w, idx, admittance):
    m = A.shape[0]
    if test == "complex":
        A[m//2:, idx] = w if admittance else -1/w
    elif test == "imaginary":
        A[:, idx] = w if admittance else -1/w
    else:
        A[:m, idx] = -1/w
#------------------------------------------------------------------------------
def _add_ind_to_A(A, test, w, idx, admittance):
    m = A.shape[0]
    if test == "complex":
        A[m//2:, idx] = 1/w if admittance else w
    elif test == "imaginary":
        A[:, idx] = 1/w if admittance else w
    else:
        A[:m, idx] = w
#------------------------------------------------------------------------------
def _generate_A_matrix(test, w, taus, add_cap, add_ind, admittance):
    A = _initialize_A_matrix(test, w, taus, add_cap, add_ind)
    _add_resistance_to_A_matrix(A, test)
    for i, tau in enumerate(taus, start=1):
        _add_kth_to_A(A, test, w, tau, i, admittance)
    idx = len(taus) + 1
    if add_cap:
        _add_cap_to_A(A, test, w, idx, admittance)
        idx += 1
    if add_ind:
        _add_ind_to_A(A, test, w, idx, admittance)
    return A
#------------------------------------------------------------------------------
# b-vector construction helpers
def _initialize_b_vector(test, Z_exp):
    m = len(Z_exp) * (2 if test == "complex" else 1)
    return zeros(m)
#------------------------------------------------------------------------------
def _add_to_b_vector(b, test, Z_exp, admittance):
    m = b.shape[0]
    val = Z_exp**(-1 if admittance else 1)
    if test == "complex":
        b[:m//2] = val.real
        b[m//2:] = val.imag
    elif test == "real":
        b[:m] = val.real
    else:
        b[:m] = val.imag
#------------------------------------------------------------------------------
def _generate_b_vector(test, Z_exp, admittance):
    b = _initialize_b_vector(test, Z_exp)
    _add_to_b_vector(b, test, Z_exp, admittance)
    return b
#------------------------------------------------------------------------------
# Domain-specific least-squares routines
def _real_test(Z_exp, f, w, taus, add_cap=False, add_ind=False, admittance=False):
    A = _generate_A_matrix("real", w, taus, False, False, admittance)
#
#    print("A.shape[modified]:", A.shape)
#    print("A[:3,:5]=\n", A[:3,:5])    
#    
    b = _generate_b_vector("real", Z_exp, admittance)
#
#    print("b[modified]:",b[:5])
#    
    x = lstsq(A, b, rcond=None)[0]
#
#    print("x[modified]:",x.round(5))
#    
    if not (add_cap or add_ind):
        return x
    cols = int(add_cap) + int(add_ind)
    A_corr = zeros((len(w), cols))
    j = 0
    if add_cap:
        A_corr[:, j] = (w if admittance else -1/w)
        j += 1
    if add_ind:
        A_corr[:, j] = ((1/w) if admittance else w)
    Z_fit = (1/(1/Z_exp - 1/(A@x))) if admittance else (Z_exp - 1/(A@x))
    b_corr = _generate_b_vector("imaginary", Z_fit, admittance)
    corr = lstsq(A_corr, b_corr, rcond=None)[0]
    return x, corr
#-----------------------------------------------------------------------------
def _imaginary_test(Z_exp, f, w, taus, weight, add_cap=False, add_ind=False, admittance=False):
    # 1) Build imaginary-part design matrix and RHS
    A_i = _generate_A_matrix("imaginary", w, taus, add_cap, add_ind, admittance)
    A_i[:,0] = 0
#
#    print("A_i.shape[modified]:", A_i.shape)
#    print("A[:3,:5]=\n", A_i[:3,:5])    
#    
    b_i = _generate_b_vector("imaginary", Z_exp, admittance)
#
#    print("b[modified]:",b_i[:5])
#

    # 2) Solve unweighted least-squares for imaginary coefficients
    x = lstsq(A_i, b_i, rcond=None)[0]

#
#    print("x[modified]:",x.round(6))
#

    # 3) Boukamp R0 correction using real part of impedance
    # Build real-part matrix (including R0 column)
    A_r = _generate_A_matrix("real", w, taus, add_cap, add_ind, admittance)
    # Compute predicted real component from all terms
    pred_real = A_r @ x
#
#    print("predicted[modified]:",pred_real.round(5))
#    
    # Weight‐averaged error gives correction to R0
    R0 = (weight * (Z_exp.real - pred_real)).sum() / weight.sum()
    x[0] = R0
    
    return x
#------------------------------------------------------------------------------
def _complex_test(Z_exp, f, w, taus, add_cap=False, add_ind=False, admittance=False):
    A = _generate_A_matrix("complex", w, taus, add_cap, add_ind, admittance)
#
#    print("A.shape[modified]:", A.shape)
#    print("A[:3,:5]=\n", A[:3,:5])    
#    
    b = _generate_b_vector("complex", Z_exp, admittance)
#
#    print("b[modified]:",b[:5])
#    
    return lstsq(A, b, rcond=None)[0]
#-----------------------------------------------------------------------------
# Wrapper for each num_RC
def _test_wrapper(test, f, Z_exp, weight, num_RC, add_cap, add_ind, admittance, log_F_ext):
    w = 2 * pi * f
    taus = _generate_time_constants(w, num_RC, log_F_ext)
#
#    print("taus[modified]:",taus.round(5))
#    
    if test == "real":
        res = _real_test(Z_exp, f, w, taus, add_cap, add_ind, admittance)
#
#        print("x[modified]:",res.round(5))
#        
    elif test == "imaginary":
        res = _imaginary_test(Z_exp, f, w, taus, weight, add_cap, add_ind, admittance)
#
#        print("x[modified]",res.round(5))
#        
    else:
        res = _complex_test(Z_exp, f, w, taus, add_cap, add_ind, admittance)
#
#        print("x[modified]:",res.round(5))
#        
    return num_RC, res
#------------------------------------------------------------------------------
# Simplified evaluate_log_F_ext with inline pseudo-χ² (no extra helper)
def evaluate_log_F_ext(data, test="real", num_RCs=None, add_capacitance=False, 
    add_inductance=False, admittance=False, log_F_ext=0.0):        
    """
    Simplified evaluation of log_F_ext, computing pseudo-χ², residuals, and Z_fit for each num_RC.

    Returns:
        List of tuples (num_RC, coeffs, pseudo_chi2, residuals_array, Z_fit)
    """
    import numpy as np
    
    f = data['frequencies']
    Z_exp = data['impedances']
    weight = _boukamp_weight(Z_exp, admittance)
    pts = len(f)
    
    # default num_RCs if not provided
    if num_RCs is None:
        num_RCs = list(range(2, 2 * pts - 4))
    else:
        num_RCs = sorted(num_RCs)
    
    if not all(map(lambda n: 2<= n <= 2*pts-5, num_RCs)):
        raise ValueError(f"Expected all values in num_RCs to be within range")
            
    results = []
    for n in num_RCs:
        w = 2 * pi * f
        taus = _generate_time_constants(w, n, log_F_ext)

        # Fit via wrapper to get coefficients x
        _, res = _test_wrapper(test, f, Z_exp, weight, n, add_capacitance, add_inductance,
            admittance, log_F_ext)
            
        # Handle case where res is a tuple (x, corr) rather than just x 
        if isinstance(res, tuple) and len(res) == 2:
            x_main, x_corr = res 
            x_combined = np.concatenate([x_main, x_corr])
        else:
            x_combined = res
#
#        print("x[modified]:",x.round(5))
#
        # Reconstruct full complex Z_fit using complex-domain A and same x
        A_cmp = _generate_A_matrix("complex", w, taus, add_capacitance, add_inductance, admittance)
#        
#        print("matrix shape:",A_cmp.shape)
#        print("x length:",len(x_combined))
#        print("x type:",type(x_combined))
#        
        pred = A_cmp @ x_combined
        m = len(f)
        real_part = pred[:m]
        imag_part = pred[m:]
        if admittance:
            Y_fit = real_part + 1j*imag_part
            Z_fit = 1/Y_fit
        else:
            Z_fit = real_part + 1j*imag_part
#
#        print("Z_fit[modified]:",Z_fit[:5])
#
        # Compute residuals and pseudo-χ²
        residuals = Z_exp - Z_fit
        diff_re = residuals.real
        diff_im = residuals.imag
        stat = (weight * (diff_re**2 + diff_im**2)).sum()

        results.append((n, x_combined, stat, residuals, Z_fit))

    return results
#------------------------------------------------------------------------------    

import numpy as np
from numpy import log, array, diff, isnan, nan
#-------------------------------------------------------------------------------------
def perform_kramers_kronig_test(data, test="real", add_capacitance=False, add_inductance=False,
    num_RCs = None, admittance=False, mu_criterion=0.85, log_F_ext=-0.5):
    """
    Perform Kramers-Kronig test on electrochemical impedance data.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing 'frequencies' and 'impedances' keys
    test : str, optional
        Test type ('real' or 'complex')
    add_capacitance : bool, optional
        Whether to add capacitance to the model
    add_inductance : bool, optional
        Whether to add inductance to the model
    admittance : bool, optional
        Whether to use admittance representation
    mu_criterion : float, optional
        Criterion for selecting optimal number of RC elements
    log_F_ext : float, optional
        Log extension factor
        
    Returns:
    --------
    tuple
        Best fitting test result
    """
    # evaluate_log_F_ext returns list of tuples (num_RC, coeffs, pseudo_chi2, residuals_array, Z_fit)
    tests = evaluate_log_F_ext(data=data,test=test, num_RCs=num_RCs, add_capacitance=add_capacitance,
        add_inductance=add_inductance, admittance=admittance, log_F_ext=log_F_ext)
    
    # Auto-RC: delegate to suggest_num_RC
    suggestion = suggest_num_RC(tests, mu_criterion=mu_criterion)
    
    # Return just the best test
    best_test = suggestion[0]
    return best_test
#------------------------------------------------------------------------------
def suggest_num_RC(tests, mu_criterion=0.85):
    """
    Suggest the optimal number of RC elements using Method 1.
    
    Parameters:
    -----------
    tests : list of tuples
        Each tuple contains (num_RC, coeffs, pseudo_chi2, residuals_array, Z_fit)
    mu_criterion : float, optional
        Threshold for mu values
    admittance : bool, optional
        Whether to use admittance representation
        
    Returns:
    --------
    tuple
        (best_test, scores, lower_limit, upper_limit)
    """
    # Sort tests by number of RC elements
    tests = sorted(tests, key=lambda t: t[0])  # Sort by num_RC
    
    # Calculate appropriate limits
    lower_limit, upper_limit = suggest_num_RC_limits(tests)
    
    # Filter tests based on limits
    valid_tests = [t for t in tests if lower_limit <= t[0] <= upper_limit]
    
    if not valid_tests:
        raise ValueError(f"No tests found within limits: {lower_limit} to {upper_limit}")
    
    # Calculate mu values for each test
    mu_values = {}
    for test in valid_tests:
        num_RC = test[0]
        coeffs = test[1]
        mu = calculate_mu_from_coeffs(coeffs, num_RC)
        mu_values[num_RC] = mu
    
    # Find the first RC count where mu is below threshold
    scores = {}
    best_rc = None
    
    for rc in sorted(mu_values.keys()):
        mu = mu_values[rc]
        if not isnan(mu) and mu < mu_criterion:
            best_rc = rc
            break
    
    # If no RC count meets criterion, use the highest available
    if best_rc is None:
        best_rc = max(mu_values.keys())
    
    # Set scores (1.0 for best, 0.0 for others)
    for rc in mu_values.keys():
        scores[rc] = 1.0 if rc == best_rc else 0.0
    
    # Find the best test
    best_test = next(t for t in valid_tests if t[0] == best_rc)
    
    return best_test, scores, lower_limit, upper_limit
#-----------------------------------------------------------------------
def compute_sign_change_distances(curvatures):
    sign_changes = np.diff(np.sign(curvatures))
    zero_crossings = np.where(sign_changes != 0)[0]
    if len(zero_crossings) < 2:
        return np.inf  # no oscillation
    return np.mean(np.diff(zero_crossings))
#-------------------------------------------------------------------------
def compute_curvature(Z_fit):
    dZ = np.gradient(Z_fit)
    ddZ = np.gradient(dZ)
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = np.abs(dZ.real * ddZ.imag - dZ.imag * ddZ.real) / (np.abs(dZ)**3 + 1e-12)
    return kappa
#-------------------------------------------------------------------------
def suggest_num_RC_limits(tests, threshold=4.0):
    """
    Suggest lower and upper limit using curvature inflection analysis (Method 5).
    """
    # Lower limit: χ² elbow (simple log-derivative heuristic)
    x = np.array([t[0] for t in tests], dtype=float)
    y = np.log([t[2] for t in tests])
    diffs = diff(y)
    lower_limit = tests[0][0]
    for i in range(len(diffs)):
        if abs(diffs[i]) < 0.1:
            lower_limit = int(x[i + 1])
            break

    # Upper limit: detect onset of noisy/overfitted curvature
    scores = {}
    for t in tests:
        if t[0] < lower_limit:
            continue
        Z_fit = t[4]
        kappa = compute_curvature(Z_fit)
        dist = compute_sign_change_distances(kappa)
        scores[t[0]] = dist

    # Select the last RC before spacing becomes too small (too noisy)
    sorted_rc = sorted(scores.items())
    upper_limit = sorted_rc[-1][0]
    for rc, spacing in sorted_rc:
        if spacing < threshold:
            upper_limit = rc
            break

    # Safety fallback
    if upper_limit <= lower_limit:
        upper_limit = lower_limit + 1
    
    return lower_limit, upper_limit
#-----------------------------------------------------------------------
def calculate_mu_from_coeffs(coeffs, num_RC):
    """
    Calculate mu value based on the ratio of negative to positive element values.
    
    Parameters:
    -----------
    coeffs : array
        Coefficient array from the fit
    num_RC : int
        Number of RC elements
    admittance : bool
        Whether to use admittance representation
        
    Returns:
    --------
    float
        Mu value between 0 and 1, or NaN if no positive elements
    """
    mass_of_positives = 0.0
    mass_of_negatives = 0.0
    
    # Check solution resistance (R0)
    if len(coeffs) > 0:
        value = coeffs[0]
        if value >= 0.0:
            mass_of_positives += value
        else:
            mass_of_negatives += abs(value)
    
    # Check RC element coefficients
    for i in range(1, min(1 + num_RC, len(coeffs))):
        value = coeffs[i]
        if value >= 0.0:
            mass_of_positives += value
        else:
            mass_of_negatives += abs(value)
    
    # Calculate mu according to the original formula
    if mass_of_positives == 0.0:
        return nan
    
    mu = 1.0 - mass_of_negatives / mass_of_positives
    return min(1.0, max(0.0, mu))
#-------------------------------------------------------------------------    


#----------------------------------------------------------------------------------------
def plot_pseudo_chisqr(tests, lower_limit=-1, upper_limit=-1, colors=None, markers=None, figure=None,
    legend=True, axes=None, adjust_axes=True, colored_axes=False):
    """
    Plot the pseudo chi-squared values from Kramers-Kronig test results.
    
    Parameters
    ----------
    tests: list of tuples
        Test results from evaluate_log_F_ext(): (num_RC, coeffs, pseudo_chi2, residuals, Z_fit)
    lower_limit, upper_limit: int
        Limits for valid number of RC elements (-1 means no limit)
    colors, markers: dict
        Visual styling options
    figure: matplotlib Figure
        Figure to plot on (creates new one if None)
    axes: list
        List of axes to use (uses figure's axes if None)
    legend, adjust_axes, colored_axes: bool
        Plotting options
        
    Returns
    -------
    tuple: (figure, axes)
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    
    # Setup figure and axes
    if figure is None:
        figure, ax = plt.subplots(1, 1, figsize=(8, 6))
        axes = [ax]
    else:
        if axes is None:
            axes = figure.get_axes()
    
    axis = axes[0]
    
    # Set default styles
    color_chisqr = (colors or {}).get("chisqr", 'black')
    marker_chisqr = (markers or {}).get("chisqr", 'o')
    
    # Skip plotting if no tests
    if not tests:
        return figure, axes
    
    # Extract and sort data
    sorted_tests = sorted(tests, key=lambda t: t[0])
    x = [t[0] for t in sorted_tests]  # num_RC values
    y = [t[2] for t in sorted_tests]  # pseudo_chi2 values
    
    # Create lists for included and excluded points
    included_x, included_y = [], []
    excluded_x, excluded_y = [], []
    
    # Determine which points are included or excluded
    for i, rc in enumerate(x):
        if (lower_limit > 0 and rc < lower_limit) or (upper_limit > 0 and rc > upper_limit):
            excluded_x.append(rc)
            excluded_y.append(y[i])
        else:
            included_x.append(rc)
            included_y.append(y[i])
    
    # Plot excluded points (if any)
    if excluded_x:
        axis.scatter(
            excluded_x, 
            excluded_y, 
            edgecolor=color_chisqr, 
            facecolor="none", 
            marker=marker_chisqr,
            label=r"$\chi^2_{\rm ps.}$, excl."
        )
    
    # Plot included points
    if included_x:
        axis.scatter(
            included_x, 
            included_y, 
            color=color_chisqr, 
            marker=marker_chisqr, 
            label=r"$\chi^2_{\rm ps.}$" + (", incl." if excluded_x else "")
        )
    
    # Adjust axes if requested
    if adjust_axes:
        # Configure x-axis
        num_frequencies = len(tests[0][4]) if tests else 1
        min_x = 0
        max_x = min(
            max(x) + 1 if x else 10,
            max(upper_limit + 5 if upper_limit > 0 else 10, 
                min(max(x) if x else 10, num_frequencies))
        )
        axis.set_xlim(min_x, max_x)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_xlabel(r"$N_\tau$")
        
        # Configure y-axis
        axis.set_ylabel(r"$\chi^{2}_{\rm ps.}$")
        axis.set_yscale('log')
        
        # Ensure valid log scale limits
        y_min, y_max = axis.get_ylim()
        if y_min <= 0:
            y_min = min([v for v in y if v > 0] or [1e-6]) * 0.9
            axis.set_ylim(y_min, y_max)
    
    # Add legend if requested
    if legend and (included_x or excluded_x):
        axis.legend()
    
    # Color the y-axis if requested
    if colored_axes:
        axis.spines['left'].set_color(color_chisqr)
        axis.yaxis.label.set_color(color_chisqr)
        axis.tick_params(axis='y', colors=color_chisqr)
    
    return figure, axes
#--------------------------------------------------------------------------
def plot_nyquist(kk_result, experimental_data=None, label=None, exp_label="Experimental", admittance=False,
    color="red", exp_color="blue", marker="o", exp_marker="s", line=False, figure=None, axes=None, legend=True):
    """
    Plot the Nyquist representation of a Kramers-Kronig test result with optional experimental data.
    
    Parameters
    ----------
    kk_result: tuple
        Output from perform_kramers_kronig_test(): (num_RC, coeffs, pseudo_chi2, residuals, Z_fit)
    experimental_data: dict or None, optional
        Dictionary containing experimental data with 'impedances' key
    label: str, optional
        Label for the KK test result in the legend
    exp_label: str, optional
        Label for the experimental data in the legend
    admittance: bool, optional
        Plot admittance (Y) instead of impedance (Z)
    color: str, optional
        Color for the KK test result
    exp_color: str, optional
        Color for the experimental data
    marker: str, optional
        Marker style for KK test result
    exp_marker: str, optional
        Marker style for experimental data
    line: bool, optional
        Use line instead of markers for KK test result
    figure: matplotlib Figure, optional
        Existing figure to plot on
    axes: list, optional
        List containing the axis to plot on
    legend: bool, optional
        Whether to display legend
    
    Returns
    -------
    tuple: (figure, axes)
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import EngFormatter
    
    # Setup figure and axes if not provided
    if figure is None:
        figure, ax = plt.subplots(1, 1, figsize=(8, 8))
        axes = [ax]
    else:
        if axes is None:
            axes = figure.get_axes()
    
    axis = axes[0]
    
    # Use default label if none provided
    if label is None:
        label = f"KK Test (n={kk_result[0]})"
    
    # Extract impedance data from KK test result (Z_fit is the 5th element, index 4)
    kk_impedances = kk_result[4]
    
    # Plot experimental data if provided
    if experimental_data is not None:
        # Extract experimental impedances
        if isinstance(experimental_data, dict) and 'impedances' in experimental_data:
            exp_impedances = experimental_data['impedances']
        else:
            exp_impedances = experimental_data
            
        # Calculate plotting values based on representation
        if admittance:
            exp_values = 1.0 / exp_impedances
            exp_x = exp_values.real
            exp_y = exp_values.imag  # Positive for admittance
        else:
            exp_x = exp_impedances.real
            exp_y = -exp_impedances.imag  # Negative for impedance Nyquist
        
        # Plot experimental data (always as markers)
        if exp_marker in ['o', 's', '^', 'v', '<', '>', 'D', 'p']:
            axis.scatter(exp_x, exp_y, marker=exp_marker, facecolor=exp_color, 
                        edgecolor='black', label=exp_label, s=50)
        else:
            axis.scatter(exp_x, exp_y, marker=exp_marker, color=exp_color, 
                        label=exp_label, s=50)
    
    # Calculate plotting values for KK test result
    if admittance:
        kk_values = 1.0 / kk_impedances
        kk_x = kk_values.real
        kk_y = kk_values.imag  # Positive for admittance
    else:
        kk_x = kk_impedances.real
        kk_y = -kk_impedances.imag  # Negative for impedance Nyquist
    
    # Plot the KK test result
    if line:
        axis.plot(kk_x, kk_y, color=color, label=label, linewidth=2)
    else:
        # Use the correct parameter names for scatter
        if marker in ['o', 's', '^', 'v', '<', '>', 'D', 'p']:
            axis.scatter(kk_x, kk_y, marker=marker, facecolor=color, 
                        edgecolor='black', label=label, s=40)
        else:
            axis.scatter(kk_x, kk_y, marker=marker, color=color, label=label, s=40)
    
    # Format the axes
    axis.set_xlabel(r"${\rm Re}(Y)\ (S)$" if admittance else r"${\rm Re}(Z)\ (\Omega)$")
    axis.set_ylabel(r"${\rm Im}(Y)\ (S)$" if admittance else r"$-{\rm Im}(Z)\ (\Omega)$")
    axis.xaxis.set_major_formatter(EngFormatter(unit=""))
    axis.yaxis.set_major_formatter(EngFormatter(unit=""))
    axis.set_aspect("equal", adjustable="datalim")
    
    # Add legend if requested
    if legend:
        axis.legend()
    
    return figure, axes
#-----------------------------------------------------------------------------------------
def plot_bode(kk_result, experimental_data=None, label=None, exp_label="Experimental", admittance=False,
    line=False, colors=None, markers=None, figures=None, axes=None, legend=True, colored_axes=False):
    """
    Plot Bode diagrams (Magnitude vs Frequency and Phase vs Frequency) in separate figures.
    
    Parameters
    ----------
    kk_result: tuple
        Output from perform_kramers_kronig_test(): (num_RC, coeffs, pseudo_chi2, residuals, Z_fit)
    experimental_data: dict, optional
        Dictionary with 'frequencies' and 'impedances' keys containing experimental data
    label: str, optional
        Label for the KK test result in the legend
    exp_label: str, optional
        Label for the experimental data in the legend
    admittance: bool, optional
        Plot admittance (Y) instead of impedance (Z)
    line: bool, optional
        Use lines instead of markers
    colors: dict, optional
        Colors for 'magnitude' and 'phase' elements
    markers: dict, optional
        Marker styles for 'magnitude' and 'phase' elements
    figures: list, optional
        List of existing figures to plot on [magnitude_fig, phase_fig]
    axes: list, optional
        List of axes to plot on [magnitude_axis, phase_axis]
    legend: bool, optional
        Whether to display legend
    colored_axes: bool, optional
        Whether to color the y-axes to match the data
    
    Returns
    -------
    tuple: (list of figures, list of axes)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Set up default colors and markers
    default_colors = {
        'magnitude': 'blue',
        'phase': 'orange',
        'exp_magnitude': 'darkblue',
        'exp_phase': 'darkorange'
    }
    
    default_markers = {
        'magnitude': 'o',
        'phase': 's',
        'exp_magnitude': '^',
        'exp_phase': 'D'
    }
    
    # Update with user provided values
    if colors:
        default_colors.update(colors)
    if markers:
        default_markers.update(markers)
    
    # Create figures and axes if not provided
    if figures is None or len(figures) < 2:
        fig_mag = plt.figure(figsize=(8, 6))
        fig_phase = plt.figure(figsize=(8, 6))
        figures = [fig_mag, fig_phase]
    else:
        fig_mag, fig_phase = figures[0], figures[1]
    
    if axes is None or len(axes) < 2:
        ax_mag = fig_mag.add_subplot(111)
        ax_phase = fig_phase.add_subplot(111)
        axes = [ax_mag, ax_phase]
    else:
        ax_mag, ax_phase = axes[0], axes[1]
    
    # Use default label if none provided
    if label is None:
        label = f"KK Test (n={kk_result[0]})"
    
    # Determine axis labels based on representation
    mag_suffix = r"$|Y|$ (S)" if admittance else r"$|Z|$ ($\Omega$)"
    phase_suffix = r"$\phi_Y$ (°)" if admittance else r"$\phi_Z$ (°)"
    
    # Extract KK test data
    Z_fit = kk_result[4]  # Impedance values
    
    # For KK test data, we need frequencies
    if experimental_data and 'frequencies' in experimental_data:
        frequencies = experimental_data['frequencies']
    else:
        # If no experimental data, try to extract frequencies from residuals
        frequencies = np.logspace(0, 5, len(Z_fit))  # Default fallback
    
    # Plot experimental data if available
    if experimental_data and 'impedances' in experimental_data and 'frequencies' in experimental_data:
        exp_Z = experimental_data['impedances']
        exp_f = experimental_data['frequencies']
        
        # Calculate magnitude and phase
        if admittance:
            exp_Y = 1.0 / exp_Z
            exp_mag = np.abs(exp_Y)
            exp_phase = np.angle(exp_Y, deg=True)
        else:
            exp_mag = np.abs(exp_Z)
            exp_phase = np.angle(exp_Z, deg=True)
        
        # Plot experimental magnitude
        if line:
            ax_mag.plot(exp_f, exp_mag, color=default_colors['exp_magnitude'], 
                      marker=None, linestyle='-', label=f"{exp_label}")
        else:
            ax_mag.scatter(exp_f, exp_mag, color=default_colors['exp_magnitude'],
                        marker=default_markers['exp_magnitude'], label=f"{exp_label}")
        
        # Plot experimental phase
        if line:
            ax_phase.plot(exp_f, exp_phase, color=default_colors['exp_phase'], 
                        marker=None, linestyle='-', label=f"{exp_label}")
        else:
            ax_phase.scatter(exp_f, exp_phase, color=default_colors['exp_phase'],
                          marker=default_markers['exp_phase'], label=f"{exp_label}")
    
    # Calculate magnitude and phase for KK test result
    if admittance:
        Y_fit = 1.0 / Z_fit
        mag = np.abs(Y_fit)
        phase = np.angle(Y_fit, deg=True)
    else:
        mag = np.abs(Z_fit)
        phase = np.angle(Z_fit, deg=True)
    
    # Plot KK test magnitude
    if line:
        ax_mag.plot(frequencies, mag, color=default_colors['magnitude'], 
                  marker=None, linestyle='--', linewidth=2, label=f"{label}")
    else:
        ax_mag.scatter(frequencies, mag, color=default_colors['magnitude'],
                     marker=default_markers['magnitude'], label=f"{label}")
    
    # Plot KK test phase
    if line:
        ax_phase.plot(frequencies, phase, color=default_colors['phase'], 
                    marker=None, linestyle='--', linewidth=2, label=f"{label}")
    else:
        ax_phase.scatter(frequencies, phase, color=default_colors['phase'],
                       marker=default_markers['phase'], label=f"{label}")
    
    # Set up magnitude axis
    ax_mag.set_xscale('log')
    ax_mag.set_yscale('log')
    ax_mag.set_xlabel('Frequency (Hz)')
    ax_mag.set_ylabel(mag_suffix)
    ax_mag.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Set up phase axis
    ax_phase.set_xscale('log')
    ax_phase.set_xlabel('Frequency (Hz)')
    ax_phase.set_ylabel(phase_suffix)
    ax_phase.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Set y-axis limits for phase
    min_phase = min(phase) if len(phase) > 0 else -90
    max_phase = max(phase) if len(phase) > 0 else 0
    if experimental_data and 'impedances' in experimental_data:
        exp_phase = np.angle(1.0/experimental_data['impedances'] if admittance 
                            else experimental_data['impedances'], deg=True)
        min_phase = min(min_phase, min(exp_phase))
        max_phase = max(max_phase, max(exp_phase))
    
    # Add some padding to phase limits
    phase_padding = (max_phase - min_phase) * 0.1
    ax_phase.set_ylim(min_phase - phase_padding, max_phase + phase_padding)
    
    # Color the axes if requested
    if colored_axes:
        ax_mag.spines['left'].set_color(default_colors['magnitude'])
        ax_mag.yaxis.label.set_color(default_colors['magnitude'])
        ax_mag.tick_params(axis='y', colors=default_colors['magnitude'])
        
        ax_phase.spines['left'].set_color(default_colors['phase'])
        ax_phase.yaxis.label.set_color(default_colors['phase'])
        ax_phase.tick_params(axis='y', colors=default_colors['phase'])
    
    # Add legends if requested
    if legend:
        ax_mag.legend(loc='best')
        ax_phase.legend(loc='best')
    
    # Adjust layout for better appearance
    fig_mag.tight_layout()
    fig_phase.tight_layout()
    
    return figures, axes
#-------------------------------------------------------------------------------------
def plot_real_imaginary(kk_result,experimental_data=None,label=None,exp_label="Experimental",
    admittance=False,line=False,colors=None,markers=None,figures=None,axes=None,legend=True,
    colored_axes=False):
    """
    Plot real and imaginary parts of impedance vs frequency in separate figures.
    
    Parameters
    ----------
    kk_result: tuple
        Output from perform_kramers_kronig_test(): (num_RC, coeffs, pseudo_chi2, residuals, Z_fit)
    experimental_data: dict, optional
        Dictionary with 'frequencies' and 'impedances' keys containing experimental data
    label: str, optional
        Label for the KK test result in the legend
    exp_label: str, optional
        Label for the experimental data in the legend
    admittance: bool, optional
        Plot admittance (Y) instead of impedance (Z)
    line: bool, optional
        Use lines instead of markers
    colors: dict, optional
        Colors for 'real' and 'imaginary' elements
    markers: dict, optional
        Marker styles for 'real' and 'imaginary' elements
    figures: list, optional
        List of existing figures to plot on [real_fig, imaginary_fig]
    axes: list, optional
        List of axes to plot on [real_axis, imaginary_axis]
    legend: bool, optional
        Whether to display legend
    colored_axes: bool, optional
        Whether to color the y-axes to match the data
    
    Returns
    -------
    tuple: (list of figures, list of axes)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Set up default colors and markers
    default_colors = {
        'real': 'teal',
        'imaginary': 'magenta',
        'exp_real': 'darkcyan',
        'exp_imaginary': 'purple'
    }
    
    default_markers = {
        'real': 'o',
        'imaginary': 's',
        'exp_real': 'D',
        'exp_imaginary': '^'
    }
    
    # Update with user provided values
    if colors:
        default_colors.update(colors)
    if markers:
        default_markers.update(markers)
    
    # Create figures and axes if not provided
    if figures is None or len(figures) < 2:
        fig_real = plt.figure(figsize=(8, 6))
        fig_imag = plt.figure(figsize=(8, 6))
        figures = [fig_real, fig_imag]
    else:
        fig_real, fig_imag = figures[0], figures[1]
    
    if axes is None or len(axes) < 2:
        ax_real = fig_real.add_subplot(111)
        ax_imag = fig_imag.add_subplot(111)
        axes = [ax_real, ax_imag]
    else:
        ax_real, ax_imag = axes[0], axes[1]
    
    # Use default label if none provided
    if label is None:
        label = f"KK Test (n={kk_result[0]})"
    
    # Determine axis labels based on representation
    real_suffix = r"Re($Y$) (S)" if admittance else r"Re($Z$) ($\Omega$)"
    imag_suffix = r"Im($Y$) (S)" if admittance else r"-Im($Z$) ($\Omega$)"
    
    # Extract KK test data
    Z_fit = kk_result[4]  # Impedance values
    
    # For KK test data, we need frequencies
    if experimental_data and 'frequencies' in experimental_data:
        frequencies = experimental_data['frequencies']
    else:
        # If no experimental data, assume a frequency range
        frequencies = np.logspace(0, 5, len(Z_fit))
    
    # Plot experimental data if available
    if experimental_data and 'impedances' in experimental_data and 'frequencies' in experimental_data:
        exp_Z = experimental_data['impedances']
        exp_f = experimental_data['frequencies']
        
        # Calculate real and imaginary parts
        if admittance:
            exp_Y = 1.0 / exp_Z
            exp_real = exp_Y.real
            exp_imag = exp_Y.imag  # Positive for admittance
        else:
            exp_real = exp_Z.real
            exp_imag = -exp_Z.imag  # Negative for impedance
        
        # Plot experimental real part
        if line:
            ax_real.plot(exp_f, exp_real, color=default_colors['exp_real'], 
                        marker=None, linestyle='-', label=f"{exp_label}")
        else:
            ax_real.scatter(exp_f, exp_real, color=default_colors['exp_real'],
                          marker=default_markers['exp_real'], label=f"{exp_label}")
        
        # Plot experimental imaginary part
        if line:
            ax_imag.plot(exp_f, exp_imag, color=default_colors['exp_imaginary'], 
                        marker=None, linestyle='-', label=f"{exp_label}")
        else:
            ax_imag.scatter(exp_f, exp_imag, color=default_colors['exp_imaginary'],
                          marker=default_markers['exp_imaginary'], label=f"{exp_label}")
    
    # Calculate real and imaginary parts for KK test result
    if admittance:
        Y_fit = 1.0 / Z_fit
        real_part = Y_fit.real
        imag_part = Y_fit.imag  # Positive for admittance
    else:
        real_part = Z_fit.real
        imag_part = -Z_fit.imag  # Negative for impedance
    
    # Plot KK test real part
    if line:
        ax_real.plot(frequencies, real_part, color=default_colors['real'], 
                    marker=None, linestyle='--', linewidth=2, label=f"{label}")
    else:
        ax_real.scatter(frequencies, real_part, color=default_colors['real'],
                      marker=default_markers['real'], label=f"{label}")
    
    # Plot KK test imaginary part
    if line:
        ax_imag.plot(frequencies, imag_part, color=default_colors['imaginary'], 
                    marker=None, linestyle='--', linewidth=2, label=f"{label}")
    else:
        ax_imag.scatter(frequencies, imag_part, color=default_colors['imaginary'],
                      marker=default_markers['imaginary'], label=f"{label}")
    
    # Set up real part axis
    ax_real.set_xscale('log')
    ax_real.set_xlabel('Frequency (Hz)')
    ax_real.set_ylabel(real_suffix)
    ax_real.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Set up imaginary part axis
    ax_imag.set_xscale('log')
    ax_imag.set_xlabel('Frequency (Hz)')
    ax_imag.set_ylabel(imag_suffix)
    ax_imag.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Check if any values are negative for log scale
    if np.any(real_part <= 0) or (experimental_data and 'impedances' in experimental_data and 
             np.any((1.0/experimental_data['impedances']).real <= 0 if admittance else 
                   experimental_data['impedances'].real <= 0)):
        # Linear scale for real axis if negative values exist
        ax_real.set_yscale('linear')
    else:
        # Log scale is safe to use
        ax_real.set_yscale('log')
    
    # Check if any imaginary values are negative for log scale
    # We don't use log scale for imaginary parts as they often cross zero
    ax_imag.set_yscale('linear')
    
    # Color the axes if requested
    if colored_axes:
        ax_real.spines['left'].set_color(default_colors['real'])
        ax_real.yaxis.label.set_color(default_colors['real'])
        ax_real.tick_params(axis='y', colors=default_colors['real'])
        
        ax_imag.spines['left'].set_color(default_colors['imaginary'])
        ax_imag.yaxis.label.set_color(default_colors['imaginary'])
        ax_imag.tick_params(axis='y', colors=default_colors['imaginary'])
    
    # Add legends if requested
    if legend:
        ax_real.legend(loc='best')
        ax_imag.legend(loc='best')
    
    # Adjust layout for better appearance
    fig_real.tight_layout()
    fig_imag.tight_layout()
    
    return figures, axes
#-------------------------------------------------------------    


# ------------------------------------------------------------------------------
# Thin wrapper used by SIP_ml_2.24_V2 pipeline
def compute_kk_features(freq, imp_mag, phase_rad, max_freq=3000.0, test="real",
                        mu_criterion=0.85, log_F_ext=-0.5, num_RCs_max=30,
                        add_capacitance=False, add_inductance=False, admittance=False):
    """Return KK/log-F diagnostics as lightweight numeric features.

    Returns dict with keys: num_rc, pseudo_chi2, mu, resid_rms, resid_max.
    Safe: returns NaNs on failure.
    """
    try:
        f = np.asarray(freq, dtype=float)
        mag = np.asarray(imp_mag, dtype=float)
        ph = np.asarray(phase_rad, dtype=float)

        # Basic cleanup
        ok = np.isfinite(f) & np.isfinite(mag) & np.isfinite(ph) & (f > 0) & (mag > 0)
        f = f[ok]; mag = mag[ok]; ph = ph[ok]
        if f.size < 8:
            return {"num_rc": np.nan, "pseudo_chi2": np.nan, "mu": np.nan, "resid_rms": np.nan, "resid_max": np.nan}

        # Cap freq range (match standalone)
        if max_freq is not None:
            keep = f <= float(max_freq)
            f = f[keep]; mag = mag[keep]; ph = ph[keep]
            if f.size < 8:
                return {"num_rc": np.nan, "pseudo_chi2": np.nan, "mu": np.nan, "resid_rms": np.nan, "resid_max": np.nan}

        # Ensure descending frequency (matches create_impedance_data behavior)
        if f[-1] > f[0]:
            f = f[::-1]
            mag = mag[::-1]
            ph = ph[::-1]

        Z = mag * np.exp(1j * ph)

        data = {"frequencies": f, "impedances": Z}

        # Keep RC sweep bounded for runtime
        pts = int(f.size)
        max_allowed = max(2, 2 * pts - 5)
        if num_RCs_max is None:
            num_RCs = None
        else:
            upper = int(min(num_RCs_max, max_allowed))
            num_RCs = list(range(2, upper + 1))

        best_test = perform_kramers_kronig_test(
            data=data,
            test=test,
            add_capacitance=add_capacitance,
            add_inductance=add_inductance,
            num_RCs=num_RCs,
            admittance=admittance,
            mu_criterion=mu_criterion,
            log_F_ext=log_F_ext
        )

        if best_test is None:
            return {"num_rc": np.nan, "pseudo_chi2": np.nan, "mu": np.nan, "resid_rms": np.nan, "resid_max": np.nan}

        num_rc, coeffs, pseudo_chi2, residuals, Z_fit = best_test

        # Residuals are complex or stacked real/imag depending on test wrapper.
        res = np.asarray(residuals)
        if np.iscomplexobj(res):
            abs_res = np.abs(res)
        else:
            abs_res = np.abs(res.astype(float))

        resid_rms = float(np.sqrt(np.mean(abs_res**2))) if abs_res.size else np.nan
        resid_max = float(np.max(abs_res)) if abs_res.size else np.nan

        mu = float(calculate_mu_from_coeffs(coeffs, num_rc)) if coeffs is not None else np.nan

        return {
            "num_rc": float(num_rc),
            "pseudo_chi2": float(pseudo_chi2),
            "mu": mu,
            "resid_rms": resid_rms,
            "resid_max": resid_max,
        }
    except Exception:
        return {"num_rc": np.nan, "pseudo_chi2": np.nan, "mu": np.nan, "resid_rms": np.nan, "resid_max": np.nan}


__all__ = [name for name in globals() if not name.startswith("__")]
