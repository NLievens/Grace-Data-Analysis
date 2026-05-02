
# External Imports
import os
import tqdm
import sys
import openpyxl
import datetime
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from scipy.interpolate import interp1d

# Internal Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data.Data_Reader import *
from Filters import *

# Define Delta Harmonic Coefficients
def compute_delta_harmonics(data_year_arr, date_year_arr, year_lst, max_order=96):
    '''
    Computes the delta spherical harmonic coefficients for C and S, applies necessary corrections (geocenter, SLR, GIA), and prepares the data for LSQR regression.

    Args:
        data_year_arr (np.ndarray): 5D array containing the original spherical harmonic coefficients and their standard deviations, indexed by year and month.
        date_year_arr (np.ndarray): 2D array containing the corresponding date strings for each year and month.
        year_lst (list): List of years to consider for the analysis.
        max_order (int): Maximum spherical harmonic degree/order to consider (default is 96).

    Returns:
        delta_t_lst (list): List of time deltas in days from the reference time t0.
        mask (np.ndarray): Boolean array indicating which coefficients are non-zero.
        org_tot_size (int): Original total number of coefficients before masking.
        CS_delta_vectors_arr (np.ndarray): Array of shape (num_months, num_coeffs) containing the delta coefficients for C and S.
        CS_std_delta_vectors_arr (np.ndarray): Array of shape (num_months, num_coeffs) containing the standard deviations of the delta coefficients for C and S.
    '''

    # Indexing & Pre-allocation
    year_index_arr = np.array(year_lst) - 2002
    considered_date_1D = date_year_arr[year_index_arr].flatten()    # Shape (288,) -> Years * 12 Months
    considered_data = data_year_arr[year_index_arr]                 # Shape (Years, 12, m_limit, m_limit, 4) -> (24, 12, 97, 97, 4) For max_order=96

    # Calculate t0
    first_valid = next((d for d in considered_date_1D if d.strip().lower() not in ("", "empty", "none")), None)
    if not first_valid:
        raise ValueError("❌ ERROR // No Valid Date Found")

    t0_year = datetime.strptime(first_valid.split("_")[0], "%d%m%Y").year   # 2002
    t0_time = datetime(t0_year, 1, 1)                                       # 2002-01-01 00:00:00

    # Initialize Lists
    data_C = []
    data_S = []
    data_C_std = []
    data_S_std = []
    delta_t_lst = []

    # Pre-Calculate Slice Limits
    m_limit = max_order + 1

    # Pre-Read GEO, SLR, and GIA Libraries
    geo_lib = extract_geocentric_coefficients()
    SLR_lib = extract_slr_coefficients()
    gia_C, gia_S = extract_GIA()

    print("✅ Read And Pre-Processed GEO, SLR, And GIA Data\n")

    # Process Each Month
    for i, month_date in enumerate(tqdm.tqdm(considered_date_1D, desc="Processing Months")):
        if month_date == "empty":
            continue

        # Calculate Delta t In Days
        current_dt = datetime.strptime(month_date.split("_")[0], "%d%m%Y")
        delta_t_lst.append((current_dt - t0_time).days)
        
        # Locate data (year index i // 12, month i % 12) (Only take the rows/cols up to max_order immediately)
        month_data = considered_data[i // 12, i % 12, :m_limit, :m_limit]
        
        # Split channels Of Original 97x97 Arrays
        C_orig, S_orig = month_data[:, :, 0].copy(), month_data[:, :, 1].copy()     # Shape (m_limit, m_limit) -> (97, 97) For max_order=96
        C_std, S_std   = month_data[:, :, 2].copy(), month_data[:, :, 3].copy()     # Shape (m_limit, m_limit) -> (97, 97) For max_order=96

        # Define Current Day Key
        current_key = month_date.split("_")[0][2:]

        # Add Geocenter Terms
        geo_vals = geo_lib.get(current_key)
        if geo_vals:
            C_orig[1, 0] = geo_vals['C10']
            C_orig[1, 1] = geo_vals['C11']
            S_orig[1, 1] = geo_vals['S11']

            C_std[1, 0] = geo_vals['C10_std']
            C_std[1, 1] = geo_vals['C11_std']
            S_std[1, 1] = geo_vals['S11_std']

        # Replace C20 And C30 With SLR Data
        SLR_vals = SLR_lib.get(current_key)
        if SLR_vals:
            C_orig[2, 0] = SLR_vals['C20']
            C_std[2, 0] = SLR_vals['C20_std']
            # Only replace C30 if it's not NaN (TN-14 has NaNs before 2012)
            if not np.isnan(SLR_vals['C30']):
                C_orig[3, 0] = SLR_vals['C30']
                C_std[3, 0] = SLR_vals['C30_std']
        
        # Remove Isostatic Rebound (GIA) (https://www.atmosp.physics.utoronto.ca/~peltier/data.php) (Inverse Sign Convention)
        dt_years = (current_dt - t0_time).days / 365.25

        C_orig += gia_C * dt_years
        S_orig += gia_S * dt_years

        # Filtering
        C_semi, S_semi = correlated_error_filter(C_orig, S_orig, max_order)
        C_fil, S_fil = gaussian_filter(C_semi, S_semi)

        # Append Data To Lists
        data_C.append(C_fil)
        data_S.append(S_fil)
        data_C_std.append(C_std)
        data_S_std.append(S_std)
    
    # Convert Lists To Arrays
    data_C_array = np.array(data_C)           # Shape (Months, 97, 97)
    data_S_array = np.array(data_S)           # Shape (Months, 97, 97)
    
    # Define Delta Values By Subtracting Mean
    C_mean = np.mean(data_C_array, axis=0)      # Shape (97, 97) -> Mean Across Months For Each Coefficient
    S_mean = np.mean(data_S_array, axis=0)      # Shape (97, 97) -> Mean Across Months For Each Coefficient

    delta_C = data_C_array - C_mean
    delta_S = data_S_array - S_mean

    # Create Global Mask
    flat_mean = np.concatenate([C_mean.ravel(), S_mean.ravel()])
    mask = (flat_mean != 0)
    org_tot_size = len(flat_mean)

    # Flatten & Concatenate
    CS_delta_vectors = []
    CS_std_delta_vectors = []
    for i in range(len(delta_C)):
        # Flatten C then S
        flat_CS = np.concatenate([delta_C[i].ravel(), delta_S[i].ravel()])
        flat_CS_std = np.concatenate([data_C_std[i].ravel(), data_S_std[i].ravel()])
        
        # Apply The Mask
        CS_delta_vectors.append(flat_CS[mask])
        CS_std_delta_vectors.append(flat_CS_std[mask])

    # Turn To Arrays
    CS_delta_vectors_arr = np.array(CS_delta_vectors)               # Shape (Months, Non-Zero Coefficients) -> (231, 9408) For max_order=96 (231, 97x97)
    CS_std_delta_vectors_arr = np.array(CS_std_delta_vectors)       # Shape (Months, Non-Zero Coefficients) -> (231, 9408) For max_order=96 (231, 97x97)

    return delta_t_lst, mask, org_tot_size, CS_delta_vectors_arr, CS_std_delta_vectors_arr

# Define Regression Model
def T_row(ti):
    '''
    Defines the regression model row for a given time delta ti. The model includes a constant term, linear and quadratic trends, and multiple periodic terms (annual, semi-annual, seasonal, 2-annual, and 5-annual).

    Args:
        ti (float): Time delta in days from the reference time t0.
        
    Returns:
        np.ndarray: A 1D array containing the regression model terms for the given time delta ti.
    '''

    return np.array([ 
        1,                                              # 0 // Constant
        ti,                                             # 1 // Linear trend
        ti**2,                                          # 2 // Quadratic trend
        np.cos(2 * np.pi * ti / 365.2500),              # 3 // Annual (Cosine)
        np.sin(2 * np.pi * ti / 365.2500),              # 4 // Annual (Sine)
        np.cos(2 * np.pi * ti / 182.6250),              # 5 // Semi-Annual (Cosine)
        np.sin(2 * np.pi * ti / 182.6250),              # 6 // Semi-Annual (Sine)
        np.cos(2 * np.pi * ti / 91.3125),               # 7 // Seasonal (Cosine)
        np.sin(2 * np.pi * ti / 91.3125),               # 8 // Seasonal (Sine)
        np.cos(2 * np.pi * ti / (2 * 365.2500)),        # 9 // 2-Annual (Cosine)
        np.sin(2 * np.pi * ti / (2 * 365.2500)),        # 10 // 2-Annual (Sine)
        np.cos(2 * np.pi * ti / (5 * 365.2500)),        # 11 // 5-Annual (Cosine)
        np.sin(2 * np.pi * ti / (5 * 365.2500)),        # 12 // 5-Annual (Sine)
        np.cos(2 * np.pi * ti / (10 * 365.2500)),       # 13 // 10-Annual (Cosine)
        np.sin(2 * np.pi * ti / (10 * 365.2500))        # 14 // 10-Annual (Sine)
    ])

# Define Love Numbers
def get_love_numbers(max_degree=96):
    # Define Reference Values (https://doi.org/10.1029/98jb02844) (Table 1)
    l_values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 40, 50, 70, 100, 150, 200])
    k_l_values = np.array([0.000, 0.027, -0.303, -0.194, -0.132, -0.104, -0.089, -0.081,
                           -0.076, -0.072, -0.069, -0.064, -0.058, -0.051, -0.040, -0.033,
                           -0.027, -0.020, -0.014, -0.010, -0.007])
    
    # Set up interpolation function
    interp_func = interp1d(l_values, k_l_values, kind='linear', fill_value='extrapolate')
    
    # Generate interpolated Love numbers from l=0 to l_max
    l_full = np.arange(0, max_degree + 1)
    k_l_full = interp_func(l_full)
    
    return l_full, k_l_full

# Define Model Coefficients Through LSQR
def compute_model_coefficients(delta_t_lst, CS_delta_vectors, CS_std_delta_vectors, mask, org_tot_size, calc_uncertainty=False, max_order=96):
    '''
    Computes the model coefficients for each spherical harmonic coefficient using weighted least squares regression.

    Args:
        delta_t_lst (list): List of time deltas in days.
        CS_delta_vectors (np.ndarray): Array of shape (num_months, num_coeffs) containing the delta coefficients for C and S.
        CS_std_delta_vectors (np.ndarray): Array of shape (num_months, num_coeffs) containing the standard deviations of the delta coefficients for C and S.
        mask (np.ndarray): Boolean array indicating which coefficients are non-zero.
        org_tot_size (int): Original total number of coefficients before masking.
        calc_uncertainty (bool): Whether to calculate the covariance matrices for the model coefficients.
        max_order (int): Maximum spherical harmonic degree/order considered.

    Returns:
        coeff_models (np.ndarray): Array of shape (num_coeffs, num_params) containing the model coefficients for each spherical harmonic coefficient.
        SH_arr (np.ndarray): Array of shape (side, side, num_params, 2) containing the reconstructed C and S coefficients for each spherical harmonic degree/order.
        cov_SH_arr (np.ndarray or None): Array of shape (side, side, num_params, num_params, 2) containing the covariance matrices for the model coefficients of each spherical harmonic degree/order, or None if calc_uncertainty is False.
    '''

    # Setup Data
    delta_t_arr = np.array(delta_t_lst)
    Y = CS_delta_vectors.T                  # Shape (num_months, num_coeffs) -> Transpose to (num_coeffs, num_months)
    sigma = CS_std_delta_vectors.T          # Shape (num_months, num_coeffs) -> Transpose to (num_coeffs, num_months)
    weights = 1.0 / (sigma**2)              # Shape (num_coeffs, num_months) -> e.g. (9408, 231) For max_order=96
    
    num_coeffs, num_months = Y.shape
    T = np.vstack([T_row(ti) for ti in delta_t_arr])    # Shape (num_months, num_params) -> e.g. (231, 15)
    num_params = T.shape[1]
    TT = T.T                                            # Shape (num_params, num_months) -> e.g. (15, 231)

    coeff_models = np.zeros((num_coeffs, num_params))   # Shape (num_coeffs, num_params) -> e.g. (9408, 15) For max_order=96
    cov_beta_array = np.zeros((num_coeffs, num_params, num_params)) if calc_uncertainty else None

    # Weighted Least Squares Loop
    for i in range(num_coeffs):
        W_i = weights[i, :]
        XtWX = (TT * W_i) @ T
        try:
            XtWX_inv = np.linalg.inv(XtWX)
            coeff_models[i] = XtWX_inv @ (TT * W_i) @ Y[i]
            if calc_uncertainty:
                cov_beta_array[i] = XtWX_inv
        except np.linalg.LinAlgError:
            coeff_models[i] = np.linalg.lstsq(XtWX, (TT * W_i) @ Y[i], rcond=None)[0]

    # R-Squared
    Y_pred = (T @ coeff_models.T).T
    weighted_ssr = np.sum(weights * (Y - Y_pred)**2)
    Y_mean = np.mean(Y, axis=1, keepdims=True)
    weighted_sst = np.sum(weights * (Y - Y_mean)**2)
    r_squared = 1 - (weighted_ssr / weighted_sst)

    # Print Progress Statement
    print("✅ Model Coefficients Computed (R-Squared: {:.4f})".format(r_squared))

    # Reconstruct 4D SH_arr
    full_reconstructed = np.zeros((org_tot_size, num_params))
    full_reconstructed[mask] = coeff_models
    
    side = max_order + 1
    split = side * side
    C_rec = full_reconstructed[:split].reshape(side, side, num_params)
    S_rec = full_reconstructed[split:].reshape(side, side, num_params)
    SH_arr = np.stack([C_rec, S_rec], axis=-1)

    # Reconstruct 5D Covariance
    cov_SH_arr = None
    if calc_uncertainty:
        cov_SH_arr = np.zeros((side, side, num_params, num_params, 2))
        full_cov = np.zeros((org_tot_size, num_params, num_params))
        full_cov[mask] = cov_beta_array
        cov_SH_arr[:,:,:,:,0] = full_cov[:split].reshape(side, side, num_params, num_params)
        cov_SH_arr[:,:,:,:,1] = full_cov[split:].reshape(side, side, num_params, num_params)

    return coeff_models, SH_arr, cov_SH_arr


# 

'''
def compute_model_coefficients(delta_t_lst, CS_delta_vectors, CS_std_delta_vectors, calc_uncertainty=False, max_order=96):
    # Transform Time List To Array
    delta_t_arr = np.array(delta_t_lst)

    # Define The Y And std Values For Each Coefficient // Stack all column vectors horizontally to get a 2D array of shape (num_coeffs, num_months)
    stacked_CS_array = np.hstack(CS_delta_vectors)
    stacked_CS_std_array = np.hstack(CS_std_delta_vectors)

    grouped_coefficients = [list(row) for row in stacked_CS_array]
    grouped_std_coefficients = [list(row) for row in stacked_CS_std_array]

    # Initialise Model Coefficients List
    coeff_models = []
    coeff_cov_beta = []

    # Initialise Weighted Aggregate
    total_ssr = 0.0
    total_sst = 0.0

    # Calculate Model Coefficients
    for coefficient_index in range(len(grouped_coefficients)):
        # Extract Y for the current coefficient across all months
        Y = np.array([grouped_coefficients[coefficient_index][i] for i in range(len(grouped_coefficients[coefficient_index]))])
        
        # Extract sigma (standard deviations) for the same coefficient
        sigma = np.array([grouped_std_coefficients[coefficient_index][i] for i in range(len(grouped_std_coefficients[coefficient_index]))])

        # Build the matrix T for the current coefficient
        T = np.vstack([T_row(ti) for ti in delta_t_arr])

        # Create The Weighted Least Squares Model
        model = sm.WLS(Y, T, weights=(1/(sigma**2)))
        
        # Fit The Model
        results = model.fit()
        
        # Store The Model Coefficients Of The Current Spherical Harmonic
        coeff_models.append(results.params)

        # Determine Covariance Matrices Per Coefficient Of The Model [(p x p) NumPy array]
        coeff_cov_beta.append(results.cov_params())

        # Define Residuals And Weighted Aggregate
        Y_pred = results.predict()
        residuals = Y - Y_pred

        total_ssr += np.sum(residuals**2)  # Sum of squared residuals
        total_sst += np.sum((Y - np.mean(Y))**2)  # Total sum of squares
        
    # Define Output
    model_coef = np.array(coeff_models)  # (9405, coef)

    cov_beta_array = np.array(coeff_cov_beta)  # (9405, coef, coef)

    # Define Weighted Aggregate
    r_squared = 1 - (total_ssr / total_sst)

    # Print Progress Statement
    print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    print(f"Coefficients Determined {model_coef.shape}")
    print(f"Filtered R² // {round(r_squared, 4)}")
    print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")

    # Reconstruct The Original Input Matrix Shape And Add Back The Zero Entries
    CS_reconstructed = np.zeros((CS_array_original.shape[0], model_coef.shape[1]))
    CS_reconstructed[nonzero_mask] = model_coef

    #split The Reconstructed Matrix Into The C And S Arrays
    C_reconstructed = CS_reconstructed[:9409,:]
    S_reconstructed = CS_reconstructed[9409:,:]

    # Recreate A 3D Array Of The Matrices (C, S, Model Coefficients)
    reconstructed_data_arr = np.stack([C_reconstructed, S_reconstructed], axis=2)

    #Reshape (Unflatten In Same Way) The Array To Get The Needed Input For The Spherical Harmonics Code
    shape = (97, 97, model_coef.shape[1], 2)
    SH_arr = reconstructed_data_arr.reshape(shape)

    # Create empty arrays to hold the full (97, 97, p, p, 2) structures
    cov_SH_arr = np.zeros((97, 97, model_coef.shape[1], model_coef.shape[1], 2))

    # Find the indices where coefficients are non-zero (True in the mask)
    nonzero_indices = np.where(nonzero_mask)[0]

    # Loop over the nonzero coefficients and assign values accordingly
    for idx, flat_idx in enumerate(nonzero_indices):
        if flat_idx < 9409:
            # It's a C coefficient
            i = flat_idx // 97
            j = flat_idx % 97
            cov_SH_arr[i, j, :, :, 0] = cov_beta_array[idx]
        else:
            # It's an S coefficient
            s_idx = flat_idx - 9409
            i = s_idx // 97
            j = s_idx % 97
            cov_SH_arr[i, j, :, :, 1] = cov_beta_array[idx]

    # Print Completion Statement
    print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    print(f"Created // model_coef {np.shape(model_coef)}")
    print(f"Created // SH_arr {np.shape(SH_arr)}")
    print(f"Created // cov_SH_arr {np.shape(cov_SH_arr)}")
    print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")

    return model_coef, SH_arr, cov_SH_arr
'''





# Define Single Plot Render
def render_single(earth_grid_EWH, date, sample_time, lat_range=(-np.pi/2, np.pi/2), lon_range=(-np.pi, np.pi)):
    """
    Renders a gravity acceleration heatmap using Cartopy.
    
    Args:
        earth_grid_acc (np.ndarray): 2D array of values.
        date (str): Date string for the title/filename.
        lat_range/lon_range (tuple): Radians, defining the extent of earth_grid_acc.
    """
    # Convert Input Ranges From Radians To Degrees
    lon_deg = np.degrees(lon_range)
    lat_deg = np.degrees(lat_range)

    # Setup Plot And Projection
    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(16, 8), subplot_kw={'projection': projection})

    # Set The Map Extent, i.e. Crop
    ax.set_extent([lon_deg[0], lon_deg[1], lat_deg[0], lat_deg[1]], crs=ccrs.PlateCarree())

    # Add Background Features 
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='aliceblue')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)

    # Overlay The Data (Extent: [min_lon, max_lon, min_lat, max_lat])
    im = ax.imshow(
        earth_grid_EWH, 
        extent=[lon_deg[0], lon_deg[1], lat_deg[0], lat_deg[1]],
        transform=ccrs.PlateCarree(),
        origin='upper', 
        cmap='coolwarm', 
        alpha=0.7,
        zorder=3 # Ensures data stays above the land/ocean colors
    )

    # Aesthetics
    plt.title(f"LSQR Model At t = {sample_time} Days ({date[:2]}-{date[2:4]}-{date[4:]})")
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3)
    gl.top_labels = False
    gl.right_labels = False

    # Add Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, aspect=30)
    cbar.set_label("Equivalent Water Height [mm]", fontsize=12)

    # Save and Show
    filename = f"LSQR_Analysis/Output/EWH_Plot_{date}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ Map successfully rendered and saved to {filename}")
    plt.show()

    return
