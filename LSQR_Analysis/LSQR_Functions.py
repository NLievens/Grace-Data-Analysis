'''
DOCSTRING
'''

# External Imports
import os
import tqdm
import openpyxl
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import lpmv, factorial

# Internal Imports
from Filters import correlated_error_filter, gaussian_filter


# Define Least Squares Regression Basis Functions
def T_row(ti, year_lst):
    # Standard Included Constant And Linear Terms
    row = [1, ti]
    
    # Periods (in days) and labels for each harmonic term
    harmonics = [
        (91.3125, "Seasonal (~Quarterly)"),
        (121.75, "4-Monthly"),
        (182.625, "Semi-Annual"),
        (365.25, "Annual"),
        (2 * 365.25, "2-Annual"),
        (3 * 365.25, "3-Annual"),
        (4 * 365.25, "4-Annual"),
        (5 * 365.25, "5-Annual"),
        (6 * 365.25, "6-Annual"),
        (7 * 365.25, "7-Annual"),
        (8 * 365.25, "8-Annual"),
        (9 * 365.25, "9-Annual"),
        (10 * 365.25, "10-Annual"),
        (15 * 365.25, "15-Annual"),
        (20 * 365.25, "20-Annual")
    ]
    
    # Minimum cycles required to include the term
    min_cycles = 1.5

    # Define Dataset Range
    window_days = 365.25 * (max(year_lst) - min(year_lst))
    
    # Append Considered Harmonics
    for period, label in harmonics:
        if window_days >= min_cycles * period:
            row.extend([
                np.cos(2 * np.pi * ti / period),
                np.sin(2 * np.pi * ti / period)
            ])
        # else: skip term entirely
    
    return np.array(row)

# Define Love Numbers
def get_love_numbers(max_degree):
    # Table values from your image
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

# Define Least Squares Regression
def LSQR_coefficients(data_year_arr, date_year_arr, year_lst, max_order=96):
    """
    Fits linear regression models to spherical harmonic coefficients over a specified year range.

    Args:
        data_year_arr (np.ndarray): 5D array containing coefficients and their standard deviations, structured as (years, months, degree, order, [C coeff, S coeff, C std, S std]).
            Imported from Data/Data_Reader.
        date_year_arr (np.ndarray): 2D array of date strings corresponding to each month/year.
            Imported from Data/Data_Reader.
        year_lst (list[int]): List of years to include in the regression model.
        max_order (int): Maximum spherical harmonic degree/order to consider.

    Returns:
        model_coef (np.ndarray): 2D array of shape (9405, num_coefficients). Each row contains the
            regression model coefficients for a corresponding spherical harmonic coefficient (unfiltered).
        model_coef_fil (np.ndarray): 2D array of shape (9405, num_coefficients). Model coefficients
            for filtered data.
        SH_arr (np.ndarray): 4D array of shape (97, 97, num_coefficients, 2) containing the
            unfiltered model coefficients formatted for spherical harmonics code.
        SH_arr_fil (np.ndarray): 4D array of shape (97, 97, num_coefficients, 2) containing the
            filtered model coefficients formatted for spherical harmonics code.
        cov_SH_arr (np.ndarray): 5D array of shape (97, 97, num_coefficients, num_coefficients, 2)
            holding covariance matrices for unfiltered model coefficients.
        cov_fil_SH_arr (np.ndarray): 5D array of shape (97, 97, num_coefficients, num_coefficients, 2)
            holding covariance matrices for filtered model coefficients.
    """

    # Define Year Index Array
    year_arr = np.array(year_lst)
    year_index_arr = year_arr - 2002

    # Define Considered Months
    iter_val = len(year_lst) * 12
    considered_date_arr = date_year_arr[year_index_arr]
    considered_date_1D_array = considered_date_arr.flatten(order='C')
    considered_data_arr = data_year_arr[year_index_arr]

    # Initialise Data Lists
    data_CS_vector_lst = []
    data_CS_vector_lst_fil = []
    data_CS_std_vector_lst = []
    delta_t_lst = []

    # Initialise delta_val For Delta Coefficients
    delta_val = 0

    # Define t0 As 1st Of January Of The First Considered Year
    for month_date in considered_date_1D_array:
        if month_date.strip().lower() not in ("", "empty", "none"):
            first_valid_date_str = month_date.split("_")[0]
            t0_date = datetime.strptime(first_valid_date_str, "%d%m%Y")
            t0_time = datetime(t0_date.year, 1, 1)
            break
    else:
        raise ValueError("\n❌ ERROR // No Valid Date Found In date_list\n")

    # Define Data
    for month_index in tqdm.tqdm(range(iter_val), desc="Processing Months In Dataset"):
        # Define year_index And month_date
        year_index = month_index // 12
        month_within_year = month_index % 12
        month_date = considered_date_1D_array[month_index]

        if month_date != "empty":
            # Define Data
            month_data = considered_data_arr[year_index, month_within_year]
            C_array_original = month_data[:, :, 0]
            S_array_original = month_data[:, :, 1]
            C_std = month_data[:, :, 2]
            S_std = month_data[:, :, 3]

            # Filter Data
            C_array_original_semi_fil, S_array_original_semi_fil = correlated_error_filter(C_array_original, S_array_original, max_order)
            C_array_original_fil, S_array_original_fil = gaussian_filter(C_array_original_semi_fil, S_array_original_semi_fil)

            # Cut Data To Defined Order To Improve Runtime
            C_array = C_array_original[:(max_order + 1)]
            C_array_fil = C_array_original_fil[:(max_order + 1)]

            S_array = S_array_original[:(max_order + 1)]
            S_array_fil = S_array_original_fil[:(max_order + 1)]

            C_std = C_std[:(max_order + 1)]
            S_std = S_std[:(max_order + 1)]

            # Flatten 2D Matrices
            C_array = C_array.flatten(order='C')
            C_array_fil = C_array_fil.flatten(order='C')

            S_array = S_array.flatten(order='C')
            S_array_fil = S_array_fil.flatten(order='C')

            C_std = C_std.flatten(order='C')
            S_std = S_std.flatten(order='C')

            # Join Arrays Accordingly
            CS_array_original = np.concatenate([C_array, S_array])
            CS_array_original_fil = np.concatenate([C_array_fil, S_array_fil])
            CS_std_array = np.concatenate([C_std, S_std])

            # Remove Zero Entries
            nonzero_mask = (CS_array_original != 0)
            CS_array = CS_array_original[nonzero_mask]
            CS_array_fil = CS_array_original_fil[nonzero_mask]
            CS_std_array = CS_std_array[nonzero_mask]          

            # Reshape To Vector And Store
            data_CS_vector_lst.append(CS_array.reshape(-1, 1))
            data_CS_vector_lst_fil.append(CS_array_fil.reshape(-1, 1))
            data_CS_std_vector_lst.append(CS_std_array.reshape(-1, 1))

            # Extract day-of-year for this month
            start_str = month_date.split("_")[0]
            current_date = datetime.strptime(start_str, "%d%m%Y")
            days_since_t0 = (current_date - t0_time).days
            delta_t_lst.append(days_since_t0)

            # Update delta_val For Delta Coefficients
            delta_val += 1

    # Define Delta Values
    delta_CS_vector_fil = []
    delta_CS_std_vector_fil = []

    for i in range(delta_val):
        delta_CS_vector_fil.append(data_CS_vector_lst_fil[i] - data_CS_vector_lst_fil[0])
        delta_CS_std_vector_fil.append(np.sqrt((data_CS_std_vector_lst[i] ** 2) + (data_CS_std_vector_lst[0] ** 2)))

    # Transform To Array
    delta_t_arr = np.array(delta_t_lst)

    # Define The Y And std Values For Each Coefficient // Stack all column vectors horizontally to get a 2D array of shape (num_coeffs, num_months)
    stacked_CS_array = np.hstack(data_CS_vector_lst)
    stacked_CS_array_fil = np.hstack(delta_CS_vector_fil)
    stacked_CS_std_array = np.hstack(delta_CS_std_vector_fil)

    grouped_coefficients = [list(row) for row in stacked_CS_array]
    grouped_coefficients_fil = [list(row) for row in stacked_CS_array_fil]
    grouped_std_coefficients = [list(row) for row in stacked_CS_std_array]

    # Initialise Model Coefficients List
    coeff_models = []
    coeff_models_fil = []
    coeff_cov_beta = []
    coeff_cov_beta_fil = []

    # Initialise Weighted Aggregate
    total_ssr = 0.0
    total_sst = 0.0
    total_ssr_fil = 0.0
    total_sst_fil = 0.0

    # Layout
    print()

    # Calculate Model Coefficients
    for coefficient_index in tqdm.tqdm(range(len(grouped_coefficients)), desc="Calculating Model Coefficients"):
        # Extract Y for the current coefficient across all months
        Y = np.array([grouped_coefficients[coefficient_index][i] for i in range(len(grouped_coefficients[coefficient_index]))])
        Y_fil = np.array([grouped_coefficients_fil[coefficient_index][i] for i in range(len(grouped_coefficients_fil[coefficient_index]))])
        
        # Extract sigma (standard deviations) for the same coefficient
        sigma = np.array([grouped_std_coefficients[coefficient_index][i] for i in range(len(grouped_std_coefficients[coefficient_index]))])

        # Build the matrix T for the current coefficient
        T = np.vstack([T_row(ti, year_lst) for ti in delta_t_arr])

        # Create The Weighted Least Squares Model
        model = sm.WLS(Y, T, weights=1 / sigma**2)
        model_fil = sm.WLS(Y_fil, T, weights=1 / sigma**2)
        
        # Fit The Model
        results = model.fit()
        results_fil = model_fil.fit()
        
        # Store The Model Coefficients Of The Current Spherical Harmonic
        coeff_models.append(results.params)
        coeff_models_fil.append(results_fil.params)

        # Determine Covariance Matrices Per Coefficient Of The Model [(p x p) NumPy array]
        coeff_cov_beta.append(results.cov_params())
        coeff_cov_beta_fil.append(results_fil.cov_params())

        # Define Residuals And Weighted Aggregate
        Y_pred = results.predict()
        residuals = Y - Y_pred

        total_ssr += np.sum(residuals**2)  # Sum of squared residuals
        total_sst += np.sum((Y - np.mean(Y))**2)  # Total sum of squares

        Y_pred_fil = results_fil.predict()
        residuals_fil = Y_fil - Y_pred_fil

        total_ssr_fil += np.sum(residuals_fil**2)  # Sum of squared residuals
        total_sst_fil += np.sum((Y_fil - np.mean(Y_fil))**2)  # Total sum of squares
        
    # Define Output
    model_coef = np.array(coeff_models)  # (9405, coef)
    model_coef_fil = np.array(coeff_models_fil)  # (9405, coef)

    cov_beta_array = np.array(coeff_cov_beta)  # (9405, coef, coef)
    cov_beta_fil_array = np.array(coeff_cov_beta_fil)  # (9405, coef, coef)

    # Define Weighted Aggregate
    r_squared = 1 - (total_ssr / total_sst)
    r_squared_fil = 1 - (total_ssr_fil / total_sst_fil)

    # Print Progress Statement
    print(f"\nUnfiltered R²  ➜  {r_squared:.4f}")
    print(f"Filtered R²    ➜  {r_squared_fil:.4f}")

    # Reconstruct The Original Input Matrix Shape And Add Back The Zero Entries
    CS_reconstructed = np.zeros((CS_array_original.shape[0], model_coef.shape[1]))
    CS_reconstructed[nonzero_mask] = model_coef

    CS_reconstructed_fil = np.zeros((CS_array_original_fil.shape[0], model_coef_fil.shape[1]))
    CS_reconstructed_fil[nonzero_mask] = model_coef_fil

    #split The Reconstructed Matrix Into The C And S Arrays
    C_reconstructed = CS_reconstructed[:9409,:]
    S_reconstructed = CS_reconstructed[9409:,:]

    C_reconstructed_fil = CS_reconstructed_fil[:9409,:]
    S_reconstructed_fil = CS_reconstructed_fil[9409:,:]

    # Recreate A 3D Array Of The Matrices (C, S, Model Coefficients)
    reconstructed_data_arr = np.stack([C_reconstructed, S_reconstructed], axis=2)
    reconstructed_data_arr_fil = np.stack([C_reconstructed_fil, S_reconstructed_fil], axis=2)

    #Reshape (Unflatten In Same Way) The Array To Get The Needed Input For The Spherical Harmonics Code
    shape = (97, 97, model_coef.shape[1], 2)
    SH_arr = reconstructed_data_arr.reshape(shape)
    SH_arr_fil = reconstructed_data_arr_fil.reshape(shape)

    # Create empty arrays to hold the full (97, 97, p, p, 2) structures
    cov_SH_arr = np.zeros((97, 97, model_coef.shape[1], model_coef.shape[1], 2))
    cov_fil_SH_arr = np.zeros((97, 97, model_coef.shape[1], model_coef.shape[1], 2))

    # Find the indices where coefficients are non-zero (True in the mask)
    nonzero_indices = np.where(nonzero_mask)[0]

    # Loop over the nonzero coefficients and assign values accordingly
    for idx, flat_idx in enumerate(nonzero_indices):
        if flat_idx < 9409:
            # It's a C coefficient
            i = flat_idx // 97
            j = flat_idx % 97
            cov_SH_arr[i, j, :, :, 0] = cov_beta_array[idx]
            cov_fil_SH_arr[i, j, :, :, 0] = cov_beta_fil_array[idx]
        else:
            # It's an S coefficient
            s_idx = flat_idx - 9409
            i = s_idx // 97
            j = s_idx % 97
            cov_SH_arr[i, j, :, :, 1] = cov_beta_array[idx]
            cov_fil_SH_arr[i, j, :, :, 1] = cov_beta_fil_array[idx]

    # Print Completion Statement
    print(f"\nmodel_coef  ➜  {model_coef.shape}")
    print(f"SH_arr      ➜  {SH_arr.shape}")
    print(f"cov_SH_arr  ➜  {cov_SH_arr.shape}")

    return model_coef, model_coef_fil, SH_arr, SH_arr_fil, cov_SH_arr, cov_fil_SH_arr

# Define The Total Spherical Harmonics Heatmap Values
def EWH_grid(SH_arr, cov_SH_arr, year_lst, sample_time, lat_precis=30, lon_precis=60, lat_range=(-np.pi/2, np.pi/2), lon_range=(-np.pi, np.pi), max_order=96, J2=False, calc_uncertainty=False, excel_output=True, file_name='output.xlsx'):
    """
    Calculate Equivalent Water Height (EWH) grid from spherical harmonics coefficients.

    Args:
        SH_arr (np.ndarray): 4D array of model coefficients in spherical harmonics format with shape (97, 97, coefficients, 2).
        cov_SH_arr (np.ndarray): 5D array of covariance matrices with shape (97, 97, coefficients, coefficients, 2).
        year_lst (list[int]): List of years included in the regression model.
        sample_time (float): Time in days at which the model is sampled.
        lat_precis (int, optional): Number of latitude points in the grid. Defaults to 30.
        lon_precis (int, optional): Number of longitude points in the grid. Defaults to 60.
        lat_range (tuple, optional): Latitude range in radians as (min_lat, max_lat). Defaults to (-π/2, π/2).
        lon_range (tuple, optional): Longitude range in radians as (min_lon, max_lon). Defaults to (-π, π).
        max_order (int, optional): Maximum degree/order of spherical harmonics. Defaults to 96.
        J2 (bool, optional): Include (True) or exclude (False) the J2 term in calculations. Defaults to False.
        calc_uncertainty (bool, optional): Whether to calculate uncertainty of EWH values. Defaults to False.
        excel_output (bool, optional): Whether to save results to an Excel file. Defaults to True.
        file_name (str, optional): Name of the Excel output file (must end with '.xlsx'). Defaults to 'output.xlsx'.

    Returns:
        - earth_grid_EWH (np.ndarray): 2D array of Equivalent Water Height [mm], shape (lat_precis, lon_precis).
        - earth_grid_EWH_uncertainty (np.ndarray or float): 2D array of EWH uncertainties [mm] if calculated; otherwise np.nan.

    Notes:
        - Always close the Excel file before running this function to avoid file access conflicts.
        - Latitude and longitude inputs and outputs are in radians internally; output grid corresponds to latitudes (y-axis) and longitudes (x-axis).
    """

    # Define Constants
    mu = 3.9860044150e+14  # In m^3/s^2
    Re = 6378.13630000e+3  # In m
    r = Re  # Gravity measured at Earth's surface
    rho_water = 1000     #kg/m^3 density of water
    rho_average = 5515    #kg/m^3 average surface density of the earth
    max_order = 96        #max order of degrees

    # Define Coordinate Grid In Radians
    colat_range = (((np.pi / 2) - lat_range[1]), ((np.pi / 2) - lat_range[0]))  # Convert latitude into colatitude and switch the order around such that the first value is closest to the North Pole
    colat_array = np.linspace(colat_range[0], colat_range[1], lat_precis)
    colon_range = ((lon_range[0] + np.pi), (lon_range[1] + np.pi))  # Convert longitude into colongitude
    colon_array = np.linspace(colon_range[0], colon_range[1], lon_precis)

    # Set Up Coefficient Array
    row_values = np.arange(2, 97)[:, np.newaxis]  # Shape (95, 1), degree n from 2 to 96
    col_values = np.arange(0, 97)[np.newaxis, :]  # Shape (1, 97), order m from 0 to 96

    rows = np.tile(row_values, (1, 97))  # Shape (95, 97)
    cols = np.tile(col_values, (95, 1))  # Shape (95, 97)

    coefficient_array = np.stack((rows, cols), axis=-1)  # Shape (95, 97, 2), First = Degree (n), Second = Order (m)

    n = coefficient_array[:, :, 0]
    m = coefficient_array[:, :, 1]
    
    # Define Legendre Normalization Since Data States Fully Normalized
    valid_mask = m <= n # Mask invalid entries where m > n (those are not valid for Legendre)
    delta_0_m = (m == 0).astype(np.float64)  # 1 if m == 0 else 0
    numer = (2 - delta_0_m) * ((2 * n) + 1) * factorial(n - m)
    denom = factorial(n + m)
    norm = np.zeros_like(n, dtype=np.float64)
    norm[valid_mask] = np.sqrt(numer[valid_mask] / denom[valid_mask])

    if not J2:
        norm[0, 0] = 0.0 # Replace The J2 Norm Value With Zero If Not Wanted
    
    # Define Love Numbers
    degrees, love_numbers = get_love_numbers(max_order)
    love_array = love_numbers[n.astype(int)]

    # Define Basis Values
    basis_val = T_row(sample_time, year_lst)

    # Extract Data
    SH_arr = SH_arr[2:, :, :, :]           # Now shape (95, 97, n_coef, 2)
    cov_SH_arr = cov_SH_arr[2:, :, :, :, :]  # Now shape (95, 97, n_coef, n_coef, 2)

    C_array = np.tensordot(SH_arr[:, :, :, 0], basis_val, axes=(2, 0))  # shape (95, 97)
    S_array = np.tensordot(SH_arr[:, :, :, 1], basis_val, axes=(2, 0))  # shape (95, 97)

    # Initialise Datasets
    earth_grid_rho = np.zeros((len(colat_array), len(colon_array)))
    earth_grid_rho_uncertainty = np.zeros((len(colat_array), len(colon_array)))
    
    earth_grid_EWH = np.zeros((len(colat_array), len(colon_array)))
    earth_grid_EWH_uncertainty = np.zeros((len(colat_array), len(colon_array)))

    # Initialise Progress Bar
    pbar = tqdm.tqdm(total=(lat_precis * lon_precis), desc="Processing Coordinates")
    
    # Compute potential for each latitude/longitude pair
    for j, colat in enumerate(colat_array): # Loop over latitudes
        # Define lmpv Values
        lpmv_values = lpmv(m, n, np.cos(colat))
        lpmv_values = np.nan_to_num(lpmv_values, nan=0.0)
        

        for i, lon in enumerate(colon_array): # Loop over longitudes
            # Define Value
            surface_density_terms = Re * (rho_average/3) * ((C_array * np.cos(m * lon)) + (S_array * np.sin(m * lon))) * (norm * lpmv_values) * ((1 + (2 * n))/(1 + love_array))
            surface_density = np.sum(surface_density_terms)

            # Append Value To Coordinate Grid
            earth_grid_rho[j,i] = surface_density

            if calc_uncertainty:
                # Determine Uncertainty
                constant = Re * (rho_average / 3)

                basis_cos = (norm * lpmv_values) * np.cos(m * lon) * ((1 + (2 * n))/(1 + love_array))
                basis_sin = (norm * lpmv_values) * np.sin(m * lon) * ((1 + (2 * n))/(1 + love_array))

                v_cos = basis_cos[:, :, np.newaxis] * basis_val[np.newaxis, np.newaxis, :]  # (95, 97, n_coef)
                v_sin = basis_sin[:, :, np.newaxis] * basis_val[np.newaxis, np.newaxis, :]  # (95, 97, n_coef)

                # Now compute variance as sum of quadratic forms over (n,m)
                var_cos = 0.0
                var_sin = 0.0
                for deg in range(v_cos.shape[0]):
                    for ord_ in range(v_cos.shape[1]):
                        vc = v_cos[deg, ord_, :]  # (n_coef,)
                        vs = v_sin[deg, ord_, :]  # (n_coef,)

                        C_cov = cov_SH_arr[deg, ord_, :, :, 0]  # (n_coef, n_coef)
                        S_cov = cov_SH_arr[deg, ord_, :, :, 1]  # (n_coef, n_coef)

                        var_cos += vc @ C_cov @ vc
                        var_sin += vs @ S_cov @ vs

                variance = var_cos + var_sin

                # Append Uncertainty To Coordinate Grid
                earth_grid_rho_uncertainty[j, i] = np.sqrt(variance) * constant

            # Update Progress Bar
            pbar.update(1)

    # Close Progress Bar
    pbar.close()

    # Translate Surface Density From kg/m^2 To EWH [mm]
    earth_grid_EWH = (earth_grid_rho / rho_water) * 1e3
    
    # Print Dataset Characteristics
    print("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    print("True Values")
    print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    print("EWH max [mm]:", np.max(earth_grid_EWH))
    print("EWH min [mm]:", np.min(earth_grid_EWH))
    print("EWH mean [mm]:", np.mean(earth_grid_EWH))
    print("EWH std dev [mm]:", np.std(earth_grid_EWH))
    print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")

    if calc_uncertainty:
        # Translate Surface Density From kg/m^2 To EWH [mm]
        earth_grid_EWH_uncertainty = (earth_grid_rho_uncertainty / rho_water) * 1e3

        # Print Dataset Characteristics
        print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        print("Uncertainty Values")
        print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        print("EWH max [mm]:", np.max(earth_grid_EWH_uncertainty))
        print("EWH min [mm]:", np.min(earth_grid_EWH_uncertainty))
        print("EWH mean [mm]:", np.mean(earth_grid_EWH_uncertainty))
        print("EWH std dev [mm]:", np.std(earth_grid_EWH_uncertainty))
        print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")
    
    else:
        earth_grid_EWH_uncertainty = np.nan

    if excel_output:
        # Format Potential And Acceleration Data
        lat_array = np.linspace(lat_range[1], lat_range[0], lat_precis)
        lat_array_degree = np.degrees(lat_array)
        lat_array_degree = lat_array_degree.reshape(-1, 1)
        EWH_first_column = np.vstack([["EWH [m]"], lat_array_degree])

        EWH_grid_data = np.vstack((np.degrees(colon_array), earth_grid_EWH))
        EWH_grid_data = np.hstack((EWH_first_column, EWH_grid_data))

        # Define DataFrames For Excel Output
        df = pd.DataFrame(EWH_grid_data)

        # Create File Path
        file_path = os.path.join("LSQR_Analysis/Output", file_name)

        # Save To Different Sheets In Excel Files
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Equivalent Water Height", index=False, header=False)
        
        # Load the workbook to modify styles
        wb = openpyxl.load_workbook(file_path)

        # Apply formatting to the last two sheets
        for sheet_name in ["Equivalent Water Height"]:
            ws = wb[sheet_name]

            # Bold the first row (headers)
            for cell in ws[1]:  # First row
                cell.font = openpyxl.styles.Font(bold=True)

            # Bold the first column (excluding header row)
            for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=1):
                for cell in row:
                    cell.font = openpyxl.styles.Font(bold=True)
        
        # Save the modified file
        wb.save(file_path)

        # Print Completion Statement
        print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
        print("Data Written To {}".format(file_name))
        print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")

    return earth_grid_EWH, earth_grid_EWH_uncertainty














def render_single(earth_grid_EWH, date, sample_time, lat_range=(-np.pi/2, np.pi/2), lon_range=(-np.pi, np.pi)):
    # Define Analysis Region In Degrees
    lon_range_deg = np.degrees(lon_range)
    lat_range_deg = np.degrees(lat_range)

    # Load Map Image & Take Its Dimensions
    map_img = plt.imread("Gravity_Maps/Mercator_Map.jpg")  # Mercator_Map
    img_height, img_width, _ = map_img.shape

    # Define Global Latitude & Longitude Range Of The Image
    global_lon_range = (-180, 180)
    global_lat_range = (-90, 90)

    # Calculate Cropping Indices For Longitude
    lon_min_idx = int((lon_range_deg[0] - global_lon_range[0]) / (global_lon_range[1] - global_lon_range[0]) * img_width)
    lon_max_idx = int((lon_range_deg[1] - global_lon_range[0]) / (global_lon_range[1] - global_lon_range[0]) * img_width)

    # Calculate Cropping Indices For Latitude (flip due to image orientation)
    lat_min_idx = int((global_lat_range[1] - lat_range_deg[1]) / (global_lat_range[1] - global_lat_range[0]) * img_height)
    lat_max_idx = int((global_lat_range[1] - lat_range_deg[0]) / (global_lat_range[1] - global_lat_range[0]) * img_height)

    # Crop the map image to match the lat/lon range (latitude indices flipped)
    cropped_map = map_img[lat_min_idx:lat_max_idx, lon_min_idx:lon_max_idx]

    # Create the heatmap
    plt.figure(figsize=(10, 5))

    # Plot the cropped map image with the adjusted extent
    plt.imshow(cropped_map, extent=[lon_range_deg[0], lon_range_deg[1], lat_range_deg[0], lat_range_deg[1]], aspect='auto')

    # Overlay the heatmap with transparency
    plt.imshow(earth_grid_EWH, extent=[lon_range_deg[0], lon_range_deg[1], lat_range_deg[0], lat_range_deg[1]], origin="upper", alpha=0.6, cmap="coolwarm", aspect="auto")  # Replace With "seismic" For More Contrast
    
    # Add labels and title
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.title(f"LSQR Model At t = {sample_time} Days ({date[:2]}-{date[2:4]}-{date[4:]})")
    plt.colorbar(label="Equivalent Water Height [m]")

    # Show And Save The Plot
    filename = f"LSQR_Analysis/Output/EWH_Plot_{date}.png"
    plt.savefig(filename)
    print(f"✅ Heatmap Saved To {filename}")
    plt.show()

    return
