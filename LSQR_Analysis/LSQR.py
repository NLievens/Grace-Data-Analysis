'''
'''

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

# Internal Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data.Data_Reader import *
from Filters import *

# Define Delta Harmonic Coefficients
def compute_delta_harmonics(data_year_arr, date_year_arr, year_lst, max_order=96):
    # Indexing & Pre-allocation
    year_index_arr = np.array(year_lst) - 2002
    considered_date_1D = date_year_arr[year_index_arr].flatten()
    considered_data = data_year_arr[year_index_arr]     # Shape (Years, 12, Rows, Cols, 4)

    # Calculate t0
    first_valid = next((d for d in considered_date_1D if d.strip().lower() not in ("", "empty", "none")), None)
    if not first_valid:
        raise ValueError("❌ ERROR // No Valid Date Found")

    t0_year = datetime.strptime(first_valid.split("_")[0], "%d%m%Y").year
    t0_time = datetime(t0_year, 1, 1)

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

    print("\n✅ Read And Pre-Processed GEO, SLR, and GIA Data\n")

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
        C_orig, S_orig = month_data[:, :, 0].copy(), month_data[:, :, 1].copy()
        C_std, S_std   = month_data[:, :, 2].copy(), month_data[:, :, 3].copy()

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
            # Only replace C30 if it's not NaN (TN-14 has NaNs before 2012)
            if not np.isnan(SLR_vals['C30']):
                C_orig[3, 0] = SLR_vals['C30']
        
        # Remove Isostatic Rebound (GIA) (https://www.atmosp.physics.utoronto.ca/~peltier/data.php) (Inverse Sign Convention)
        dt_years = (current_dt - t0_time).days / 365.25

        C_orig += gia_C * dt_years
        S_orig += gia_S * dt_years

        # Filtering
        C_semi, S_semi = correlated_error_filter(C_orig, S_orig, max_order)
        C_fil, S_fil   = gaussian_filter(C_semi, S_semi)

        # Append Data To Lists
        data_C.append(C_fil)
        data_S.append(S_fil)
        data_C_std.append(C_std)
        data_S_std.append(S_std)
    
    # Convert Lists To Arrays
    data_C_array = np.array(data_C)           # Shape (Months, 97, 97)
    data_S_array = np.array(data_S)           # Shape (Months, 97, 97)
    
    # Define Delta Values By Subtracting Mean
    C_mean = np.mean(data_C_array, axis=0)
    S_mean = np.mean(data_S_array, axis=0)

    delta_C = data_C_array - C_mean
    delta_S = data_S_array - S_mean

    # Flatten & Concatenate
    CS_delta_vectors = []
    CS_std_delta_vectors = []
    for i in range(len(delta_C)):
        # Flatten C then S
        flat_CS = np.concatenate([delta_C[i].ravel(), delta_S[i].ravel()])
        flat_CS_std = np.concatenate([data_C_std[i].ravel(), data_S_std[i].ravel()])
        
        # Apply Mask To Remove Zero/Invalid Entries
        mask = np.concatenate([C_mean.ravel(), S_mean.ravel()]) != 0
        
        CS_delta_vectors.append(flat_CS[mask, np.newaxis])
        CS_std_delta_vectors.append(flat_CS_std[mask, np.newaxis])

    return delta_t_lst, CS_delta_vectors, CS_std_delta_vectors


# TEST
folder_path = Path("Data/Parsed_Data")
data_year_arr, date_year_arr, data_lib, sorted_date_lst = read_grace_data(folder_path)

year_lst = [2020]
delta_t_lst, CS_delta_vectors, CS_std_delta_vectors = compute_delta_harmonics(data_year_arr, date_year_arr, year_lst, max_order=96)







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
