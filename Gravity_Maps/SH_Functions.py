'''
DOCSTRING
'''

# External Imports
import os
import tqdm
import openpyxl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import lpmv, factorial

# Define Spherical Harmonics Functions
def spherical_harmonics_date(data_array, lat_precis=30, lon_precis=60, lat_range=(-np.pi/2, np.pi/2), lon_range=(-np.pi, np.pi), J2=False, excel_output=True, file_name='output.xlsx'):
    """
    Args:
        data_array: 3D array with shape (97, 97, 4) of C and S coefficients and their standard deviations
        lat_precis: amount of latitudinal coordinate points
        lon_precis: amount of longitudinal coordinate points
        lat_range: tuple of minimum and maximum lattitude in radians
        lon_range: tuple of minimum and maximum longitude in radians
        J2: include (True) or exclude (False)
        excel_output: create excel sheet (True) or not (False)
        file_name: data output name file (Always add .xlsx)
    
    Returns:
        earth_grid_pot: 2D array of gravitational potential [km^2/s^2] along a coordinate grid with x as longitude and y as latitude
        earth_grid_acc: 2D array of gravitational radial acceleration [km/s^2] along a coordinate grid with x as longitude and y as latitude
    
    Notes:
        always close the excel file before running
    """

    # Define Constants
    mu = 3.9860044150e+5  # In km^3/s^2
    Re = 6378.13630000  # In km
    r = Re  # Gravity measured at Earth's surface

    # Define Coordinate Grid In Radians
    colat_range = (((np.pi / 2) - lat_range[1]), ((np.pi / 2) - lat_range[0]))  # Convert latitude into colatitude and switch the order around such that the first value is closest to the North Pole
    colat_array = np.linspace(colat_range[0], colat_range[1], lat_precis)
    colon_range = ((lon_range[0] + np.pi), (lon_range[1] + np.pi))  # Convert longitude into colongitude
    colon_array = np.linspace(colon_range[0], colon_range[1], lon_precis)
    earth_grid_pot = np.zeros((len(colat_array), len(colon_array)))
    earth_grid_acc = np.zeros((len(colat_array), len(colon_array)))

    # Format Data
    trimmed_data = data_array[2:, :, :-2] # Remove Degree 0 And 1 Rows And Standard Deviations
    C_array = trimmed_data[:, :, 0]
    S_array = trimmed_data[:, :, 1]
    
    # Set Up Coefficient Array
    row_values = np.arange(2, 97)[:, np.newaxis]  # Shape (95, 1), degree n from 2 to 96
    col_values = np.arange(0, 97)[np.newaxis, :]  # Shape (1, 97), order m from 0 to 96

    rows = np.tile(row_values, (1, 97))  # Shape (95, 97)
    cols = np.tile(col_values, (95, 1))  # Shape (95, 97)

    coefficient_array = np.stack((rows, cols), axis=-1)  # Shape (95, 97, 2), First = Degree (n), Second = Order (m)

    n = coefficient_array[:, :, 0]
    m = coefficient_array[:, :, 1]
    
    # Define Normalization For Associated Legendre Polynomials
    valid_mask = m <= n # Mask invalid entries where m > n (those are not valid for Legendre)
    delta_0_m = (m == 0).astype(np.float64)  # 1 if m == 0 else 0
    numer = (2 - delta_0_m) * ((2 * n) + 1) * factorial(n - m)
    denom = factorial(n + m)
    norm = np.zeros_like(n, dtype=np.float64)
    norm[valid_mask] = np.sqrt(numer[valid_mask] / denom[valid_mask])

    if not J2:
        norm[0, 0] = 0.0 # Replace The J2 Norm Value With Zero If Not Wanted
    
    # Initialise Progress Bar
    pbar = tqdm.tqdm(total=(lat_precis * lon_precis), desc="Processing Coordinates")

    # Compute potential for each latitude/longitude pair
    for j, colat in enumerate(colat_array): # Loop over latitudes
        # Define lmpv Values
        lpmv_values = lpmv(m, n, np.cos(colat))
        lpmv_values = np.nan_to_num(lpmv_values, nan=0.0)

        for i, lon in enumerate(colon_array): # Loop over longitudes
            # Define Constants (n=0 And n=1 Terms)
            grav_pot = (mu/r)
            grav_acc = (mu/(r**2))

            # Define Values
            pot_terms = (mu/Re) * ((C_array * np.cos(m * lon)) + (S_array * np.sin(m * lon))) * (norm * lpmv_values)
            acc_terms = (1/r) * (n + 1) * pot_terms

            grav_pot += np.sum(pot_terms)
            grav_acc += np.sum(acc_terms)

            # Append Values To Coordinate Grid
            earth_grid_pot[j,i] = grav_pot
            earth_grid_acc[j,i] = grav_acc

            # Update Progress Bar
            pbar.update(1)
    
    # Close Progress Bar
    pbar.close()

    # Translate Acceleration Data From km/s2 To mGal
    earth_grid_acc =  1e8 * earth_grid_acc

    if excel_output:
        # Format Potential And Acceleration Data
        lat_array = np.linspace(lat_range[1], lat_range[0], lat_precis)
        lat_array_degree = np.degrees(lat_array)
        lat_array_degree = lat_array_degree.reshape(-1, 1)
        pot_first_column = np.vstack([["km2/s2"], lat_array_degree])
        acc_first_column = np.vstack([["mGal"], lat_array_degree])

        pot_grid_data = np.vstack((np.degrees(colon_array), earth_grid_pot))
        pot_grid_data = np.hstack((pot_first_column, pot_grid_data))

        acc_grid_data = np.vstack((np.degrees(colon_array), earth_grid_acc))
        acc_grid_data = np.hstack((acc_first_column, acc_grid_data))

        # Define DataFrames For Excel Output
        df1 = pd.DataFrame(C_array)
        df2 = pd.DataFrame(S_array)
        df3 = pd.DataFrame(norm)
        df4 = pd.DataFrame(pot_grid_data)
        df5 = pd.DataFrame(acc_grid_data)

        # Create File Path
        file_path = os.path.join("Gravity_Maps/Output", file_name)

        # Save To Different Sheets In Excel Files
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            df1.to_excel(writer, sheet_name="C-Coefficients", index=False)
            df2.to_excel(writer, sheet_name="S-Coefficients", index=False)
            df3.to_excel(writer, sheet_name="Normalization Factors", index=False)
            df4.to_excel(writer, sheet_name="Gravity Potential", index=False, header=False)
            df5.to_excel(writer, sheet_name="Gravity Acceleration", index=False, header=False)
        
        # Load the workbook to modify styles
        wb = openpyxl.load_workbook(file_path)

        # Apply formatting to the last two sheets
        for sheet_name in ["Gravity Potential", "Gravity Acceleration"]:
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
        print(f"\n✅ Data Written To {file_name}\n")

    return earth_grid_acc

def spherical_harmonics_baseline(data_lib, sorted_date_lst, lat_precis=30, lon_precis=60, lat_range=(-np.pi/2, np.pi/2), lon_range=(-np.pi, np.pi), J2=False):
    # Define Data List
    data_lst_acc = []
    data_lst_pot = []

    # Extract Data
    data_dict = {entry['date']: entry['data_array'] for entry in data_lib}

    # Initialise Progress Bar
    pbar = tqdm.tqdm(total=(len(sorted_date_lst) * lat_precis * lon_precis), desc="Processing Dates")

    # Loop Over All Defined Dates
    for i, date in enumerate(sorted_date_lst):
        # Define Constants
        mu = 3.9860044150e+5  # In km^3/s^2
        Re = 6378.13630000  # In km
        r = Re  # Gravity measured at Earth's surface

        # Define Coordinate Grid In Radians
        colat_range = (((np.pi / 2) - lat_range[1]), ((np.pi / 2) - lat_range[0]))  # Convert latitude into colatitude and switch the order around such that the first value is closest to the North Pole
        colat_array = np.linspace(colat_range[0], colat_range[1], lat_precis)
        colon_range = ((lon_range[0] + np.pi), (lon_range[1] + np.pi))  # Convert longitude into colongitude
        colon_array = np.linspace(colon_range[0], colon_range[1], lon_precis)
        earth_grid_pot = np.zeros((len(colat_array), len(colon_array)))
        earth_grid_acc = np.zeros((len(colat_array), len(colon_array)))

        # Retrieve Data Array
        data_array = data_dict[date]

        # Format Data
        trimmed_data = data_array[2:, :, :-2] # Remove Degree 0 And 1 Rows And Standard Deviations
        C_array = trimmed_data[:, :, 0]
        S_array = trimmed_data[:, :, 1]
        
        # Set Up Coefficient Array
        row_values = np.arange(2, 97)[:, np.newaxis]  # Shape (95, 1), degree n from 2 to 96
        col_values = np.arange(0, 97)[np.newaxis, :]  # Shape (1, 97), order m from 0 to 96

        rows = np.tile(row_values, (1, 97))  # Shape (95, 97)
        cols = np.tile(col_values, (95, 1))  # Shape (95, 97)

        coefficient_array = np.stack((rows, cols), axis=-1)  # Shape (95, 97, 2), First = Degree (n), Second = Order (m)

        n = coefficient_array[:, :, 0]
        m = coefficient_array[:, :, 1]
        
        # Define Normalization For Associated Legendre Polynomials
        valid_mask = m <= n # Mask invalid entries where m > n (those are not valid for Legendre)
        delta_0_m = (m == 0).astype(np.float64)  # 1 if m == 0 else 0
        numer = (2 - delta_0_m) * ((2 * n) + 1) * factorial(n - m)
        denom = factorial(n + m)
        norm = np.zeros_like(n, dtype=np.float64)
        norm[valid_mask] = np.sqrt(numer[valid_mask] / denom[valid_mask])

        if not J2:
            norm[0, 0] = 0.0 # Replace The J2 Norm Value With Zero If Not Wanted

        # Compute potential for each latitude/longitude pair
        for j, colat in enumerate(colat_array): # Loop over latitudes
            # Define lmpv Values
            lpmv_values = lpmv(m, n, np.cos(colat))
            lpmv_values = np.nan_to_num(lpmv_values, nan=0.0)

            for i, lon in enumerate(colon_array): # Loop over longitudes
                # Define Constants (n=0 And n=1 Terms)
                grav_pot = (mu/r)
                grav_acc = (mu/(r**2))

                # Define Values
                pot_terms = (mu/Re) * ((C_array * np.cos(m * lon)) + (S_array * np.sin(m * lon))) * (norm * lpmv_values)
                acc_terms = (1/r) * (n + 1) * pot_terms

                grav_pot += np.sum(pot_terms)
                grav_acc += np.sum(acc_terms)

                # Append Values To Coordinate Grid
                earth_grid_pot[j,i] = grav_pot
                earth_grid_acc[j,i] = grav_acc

                # Update Progress Bar
                pbar.update(1)

        # Append Values To Corresponding Lists
        data_lst_pot.append(earth_grid_pot)
        data_lst_acc.append(1e8 * earth_grid_acc)
    
    # Close Progress Bar
    pbar.close()

    # Define The Averages
    average_pot = np.mean(data_lst_pot, axis=0)
    average_acc = np.mean(data_lst_acc, axis=0)

    # Format Potential And Acceleration Data
    lat_array = np.linspace(lat_range[1], lat_range[0], lat_precis)
    lat_array_degree = np.degrees(lat_array)
    lat_array_degree = lat_array_degree.reshape(-1, 1)
    pot_first_column = np.vstack([["km2/s2"], lat_array_degree])
    acc_first_column = np.vstack([["mGal"], lat_array_degree])

    pot_grid_data = np.vstack((np.degrees(colon_array), average_pot))
    pot_grid_data = np.hstack((pot_first_column, pot_grid_data))

    acc_grid_data = np.vstack((np.degrees(colon_array), average_acc))
    acc_grid_data = np.hstack((acc_first_column, acc_grid_data))

    # Convert NumPy arrays to DataFrames
    df_pot = pd.DataFrame(pot_grid_data)
    df_acc = pd.DataFrame(acc_grid_data)

    # Create File Path
    file_path = os.path.join("Gravity_Maps/Output", "Baseline Gravity Field.xlsx")

    # Write to Excel file with separate sheets
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df_pot.to_excel(writer, sheet_name='Potential', index=False, header=False)
        df_acc.to_excel(writer, sheet_name='Acceleration', index=False, header=False)

    # Load the workbook to modify styles
    wb = openpyxl.load_workbook(file_path)

    # Apply formatting to the last two sheets
    for sheet_name in ["Potential", "Acceleration"]:
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
    print(f"\n✅ Data Written To Baseline Gravity Field.xlsx\n")

    return average_acc

def render_single(earth_grid_acc, date, lat_range=(-np.pi/2, np.pi/2), lon_range=(-np.pi, np.pi)):
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
    plt.imshow(earth_grid_acc, extent=[lon_range_deg[0], lon_range_deg[1], lat_range_deg[0], lat_range_deg[1]], origin="upper", alpha=0.6, cmap="coolwarm", aspect="auto")  # Replace With "seismic" For More Contrast
    
    # Add labels and title
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.title("Gravity {}".format(date))
    plt.colorbar(label="Gravity Acceleration [mGal]")

    # Show And Save The Plot
    filename = f"Gravity_Maps/Output/Gravity_Plot_{date}.png"
    plt.savefig(filename)
    print(f"✅ Heatmap Saved To {filename}")
    plt.show()

    return

def render_double(earth_grid_acc_1, earth_grid_acc_2, selected_dates, lat_range=(-np.pi/2, np.pi/2), lon_range=(-np.pi, np.pi)):
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

    # Find global min and max across both datasets To Have Same Scaling
    vmin = min(earth_grid_acc_1.min(), earth_grid_acc_2.min())
    vmax = max(earth_grid_acc_1.max(), earth_grid_acc_2.max())

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)  # Share x and y axes

    # Ensure the subplots stay rectangular by setting the aspect ratio
    for ax in axes:
        ax.set_box_aspect(0.7)  # 1 means a square; adjust for wider rectangles

    # First plot (left)
    axes[0].imshow(cropped_map, extent=[lon_range_deg[0], lon_range_deg[1], lat_range_deg[0], lat_range_deg[1]], aspect='auto')
    heatmap1 = axes[0].imshow(earth_grid_acc_1, extent=[lon_range_deg[0], lon_range_deg[1], lat_range_deg[0], lat_range_deg[1]], origin="upper", alpha=0.6, cmap="coolwarm", aspect="auto", vmin=vmin, vmax=vmax)

    axes[0].set_title("Gravity {}".format(selected_dates[0]))
    axes[0].set_xlabel("Longitude [deg]")
    axes[0].set_ylabel("Latitude [deg]")

    # Second plot (right)
    axes[1].imshow(cropped_map, extent=[lon_range_deg[0], lon_range_deg[1], lat_range_deg[0], lat_range_deg[1]], aspect='auto')
    heatmap2 = axes[1].imshow(earth_grid_acc_2, extent=[lon_range_deg[0], lon_range_deg[1], lat_range_deg[0], lat_range_deg[1]], origin="upper", alpha=0.6, cmap="coolwarm", aspect="auto", vmin=vmin, vmax=vmax)

    axes[1].set_title("Gravity {}".format(selected_dates[1]))
    axes[1].set_xlabel("Longitude [deg]")
    axes[1].set_ylabel("Latitude [deg]")

    # Add a shared horizontal colorbar centered below both subplots
    cbar = fig.colorbar(heatmap1, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
    cbar.set_label("Gravity Anomaly [mGal]")

    # Adjust subplot spacing to maintain rectangular shapes
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.238, wspace=0.1)

    # Show And Save The Plot
    filename = f"Gravity_Maps/Output/Gravity_Plot_{selected_dates[0]}_&_{selected_dates[1]}.png"
    plt.savefig(filename)
    print(f"✅ Heatmap Saved To {filename}")
    plt.show()

