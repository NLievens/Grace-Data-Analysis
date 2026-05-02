
# External Imports
import io
import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Define Data Reading Function
def read_grace_data(folder_path):
    '''
    Reads and processes GRACE and GRACE-FO data from the specified folder.

    Args:
        folder_path (Path): Path to the folder containing the parsed data files.
        
    Returns:
        data_year_arr (np.ndarray): A 5D array with shape (n_years, 12, 97, 97, 4) containing the data
        date_year_arr (np.ndarray): A 2D array with shape (n_years, 12) containing date strings or "empty"
        data_lib (list): List of dicts with shape (236,) containing date strings and corresponding data arrays
        sorted_date_lst (list): List of reformatted dates as DDMMYYYY_DDMMYYYY strings for sorting with shape (236,)

    Raises:
        FileNotFoundError: If the expected data folder is not found.
    '''

    # Define Lists
    data_lib = []
    file_name_lst = []
    date_lst = []
    sorted_date_lst = []

    if folder_path.exists() and folder_path.is_dir(): # Check if the folder exists
        for file in folder_path.iterdir():
            file_name_lst.append(file.name)
    else:
        raise FileNotFoundError(
                "\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n"
                "ERROR // Folder Not Found!\n"
                "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n"
            )

    # Read Data Files And Append Data To Library
    for file_name in tqdm.tqdm(file_name_lst, desc="Reading Data"):
        is_data = False
        date = False

        data_matrix = np.zeros((97,97, 4)) # 3D Array With clm and slm (Down is degrees, Right is order)

        with open(f"{folder_path}/{file_name}", "r") as file:
            for line in file:
                if "# End of YAML header" in line:
                    is_data = True  # Switch to data section
                    date = True
                    continue
                
                if date == True: # Read The Date Of The Data From The First Dataline And Rewrite
                    data = line.split()
                    date_range = data[7][6:8] + data[7][4:6] + data[7][0:4] + "_" + data[8][6:8] + data[8][4:6] + data[8][0:4] #DDMMYYYY_DDMMYYYY
                    sorted_date_range = data[7][0:4] + data[7][4:6] + data[7][6:8] + "_" + data[8][0:4] + data[8][4:6] + data[8][6:8] #YYYYMMDD_YYYYMMDD
                    date = False
                
                if is_data:
                    data = line.split()
                    data_matrix[int(data[1]), int(data[2]), :] = [float(data[3]), float(data[4]), float(data[5]), float(data[6])] # Append lst Of C And S And C std And S std To Coordinate In Matrix // Create 3D Array

        data_lib.append({
            'date': str(date_range), # DDMMYYYY_DDMMYYYY
            'data_array': data_matrix
        })

        date_lst.append(str(sorted_date_range))

    # Sort Date List
    date_lst = sorted(date_lst)

    for date in date_lst:
        date = date[6:8] + date[4:6] + date[0:4] + "_" + date[15:17] + date[13:15] + date[9:13]
        sorted_date_lst.append(date)

    grouped = defaultdict(list)

    for date in sorted_date_lst:
        start_date = date.split('_')[0]  # 'DDMMYYYY'
        year = start_date[-4:]  # Get 'YYYY' part
        grouped[year].append(date)

    # Convert to a list of lists, sorted by year
    date_year_lst = [grouped[year] for year in sorted(grouped.keys())]

    # Flatten your list (assuming `data` is defined as before)
    flat_list = [item for sublist in date_year_lst for item in sublist]

    # Create mapping from YYYY-MM to entry
    date_map = {}
    for entry in flat_list:
        start_str = entry.split('_')[0]
        start_date = datetime.strptime(start_str, "%d%m%Y")
        key = start_date.strftime("%Y-%m")
        if key not in date_map:
            date_map[key] = entry

    # Get min and max year
    min_year = min(datetime.strptime(k, "%Y-%m") for k in date_map).year
    max_year = max(datetime.strptime(k, "%Y-%m") for k in date_map).year

    # Build the structured array: list of lists (one per year)
    date_year_lst = []
    for year in range(min_year, max_year + 1):
        year_entries = []
        for month in range(1, 13):
            key = f"{year}-{month:02d}"
            year_entries.append(date_map.get(key, "empty"))
        date_year_lst.append(year_entries)

    # Copy shape from date_year_arr
    n_years = len(date_year_lst)
    n_months = len(date_year_lst[0])  # Should be 12

    # Preallocate the 5D array
    data_year_arr = np.full((n_years, n_months, 97, 97, 4), np.nan)

    # Fill with data
    for year_idx, date_year_lst_row in enumerate(date_year_lst):
        for month_idx, date in enumerate(date_year_lst_row):
            if date == "empty":
                continue
            for entry in data_lib:
                if entry['date'] == date:
                    data_year_arr[year_idx, month_idx] = entry['data_array']
                    break

    # Transform To Array
    date_year_arr = np.array(date_year_lst)

    return data_year_arr, date_year_arr, data_lib, sorted_date_lst

# Define Geocentric Coefficients Extraction Function
def extract_geocentric_coefficients():

    # Define File Path
    file_path = Path("Data/LSQR_Data/geocenter_coefficients.txt")

    # Read The File
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Use .lower() and .strip() to find the line regardless of case or trailing equals signs
    try:
        data_start = next(i for i, line in enumerate(lines) 
                          if line.strip().lower().startswith("end of header")) + 1
    except StopIteration:
        raise ValueError("Could not find the end of the header in the TN-13 file.")

    # Now read the data (GRCOF2 lines)
    # The columns in TN-13 are: Key, L, M, Clm, Slm, C_std, S_std, Start, End
    df = pd.read_csv(io.StringIO("".join(lines[data_start:])), 
                     sep=r'\s+', header=None,
                     names=['key_name', 'L', 'M', 'Clm', 'Slm', 'C_std', 'S_std', 'start', 'end'])

    # Format the date key to match your 'month_date' logic (MMYYYY)
    # The 'start' column is YYYYMMDD.0000
    df['date_dt'] = pd.to_datetime(df['start'].astype(str).str[:8], format='%Y%m%d')
    df['lookup_key'] = df['date_dt'].dt.strftime('%m%Y')

    # Pivot so C10, C11, S11 are accessible by the date key
    # We need to handle the fact that each month has TWO lines (one for M=0, one for M=1)
    # This logic creates a nested dictionary: { '042002': {'C10': val, 'C11': val, 'S11': val}, ... }
    library = {}
    for key, group in df.groupby('lookup_key'):
        c10 = group[(group['L'] == 1) & (group['M'] == 0)]['Clm'].values[0]
        c11 = group[(group['L'] == 1) & (group['M'] == 1)]['Clm'].values[0]
        s11 = group[(group['L'] == 1) & (group['M'] == 1)]['Slm'].values[0]
        c10_std = group[(group['L'] == 1) & (group['M'] == 0)]['C_std'].values[0]
        c11_std = group[(group['L'] == 1) & (group['M'] == 1)]['C_std'].values[0]
        s11_std = group[(group['L'] == 1) & (group['M'] == 1)]['S_std'].values[0]
        library[key] = {'C10': c10, 'C11': c11, 'S11': s11, 'C10_std': c10_std, 'C11_std': c11_std, 'S11_std': s11_std}
        
    return library

# Define SLR Coefficients Extraction Function
def extract_slr_coefficients():
    '''
    Parses TN-14 SLR file into a dictionary keyed by MMYYYY.

    Returns:
        dict: A dictionary where keys are MMYYYY strings and values are dictionaries with C20 and C30 coefficients.
    '''

    # Define Correct File Path
    file_path = Path("Data/LSQR_Data/SLR_C20_C30.txt")

    # Read The File
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 1. Locate the start of the data block
    try:
        data_start = next(i for i, line in enumerate(lines) 
                          if line.strip().startswith('Product:')) + 1
    except StopIteration:
        raise ValueError("Could not find 'Product:' header in the TN-14 file. Check file format.")

    # 2. Read the data using the standard library io.StringIO
    # We use columns: 1 (DecYear), 2 (C20), 4 (C20_std), 5 (C30), 7 (C30_std)
    # Note: Column indices are 0-based in pandas read_csv
    data_str = "".join(lines[data_start:])
    df = pd.read_csv(io.StringIO(data_str), 
                     sep=r'\s+', 
                     header=None,
                     usecols=[1, 2, 4, 5, 7], 
                     names=['DecYear', 'C20', 'C20_std', 'C30', 'C30_std'])

    # 3. Convert Decimal Year to MMYYYY key
    def deyear_to_mmyyyy(dy):
        try:
            year = int(dy)
            # Fraction of year to month (0.0833 approx = 1 month)
            month = int(round((dy % 1) * 12)) + 1
            if month > 12: month = 12
            if month < 1: month = 1
            return f"{month:02d}{year}"
        except:
            return None

    df['key'] = df['DecYear'].apply(deyear_to_mmyyyy)
    
    # 4. Data Cleaning
    # Convert 'NaN' strings to actual numpy NaN objects
    df = df.replace('NaN', np.nan)
    # Drop any rows where DecYear parsing failed
    df = df.dropna(subset=['key'])

    # 5. Solve the Duplicate Index Problem
    # We sort by DecYear to ensure the most recent solution is at the bottom,
    # then group by 'key' and take the .last() record for that month.
    df = df.sort_values('DecYear')
    df_unique = df.groupby('key').last()

    # 6. Convert to Dictionary
    # Format: {'042002': {'C20': -0.000484, 'C30': nan}, ...}
    library = df_unique[['C20', 'C20_std', 'C30', 'C30_std']].to_dict('index')
    
    return library

# Define GIA Coefficients Extraction Function
def extract_GIA(max_degree=96):
    # Define Correct File Path
    file_path = Path("Data/LSQR_Data/ICE-6G_High_Res_Stokes_trend.txt")

    # Initialize matrices with zeros (size is max_degree + 1)
    C = np.zeros((max_degree + 1, max_degree + 1))
    S = np.zeros((max_degree + 1, max_degree + 1))
    
    # Read The File
    with open(file_path, 'r') as f:
        for line in f:
            # Skip empty lines
            if not line.strip(): continue
            
            parts = line.split()
            n = int(parts[0])
            m = int(parts[1])
            
            # Only store data within your requested 96-degree limit
            if n <= max_degree:
                C[n, m] = float(parts[2])
                S[n, m] = float(parts[3])
                    
    return C, S
