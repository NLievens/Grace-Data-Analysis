"""
Data Reader for GRACE and GRACE-FO Parsed Datasets

This script reads and processes GRACE and GRACE-FO gravity field data from a directory
containing pre-parsed text files. Each file includes spherical harmonic coefficients 
along with metadata such as start and end dates and standard deviations.

Main Features:
--------------
- Loads all `.txt` files from the "Data/Parsed_Data" folder.
- Extracts date information from each file and stores it in `DDMMYYYY_DDMMYYYY` format.
- Parses data into a structured 3D NumPy array [degree, order, values].
- Organizes data by year and month into a 5D array of shape: 
  (n_years, 12 months, 97, 97, 4) (where 4 = [clm, slm, clm_std, slm_std]).
- Maps monthly time points into a uniform structure, inserting `"empty"` if data is missing.
- Builds a structured date array aligned with the data array.

Dependencies:
-------------
- `numpy`
- `tqdm`
- `pathlib`
- `datetime`
- `collections`

Output Variables:
-----------------
- `data_year_arr` : np.ndarray
    A 5D array with shape (n_years, 12, 97, 97, 4) containing the data
- `date_year_arr` : np.ndarray
    A 2D array with shape (n_years, 12) containing date strings or `"empty"`

Raises:
-------
- `FileNotFoundError` if the expected data folder is not found.
"""

# External Imports
import tqdm
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Define Lists
data_lib = []
file_name_lst = []
date_lst = []
sorted_date_lst = []

# Read Data Folder
folder_path = Path("Data/Parsed_Data") # Define the correct folder path

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
