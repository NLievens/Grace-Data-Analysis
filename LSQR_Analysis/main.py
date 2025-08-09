'''
DOCSTRING
'''

# External Imports
import os
import sys
import numpy as np
from datetime import date

# Internal Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data.Data_Reader import data_year_arr, date_year_arr, data_lib, sorted_date_lst
from LSQR_Functions import LSQR_coefficients, EWH_grid, render_single

# Functions
def clear_lines(n):
    for _ in range(n):
        sys.stdout.write("\033[F")  # Move cursor up one line
        sys.stdout.write("\033[K")  # Clear to end of line

def extract_min_max_dates_str(date_year_arr):
    flat = date_year_arr.flatten()
    
    # Remove "empty" entries
    valid = [d for d in flat if d.lower() != "empty"]
    
    # Extract starts and ends, convert to YYYYMMDD for easy comparison
    starts = []
    ends = []
    for rng in valid:
        start, end = rng.split("_")
        starts.append(start[4:] + start[2:4] + start[:2])  # YYYYMMDD
        ends.append(end[4:] + end[2:4] + end[:2])
    
    # Find min and max
    min_start = min(starts)
    max_end = max(ends)
    
    # Convert back to DDMMYYYY
    min_start_ddmmyyyy = min_start[6:] + min_start[4:6] + min_start[:4]
    max_end_ddmmyyyy = max_end[6:] + max_end[4:6] + max_end[:4]
    
    return min_start_ddmmyyyy, max_end_ddmmyyyy

def is_date_in_range(date_ddmmyyyy, min_start_ddmmyyyy, max_end_ddmmyyyy):
    def is_leap_year(year):
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    def is_valid_date_ddmmyyyy(date):
        if len(date) != 8 or not date.isdigit():
            return False
        day = int(date[:2])
        month = int(date[2:4])
        year = int(date[4:])
        if year < 1 or month < 1 or month > 12:
            return False
        days_in_month = [31, 29 if is_leap_year(year) else 28, 31, 30, 31, 30,
                         31, 31, 30, 31, 30, 31]
        if day < 1 or day > days_in_month[month - 1]:
            return False
        return True

    # Validate all inputs are 8-digit strings
    for d in [date_ddmmyyyy, min_start_ddmmyyyy, max_end_ddmmyyyy]:
        if len(d) != 8 or not d.isdigit():
            return False

    # Validate actual date correctness for the user date
    if not is_valid_date_ddmmyyyy(date_ddmmyyyy):
        return False

    # Convert DDMMYYYY to YYYYMMDD for string comparison
    def ddmmyyyy_to_yyyymmdd(d):
        return d[4:] + d[2:4] + d[:2]

    date = ddmmyyyy_to_yyyymmdd(date_ddmmyyyy)
    min_start = ddmmyyyy_to_yyyymmdd(min_start_ddmmyyyy)
    max_end = ddmmyyyy_to_yyyymmdd(max_end_ddmmyyyy)

    # Check if date is within range
    return min_start <= date <= max_end

# User Selects Analysis Choice
print("""
=== GRACE Data Selection Menu ===
      
1. Render Heatmap Of A Single Date [EWH]
2. Render Heatmap Of A Two Dates [EWH]
3. Render Heatmap Of Difference Between Two Dates [EWH]
4. Calculate Volume Changes Over User Specified Areas In Time [EWH]
5. Render Model Verification, i.e. Data Versus Model [EWH]
""")

mod_choice = input("Enter Your Choice: ").strip()

clear_lines(10)

if mod_choice == '1':
    # Print Confirmation Statement
    print('\n=== Single Date Heatmap ===\n')

    # Print Available Date Range
    min_start, max_end = extract_min_max_dates_str(date_year_arr)
    print(f"Available date range: {min_start[:2]}-{min_start[2:4]}-{min_start[4:]} to "f"{max_end[:2]}-{max_end[2:4]}-{max_end[4:]}\n")

    # Ask User For A Date
    user_date = input("Enter Date For Analysis [DDMMYYYY]: ")
    
    # Check Chosen Date Is Valid
    if is_date_in_range(user_date, min_start, max_end):
        clear_lines(4)
        print(f"\n✅ Chosen Date: {user_date[:2]}-{user_date[2:4]}-{user_date[4:]}\n") 

    else:
        print("\n❌ Date is outside the range.")
        exit()
    
    # Extract year from the chosen date
    chosen_year = int(user_date[4:])

    # Ask User For Entire Or Custom Dataset
    dat_choice = input("Do You Want To Customise The Year Range Of The Dataset? (Y/N): ").strip().lower()
    custom_length = 1

    if dat_choice == 'y':
        clear_lines(1)
        print(f"Custom Dataset Chosen (Beware Of Chosen Year <{chosen_year}>)")
        # Customize by specifying start and end year
        dataset_start_year = int(input(f"Enter Start Year (>= {min_start[4:]}): ").strip())
        dataset_end_year = int(input(f"Enter End Year (<= {max_end[4:]}): ").strip())

        # Validate input bounds
        min_year = int(min_start[4:])
        max_year = int(max_end[4:])
        if dataset_start_year < min_year or dataset_end_year > max_year or dataset_start_year > dataset_end_year:
            print("❌ Invalid Year Range Chosen")
            exit()

        year_lst = list(range(dataset_start_year, dataset_end_year + 1))
        custom_length += 2
    
    else:
        clear_lines(1)
        print("Full Dataset Chosen")
        # Use entire dataset
        dataset_start_year = int(min_start[4:])
        dataset_end_year = int(max_end[4:])
        year_lst = list(range(dataset_start_year, dataset_end_year + 1))

    # Ask User If They Want To Remove Certain Years From Dataset
    exc_choice = input("\nDo You Want To Exclude Specific Years From The Dataset? (Y/N): ").strip().lower()
    custom_length += 2

    if exc_choice == 'y':
        clear_lines(1)
        exclude_years_str = input("Enter Years To Exclude (comma-separated, e.g. 2005,2010,2012): ").strip()
        try:
            exclude_years = {int(y.strip()) for y in exclude_years_str.split(",") if y.strip()}

        except ValueError:
            print("❌ Invalid Input For Years To Exclude")
            exit()

        # Filter the full list to exclude those years
        year_lst = [y for y in year_lst if y not in exclude_years]
        custom_length += 2
    
    else:
        clear_lines(1)
        print("No Exclusions")
    
    # Define Standard Or Custom Precision
    acc_choice = input("\nCustom Precision? (Y/N): ").strip().lower()
    custom_length += 2

    if acc_choice == 'y':
        custom_length += 2
        lat_precis = int(input("Latitude Precision (e.g. 30): ").strip() or 30)
        lon_precis = int(input("Longitude Precision (e.g. 60): ").strip() or 60)
    
    else:
        clear_lines(1)
        print("Standard Presicion Chosen")
        lat_precis = 30
        lon_precis = 60
    
    # Define Standard Or Custom Zone
    zon_choice = input("\nCustom Analysis Zone? (Y/N): ").strip().lower()
    custom_length += 2

    if zon_choice == 'y':
        custom_length += 4
        lat_min = float(input("Min Latitude [deg]: ").strip() or -90)
        lat_max = float(input("Max Latitude [deg]: ").strip() or 90)
        lon_min = float(input("Min Longitude [deg]: ").strip() or -180)
        lon_max = float(input("Max Longitude [deg]: ").strip() or 180)

        # Convert to radians
        lat_range = (np.radians(lat_min), np.radians(lat_max))
        lon_range = (np.radians(lon_min), np.radians(lon_max))
    
    else:
        clear_lines(1)
        print("Standard Zone Chosen")
        lat_min = -90
        lat_max = 90
        lon_min = -180
        lon_max = 180

        # Convert to radians
        lat_range = (np.radians(lat_min), np.radians(lat_max))
        lon_range = (np.radians(lon_min), np.radians(lon_max))

    # Define Sample_Time
    day = int(user_date[:2])
    month = int(user_date[2:4])
    year = int(user_date[4:])

    chosen_date = date(year, month, day)
    start_date = date(year_lst[0], 1, 1)
    sample_time = (chosen_date - start_date).days

    # Print Settings Overview
    clear_lines(custom_length)
    if chosen_year not in year_lst:
        print(f"⚠️  Warning: Chosen Date Is Not In Dataset\n")

    print(f"Analysis Dataset    ➜  {year_lst}")
    print(f"Chosen Sample Time  ➜  {sample_time} Days")
    print(f"Latitude Precision  ➜  {lat_precis}")
    print(f"Longitude Precision ➜  {lon_precis}")
    print(f"Latitude Range      ➜  {lat_min:.2f}° to {lat_max:.2f}°")
    print(f"Longitude Range     ➜  {lon_min:.2f}° to {lon_max:.2f}°\n")

    # Define LSQR Coefficients
    model_coef, model_coef_fil, SH_arr, SH_arr_fil, cov_SH_arr, cov_fil_SH_arr = LSQR_coefficients(data_year_arr, date_year_arr, year_lst)

    # Ask User If He Wants To Calculate The Uncertainty Or Not
    unc_choice = input("\nInclude Uncertainty Calculations? (⚠️  Warning: Significantly Longer Processing Times) (Y/N): ").strip().lower()

    if unc_choice == 'y':
        calc_uncertainty = True
        clear_lines(1)
        print(f"✅ Uncertainty Calculations Included \n")
    
    else:
        calc_uncertainty = False
        clear_lines(1)
        print(f"✅ Uncertainty Calculations Excluded \n")

    # Define EWH Grid
    earth_grid_EWH, earth_grid_EWH_uncertainty = EWH_grid(SH_arr_fil, cov_fil_SH_arr, year_lst, sample_time, lat_precis=lat_precis, lon_precis=lon_precis, lat_range=lat_range, lon_range=lon_range, calc_uncertainty=calc_uncertainty, file_name='EWH_Grid.xlsx')

    # Render Heatmap
    render_single(earth_grid_EWH, user_date, sample_time, lat_range=(-np.pi/2, np.pi/2), lon_range=(-np.pi, np.pi))

elif mod_choice == '2':
    # Print Confirmation Statement
    print('\n=== Double Date Heatmap ===\n')

elif mod_choice == '3':
    # Print Confirmation Statement
    print('\n=== Difference Heatmap ===\n')

elif mod_choice == '4':
    # Print Confirmation Statement
    print('\n=== Volume Change Calculation ===\n')

elif mod_choice == '5':
    # Print Confirmation Statement
    print('\n=== Model Verification ===\n')

    # All Together &/or Coefficients apart

else:
    # Print Confirmation Statement
    print("\n❌ Invalid Choice")
