'''
DOCSTRING
'''

# External Imports
import os
import sys
import numpy as np

# Internal Imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data.Data_Reader import data_year_arr, date_year_arr
from SH_Functions import cs_spherical_harmonics, cs_render_single, cs_render_double

# Functions
def clear_lines(n):
    for _ in range(n):
        sys.stdout.write("\033[F")  # Move cursor up one line
        sys.stdout.write("\033[K")  # Clear to end of line

# User Selects Analysis Choice
print("""
=== GRACE Data Selection Menu ===
      
1. Render Heatmap Of A Single Date
2. Render Heatmap Of A Two Dates
3. Render Baseline Heatmap
""")

mod_choice = input("Enter Your Choice: ").strip()

clear_lines(8)

if mod_choice == '1':
    # Print Confirmation Statement
    print('\n=== Single Date Heatmap ===\n')

    # Build year-to-date mapping
    min_year = 2002  # Adjust based on your data
    year_to_dates = {}
    for year_idx, year_row in enumerate(date_year_arr):
        year = min_year + year_idx
        dates = [(year_idx, month_idx, date) for month_idx, date in enumerate(year_row) if date != "empty"]
        if dates:
            year_to_dates[year] = dates

    # Print available years in 3-column layout
    available_years = sorted(year_to_dates.keys())
    print("Available GRACE Years:")
    col_width = 8
    cols = 3
    for i in range(0, len(available_years), cols):
        row = available_years[i:i+cols]
        print("".join(f"{str(y):<{col_width}}" for y in row))

    # User inputs year
    user_year_input = input("\nEnter a year: ")
    try:
        selected_year = int(user_year_input)
        if selected_year not in year_to_dates:
            raise ValueError

        # Show available dates in that year
        dates = year_to_dates[selected_year]
        print(f"\nAvailable GRACE Dates in {selected_year}:")
        for i, (_, _, date) in enumerate(dates):
            print(f"{i}: {date}")

        # Track how many lines were printed
        num_date_lines = int(len(dates) + len(available_years)/3 + 8) # +8 for the headers and white lines

        # User selects date
        date_choice = int(input("\nSelect a date by number: "))
        year_idx, month_idx, selected_date = dates[date_choice]

        # Automatically clear printed date lines
        clear_lines(num_date_lines)

        print(f"\n✅ Chosen Date: {selected_date}\n")
        selected_data = data_year_arr[year_idx, month_idx]

    except (ValueError, IndexError):
        print("❌ Invalid year or date selection.")
        exit()

    # Define Standard Or Custom Precision
    acc_choice = input("Custom Precision? (Y/N): ").strip().lower()
    custom_length = 4

    if acc_choice == 'y':
        custom_length += 2
        lat_precis = int(input("Latitude Precision (e.g. 30): ").strip() or 30)
        lon_precis = int(input("Longitude Precision (e.g. 60): ").strip() or 60)
    
    else:
        lat_precis = 30
        lon_precis = 60
    
    # Define Standard Or Custom Zone
    zon_choice = input("\nCustom Analysis Zone? (Y/N): ").strip().lower()

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
        lat_min = -90
        lat_max = 90
        lon_min = -180
        lon_max = 180

        # Convert to radians
        lat_range = (np.radians(lat_min), np.radians(lat_max))
        lon_range = (np.radians(lon_min), np.radians(lon_max))
    
    # Print Settings Overview
    clear_lines(custom_length)

    print(f"\nLatitude Precision  ➜  {lat_precis}")
    print(f"Longitude Precision ➜  {lon_precis}")
    print(f"Latitude Range      ➜  {lat_min:.2f}° to {lat_max:.2f}°")
    print(f"Longitude Range     ➜  {lon_min:.2f}° to {lon_max:.2f}°\n")

    # Define Gravity Values
    earth_grid_pot, earth_grid_acc = cs_spherical_harmonics(selected_data, lat_precis=lat_precis, lon_precis=lon_precis, lat_range=lat_range, lon_range=lon_range, file_name='Single Date Heatmap.xlsx')

    # Render Heatmap
    cs_render_single(earth_grid_acc, selected_date, lat_range=lat_range, lon_range=lon_range)

elif mod_choice == '2':
    # Print Confirmation Statement
    print('\n=== Double Date Heatmap ===\n')

    # Loop twice for two selections
    selected_dates = []
    selected_data = []

    for sel in range(2):
        # Build year-to-date mapping
        min_year = 2002  # Adjust based on your data
        year_to_dates = {}
        for year_idx, year_row in enumerate(date_year_arr):
            year = min_year + year_idx
            dates = [(year_idx, month_idx, date) for month_idx, date in enumerate(year_row) if date != "empty"]
            if dates:
                year_to_dates[year] = dates

        # Print available years in 3-column layout
        available_years = sorted(year_to_dates.keys())
        print("Available GRACE Years:")
        col_width = 8
        cols = 3
        for i in range(0, len(available_years), cols):
            row = available_years[i:i+cols]
            print("".join(f"{str(y):<{col_width}}" for y in row))

        # User inputs year
        user_year_input = input("\nEnter a year: ")
        try:
            selected_year = int(user_year_input)
            if selected_year not in year_to_dates:
                raise ValueError

            # Show available dates in that year
            dates = year_to_dates[selected_year]
            print(f"\nAvailable GRACE Dates in {selected_year}:")
            for i, (_, _, date) in enumerate(dates):
                print(f"{i}: {date}")

            # Track how many lines were printed
            num_date_lines = int(len(dates) + len(available_years)/3 + 8) # +8 for the headers and white lines

            # User selects date
            date_choice = int(input("\nSelect a date by number: "))
            year_idx, month_idx, selected_date = dates[date_choice]

            # Automatically clear printed date lines
            clear_lines(num_date_lines)

            print(f"\n✅ Chosen Date [{sel+1}]: {selected_date}\n")
            data = data_year_arr[year_idx, month_idx]

            # Append Date And Data To Lists
            selected_dates.append(selected_date)
            selected_data.append(data)

        except (ValueError, IndexError):
            print("❌ Invalid year or date selection.")
            exit()
        
    # Define Standard Or Custom Precision
    acc_choice = input("Custom Precision? (Y/N): ").strip().lower()
    custom_length = 4

    if acc_choice == 'y':
        custom_length += 2
        lat_precis = int(input("Latitude Precision (e.g. 30): ").strip() or 30)
        lon_precis = int(input("Longitude Precision (e.g. 60): ").strip() or 60)
    
    else:
        lat_precis = 30
        lon_precis = 60
    
    # Define Standard Or Custom Zone
    zon_choice = input("\nCustom Analysis Zone? (Y/N): ").strip().lower()

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
        lat_min = -90
        lat_max = 90
        lon_min = -180
        lon_max = 180

        # Convert to radians
        lat_range = (np.radians(lat_min), np.radians(lat_max))
        lon_range = (np.radians(lon_min), np.radians(lon_max))
    
    # Print Settings Overview
    clear_lines(custom_length)

    print(f"\nLatitude Precision  ➜  {lat_precis}")
    print(f"Longitude Precision ➜  {lon_precis}")
    print(f"Latitude Range      ➜  {lat_min:.2f}° to {lat_max:.2f}°")
    print(f"Longitude Range     ➜  {lon_min:.2f}° to {lon_max:.2f}°\n")

    # Define Gravity Values
    earth_grid_pot_1, earth_grid_acc_1 = cs_spherical_harmonics(selected_data[0], lat_precis=lat_precis, lon_precis=lon_precis, lat_range=lat_range, lon_range=lon_range, file_name=f'Double Date Heatmap {selected_dates[0]}.xlsx')
    earth_grid_pot_2, earth_grid_acc_2 = cs_spherical_harmonics(selected_data[1], lat_precis=lat_precis, lon_precis=lon_precis, lat_range=lat_range, lon_range=lon_range, file_name=f'Double Date Heatmap {selected_dates[1]}.xlsx')

    # Render Heatmap
    cs_render_double(earth_grid_acc_1, earth_grid_acc_2, selected_dates, lat_range=lat_range, lon_range=lon_range)

elif mod_choice == '3':
    # Print Confirmation Statement
    print('\n=== Baseline Heatmap ===\n')

else:
    # Print Confirmation Statement
    print("\n❌ Invalid Choice")
