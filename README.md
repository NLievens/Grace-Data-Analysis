# GRACE & GRACE-FO Gravity Data Processing

## Overview
This repository contains GRACE and GRACE-FO satellite data alongside Python scripts to process Level 2 data, generate gravity anomaly maps, and analyze temporal trends in Earth's gravity field.

## Contents
- **Data** — Contains raw and parsed GRACE and GRACE-FO datasets from 2002 through August 2025, along with a script for reading the data.
- **Gravity_Maps** — Contains the main script and supporting functions used to analyze pre-processed gravity-related data, convert it into meaningful gravity values, and render high-resolution heatmaps of Earth’s gravity field.


## Features
- Processing of Level 2 gravity data
- Generation of spatial gravity anomaly maps
- Temporal trend analysis of gravity changes

## Requirements
- Python 3.8+
- numpy
- scipy
- pandas
- cartopy
- openpyxl
- geopandas
- matplotlib
- collections

Install dependencies via:
```bash
pip install -r requirements.txt
