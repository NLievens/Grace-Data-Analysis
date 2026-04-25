"""
GRACE Filtering Utilities

This module provides functions to apply commonly used filters on GRACE and GRACE-FO 
spherical harmonic coefficients to improve data quality by reducing noise and 
correlated errors.

Functions:
----------
- `gaussian_filter`: Applies a Gaussian spatial filter to smooth high-frequency noise.
- `correlated_error_filter`: Applies a destriping filter to reduce correlated north-south 
  striping errors, based on the method of Swenson and Wahr (2006).

Dependencies:
-------------
- numpy

Usage:
------
These functions take unfiltered spherical harmonic coefficient arrays (`clm`, `slm`) 
as input and return filtered arrays of the same shape (lmax+1, lmax+1).

Typical preprocessing includes:
- Gaussian smoothing using a defined radius in kilometers.
- Destriping by fitting quadratic polynomials across degrees for each order ≥ min_order.

References:
-----------
Swenson, S., & Wahr, J. (2006). Post-processing removal of correlated errors in GRACE 
data. *Geophysical Research Letters*, 33(8).
"""

# External Imports
import numpy as np

# Define Filter Functions
def gaussian_filter(clm_unfiltered, slm_unfiltered, smoothing_radius=300):
    """
    Applies a Gaussian smoothing filter to spherical harmonic coefficients.

    This function filters degree-dependent coefficients using a Gaussian kernel
    defined by a smoothing radius (in km). The result is a spatially smoothed
    gravity field representation, reducing high-frequency noise.

    Args:
        clm_unfiltered (np.ndarray): Unfiltered cosine (C) spherical harmonic coefficients of shape (lmax+1, lmax+1).
        slm_unfiltered (np.ndarray): Unfiltered sine (S) spherical harmonic coefficients of the same shape.
        smoothing_radius (float, optional): Smoothing radius in kilometers. Higher values apply stronger smoothing. Default is 300 km.

    Returns:
        clm_filtered (np.ndarray): Filtered cosine (C) coefficients of the same shape.
        slm_filtered (np.ndarray): Filtered sine (S) coefficients of the same shape.
    """

    # Define Constant
    R_earth = 6378.13630000  # Radius of the Earth in km

    # Get lmax from shape
    lmax = clm_unfiltered.shape[0] - 1

    # Calculate theta (in radians)
    theta = smoothing_radius / R_earth

    # Compute Gaussian weights W_l
    l = np.arange(lmax + 1)
    W_l = np.exp(-0.5 * l * (l + 1) * theta**2)

    # Prepare filtered arrays
    clm_filtered = np.zeros_like(clm_unfiltered)
    slm_filtered = np.zeros_like(slm_unfiltered)

    # Apply Gaussian filter to both datasets
    for l_idx in range(lmax + 1):
        for m_idx in range(l_idx + 1):
            clm_filtered[l_idx, m_idx] = W_l[l_idx] * clm_unfiltered[l_idx, m_idx]
            slm_filtered[l_idx, m_idx] = W_l[l_idx] * slm_unfiltered[l_idx, m_idx]
    
    return clm_filtered, slm_filtered

def correlated_error_filter(clm_unfiltered, slm_unfiltered, lmax, w=10, min_order=8):
    """
    Applies a correlated-error (destriping) filter to GRACE spherical harmonic coefficients.

    This filter reduces north-south striping artifacts in GRACE data by fitting and subtracting 
    quadratic polynomials from even and odd degree coefficients separately, as described in 
    Swenson and Wahr (2006). The filter is applied to orders ≥ `min_order` using a moving 
    window of width `w`.

    Args:
        clm_unfiltered (np.ndarray): Unfiltered cosine (C) spherical harmonic coefficients of shape (lmax+1, lmax+1).
        slm_unfiltered (np.ndarray): Unfiltered sine (S) spherical harmonic coefficients of the same shape.
        lmax (int): Maximum spherical harmonic degree to be filtered.
        w (int, optional): Window width (in degrees) for the moving average fit. Default is 10.
        min_order (int, optional): Minimum spherical harmonic order at which filtering begins. Default is 8.

    Returns:
        clm_filtered (np.ndarray): Filtered cosine (C) coefficients with destriping applied.
        slm_filtered (np.ndarray): Filtered sine (S) coefficients with destriping applied.

    Notes:
        The filter separates degrees into even and odd sequences for each order m ≥ `min_order`,
        fits a quadratic polynomial to a window around each degree, and subtracts the fitted
        value to suppress correlated errors (striping).
    """
    
    # Create Copy Of Original Data For Filter Application
    clm_filtered = clm_unfiltered.copy()
    slm_filtered = slm_unfiltered.copy()

    for m in range(min_order, lmax + 1):  # Start at m=8 as per the paper
        # Separate even and odd degrees
        even_degrees = np.arange(m % 2, lmax + 1, 2)  # Degrees with same parity as m
        odd_degrees = np.arange((m % 2) ^ 1, lmax + 1, 2)  # Degrees with opposite parity

        # Process even degrees for C_lm
        for l in even_degrees:
            # Define the window range
            l_start = max(m, l - w // 2)
            l_end = min(lmax, l + w // 2)
            window_degrees = np.arange(l_start, l_end + 1)
            window_degrees = window_degrees[window_degrees % 2 == l % 2]  # Same parity as l

            if len(window_degrees) < 3:  # Need at least 3 points for quadratic fit
                continue

            # Extract coefficients in the window
            window_clm = clm_unfiltered[window_degrees, m]
            # Fit a quadratic polynomial
            coeffs = np.polyfit(window_degrees, window_clm, 2)
            # Compute the smoothed value at degree l
            smoothed = np.polyval(coeffs, l)
            # Subtract to get the destriped coefficient
            clm_filtered[l, m] = clm_unfiltered[l, m] - smoothed

        # Process odd degrees for C_lm
        for l in odd_degrees:
            l_start = max(m, l - w // 2)
            l_end = min(lmax, l + w // 2)
            window_degrees = np.arange(l_start, l_end + 1)
            window_degrees = window_degrees[window_degrees % 2 == l % 2]

            if len(window_degrees) < 3:
                continue

            window_clm = clm_unfiltered[window_degrees, m]
            coeffs = np.polyfit(window_degrees, window_clm, 2)
            smoothed = np.polyval(coeffs, l)
            clm_filtered[l, m] = clm_unfiltered[l, m] - smoothed

        # Process even degrees for S_lm
        for l in even_degrees:
            l_start = max(m, l - w // 2)
            l_end = min(lmax, l + w // 2)
            window_degrees = np.arange(l_start, l_end + 1)
            window_degrees = window_degrees[window_degrees % 2 == l % 2]

            if len(window_degrees) < 3:
                continue

            window_slm = slm_unfiltered[window_degrees, m]
            coeffs = np.polyfit(window_degrees, window_slm, 2)
            smoothed = np.polyval(coeffs, l)
            slm_filtered[l, m] = slm_unfiltered[l, m] - smoothed

        # Process odd degrees for S_lm
        for l in odd_degrees:
            l_start = max(m, l - w // 2)
            l_end = min(lmax, l + w // 2)
            window_degrees = np.arange(l_start, l_end + 1)
            window_degrees = window_degrees[window_degrees % 2 == l % 2]

            if len(window_degrees) < 3:
                continue

            window_slm = slm_unfiltered[window_degrees, m]
            coeffs = np.polyfit(window_degrees, window_slm, 2)
            smoothed = np.polyval(coeffs, l)
            slm_filtered[l, m] = slm_unfiltered[l, m] - smoothed

    return clm_filtered, slm_filtered
