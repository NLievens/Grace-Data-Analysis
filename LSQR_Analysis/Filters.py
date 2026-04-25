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
"""

# External Imports
import numpy as np

# Define Filter Functions
def gaussian_filter(clm, slm, smoothing_radius=300):
    """
    Applies a Gaussian smoothing filter to spherical harmonic coefficients
    using the Jekeli (1981) recursive algorithm.

    Unlike a simple exponential approximation, this recursive method accounts 
    for the spherical geometry of the Earth. The filter is isotropic (degree-dependent 
    only), meaning it reduces high-frequency noise equally in all directions 
    without shifting the spatial location of gravity anomalies.

    The recursion is defined by:
    W[0] = 1
    W[1] = [(1 + exp(-2b)) / (1 - exp(-2b))] - [1 / b]
    W[l+1] = -[(2l + 1) / b] * W[l] + W[l-1]
    where b = ln(2) / (1 - cos(r/R)).

    Args:
        clm (np.ndarray): Unfiltered cosine (C) coefficients [lmax+1, lmax+1].
        slm (np.ndarray): Unfiltered sine (S) coefficients [lmax+1, lmax+1].
        smoothing_radius (float): Smoothing radius (half-power) in km. 
            Standard GRACE values typically range from 200 to 500 km.

    Returns:
        tuple: (clm_filtered, slm_filtered) as np.ndarrays.

    Sources:
        - Jekeli, C. (1981). Alternative Methods to Smooth the Earth's Gravity Field. 
          OSU Department of Geodetic Science and Surveying, Report No. 327.
        - Wahr, J., Molenaar, M., & Bryan, F. (1998). Time variability of the Earth's 
          gravity field: Hydrological and oceanic effects and their possible 
          detection using GRACE. JGR: Solid Earth, 103(B12).
    """

    # Define Radius And Extract lmax From Shape
    R_earth = 6378137.0
    lmax = clm.shape[0] - 1
    
    # 1. Calculate Jekeli Gaussian weights recursively
    b = np.log(2) / (1 - np.cos(smoothing_radius / (R_earth / 1000)))
    
    W = np.zeros(lmax + 1)
    W[0] = 1.0
    W[1] = ((1 + np.exp(-2 * b)) / (1 - np.exp(-2 * b))) - (1 / b)
    
    for l in range(1, lmax):
        W[l+1] = -( (2 * l + 1) / b ) * W[l] + W[l-1]
        
    # 2. Vectorized application: Apply W_l along the degree axis (axis 0)
    # np.newaxis allows multiplying (lmax+1,) by (lmax+1, lmax+1)
    clm_filtered = clm * W[:, np.newaxis]
    slm_filtered = slm * W[:, np.newaxis]
    
    return clm_filtered, slm_filtered

def correlated_error_filter(clm_unfiltered, slm_unfiltered, w=10, min_order=8):
    """
    Applies a destriping filter to reduce correlated N-S errors in GRACE coefficients.

    GRACE data often exhibits characteristic N-S "stripes" due to the satellite's 
    polar orbit and resonance errors. This filter removes these by fitting a 
    polynomial to the coefficients across degrees for a specific order and 
    subtracting the fit, processing even and odd degrees separately to preserve 
    the distinct physical signals in those parity groups.

    Args:
        clm_unfiltered (np.ndarray): Unfiltered cosine (C) coefficients [lmax+1, lmax+1].
        slm_unfiltered (np.ndarray): Unfiltered sine (S) coefficients [lmax+1, lmax+1].
        w (int): Window width for the sliding polynomial fit. Default is 10.
        min_order (int): The order (m) at which to start filtering. Orders 
            below this (usually < 8 or 10) are largely stripe-free.

    Returns:
        tuple: (clm_filtered, slm_filtered) as np.ndarrays.

    Notes:
        Standard practice involves using a 3rd or 4th-degree polynomial fit 
        within the sliding window. High-degree fits (e.g., 4) are more aggressive 
        at signal preservation but may leave more residual noise.

    Sources:
        - Swenson, S., & Wahr, J. (2006). Post-processing removal of correlated 
          errors in GRACE data. Geophysical Research Letters, 33(8).
    """

    # Create Copies Of Coefficients And Extract lmax From Shape
    clm_filtered = clm_unfiltered.copy()
    slm_filtered = slm_unfiltered.copy()
    lmax = clm_unfiltered.shape[0] - 1

    for m in range(min_order, lmax + 1):
        # Handle Even And Odd Degrees Separately
        for parity in [0, 1]:
            # Degrees for this order and parity
            degs = np.arange(m, lmax + 1)
            degs = degs[degs % 2 == parity]
            
            if len(degs) < 5: # Need enough points for a stable fit
                continue

            for l in degs:
                # Sliding window indices
                idx = np.where((degs >= l - w) & (degs <= l + w))[0]
                if len(idx) < 4: continue 
                
                win_degs = degs[idx]
                
                # Fit and subtract for C
                poly_c = np.polyfit(win_degs, clm_unfiltered[win_degs, m], 3)
                clm_filtered[l, m] -= np.polyval(poly_c, l)
                
                # Fit and subtract for S
                poly_s = np.polyfit(win_degs, slm_unfiltered[win_degs, m], 3)
                slm_filtered[l, m] -= np.polyval(poly_s, l)

    return clm_filtered, slm_filtered

