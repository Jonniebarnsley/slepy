"""
Utility functions for data validation and preprocessing.
"""
import numpy as np
import xarray as xr
from typing import Union
from pathlib import Path
from xarray import DataArray, Dataset
from typing import Set, Literal

from math import pi
from numpy.typing import ArrayLike

from .config import REQUIRED_DIMS


def check_alignment(da1: DataArray, da2: DataArray) -> None:
    """
    Check that two DataArrays have aligned coordinates.
    
    Parameters
    ----------
    da1, da2 : xarray.DataArray
        DataArrays to check for alignment
        
    Raises
    ------
    ValueError
        If DataArrays are not aligned
    """
    # Check dimensions match
    if da1.dims != da2.dims:
        raise ValueError(f"DataArrays have different dimensions: {da1.dims} vs {da2.dims}")
    
    # Check shapes match
    if da1.shape != da2.shape:
        raise ValueError(f"DataArrays have different shapes: {da1.shape} vs {da2.shape}")
    
    # Check coordinates match
    for dim in da1.dims:
        if not da1[dim].equals(da2[dim]):
            raise ValueError(f"DataArrays are not aligned along dimension '{dim}'")


def check_dims(da: DataArray, required_dims: Set[str]) -> None:
    """
    Check that a DataArray has the required dimensions.
    
    Parameters
    ----------
    da : xarray.DataArray
        DataArray to check
    required_dims : set of str
        Set of required dimension names
        
    Raises
    ------
    ValueError
        If required dimensions are missing
    """
    missing_dims = required_dims - set(da.dims)
    if missing_dims:
        raise ValueError(
            f"DataArray missing required dimensions: {missing_dims}. "
            f"Found dimensions: {set(da.dims)}"
        )


def xy2ll(
    x: ArrayLike,
    y: ArrayLike,
    sgn: Literal[1, -1],
    *args,
) -> tuple[ArrayLike, ArrayLike]:
    """
    Courtesy of NSIDC scripts: https://github.com/nsidc/nsidc0756-scripts.git
    Thank you Hannah and Trey.

    Convert x y arrays to lat long arrays.

    Converts Polar  Stereographic (X, Y) coordinates for the polar regions to
    latitude and longitude Stereographic (X, Y) coordinates for the polar
    regions.
    Author: Michael P. Schodlok, December 2003 (map2xy.m)

    Parameters:
        - x: ArrayLike (float scalar or array)
        - y: ArrayLike (float scalar or array)
        - sgn (sign of latitude): integer (1 or -1) inidcating the hemisphere.
              1 : north latitude (default is mer = 45 lat = 70).
              -1 : south latitude (default is mer = 0  lat = 71).
        - *args: optional args. First optional arg is `delta` and second is
           `slat`. Review code for how these are used in practice.
    Returns:
        - (lat, lon)

    Usage:
        [lat, lon] = xy2ll(x, y, sgn)
        [lat, lon] = xy2ll(x, y, sgn, central_meridian, standard_parallel)
    """
    # Get central_meridian and standard_parallel depending on hemisphere
    if len(args) == 2:
        delta = args[0]
        slat = args[1]
    elif len(args) == 0:
        if sgn == 1:
            delta = 45.
            slat = 70.
        elif sgn == -1:
            delta = 0.
            slat = 71.
        else:
            raise ValueError('sgn should be either 1 or -1')
    else:
        raise Exception('bad usage: type "help(xy2ll)" for details')

    # if x, y passed as lists, convert to np.arrays
    if not np.issubdtype(type(x), np.ndarray):
        x = np.array(x)
    if not np.issubdtype(type(y), np.ndarray):
        y = np.array(y)

    # Conversion constant from degrees to radians
    # cde = 57.29577951
    # Radius of the earth in meters
    re = 6378.273 * 10**3
    # Eccentricity of the Hughes ellipsoid squared
    ex2 = .006693883
    # Eccentricity of the Hughes ellipsoid
    ex = np.sqrt(ex2)

    sl = slat * pi / 180.
    rho = np.sqrt(x**2 + y**2)
    cm = np.cos(sl) / np.sqrt(1.0 - ex2 * (np.sin(sl)**2))
    T = np.tan((pi / 4.0) - (sl / 2.0)) / ((1.0 - ex * np.sin(sl)) / (1.0 + ex * np.sin(sl)))**(ex / 2.0)

    if abs(slat - 90.) < 1.e-5:
        T = rho * np.sqrt((1. + ex)**(1. + ex) * (1. - ex)**(1. - ex)) / 2. / re
    else:
        T = rho * T / (re * cm)

    chi = (pi / 2.0) - 2.0 * np.arctan(T)
    lat = chi + ((ex2 / 2.0) + (5.0 * ex2**2.0 / 24.0) + (ex2**3.0 / 12.0)) * np.sin(2 * chi) + ((7.0 * ex2**2.0 / 48.0) + (29.0 * ex2**3 / 240.0)) * np.sin(4.0 * chi) + (7.0 * ex2**3.0 / 120.0) * np.sin(6.0 * chi)

    lat = sgn * lat
    lon = np.arctan2(sgn * x, -sgn * y)
    lon = sgn * lon

    res1 = np.nonzero(rho <= 0.1)[0]
    if len(res1) > 0:
        lat[res1] = pi / 2. * sgn
        lon[res1] = 0.0

    lon = lon * 180. / pi
    lat = lat * 180. / pi
    lon = lon - delta

    return lat, lon
    

def scale_factor(data: Union[DataArray, Dataset], sgn: int) -> DataArray:

    '''
    Calculates the area scale factor for a DataArray on a Polar Stereographic
    grid.

    Inputs:
        - da: DataArray with dimensions [x, y, ...]
        - sgn: integer indicating the hemisphere.
            +1 if North Pole
            -1 if South Pole
    Returns:
        - DataArray for k, the area scale factor (Geolzer et al., 2020)
    '''

    check_dims(data, {'x', 'y'})
    x = data.x
    y = data.y

    # centre origin on the pole if not already
    xs = x - x.mean()
    ys = y - y.mean()
 
    lat, _ = xy2ll(xs, ys, sgn)
    k = 2/(1+np.sin(sgn*lat*2*pi/360))

    return k


def validate_input_data(thickness: DataArray, z_base: DataArray) -> None:
    """
    Validate input data arrays for SLE calculation.
    
    Parameters
    ----------
    thickness : xarray.DataArray
        Ice thickness data
    z_base : xarray.DataArray
        Bed elevation data
        
    Raises
    ------
    ValueError
        If data validation fails
    """
    # Check dimensions
    for da in (thickness, z_base):
        check_dims(da, REQUIRED_DIMS)
    
    # Check alignment
    check_alignment(thickness, z_base)
    
    # Check for reasonable data ranges
    if thickness.min() < 0:
        raise ValueError("Negative thickness values found")
        
    if np.abs(z_base).max() > 10000:  # 10km seems reasonable for bed elevation
        raise ValueError("Extreme bed elevation values found (>10km)")

def validate_and_queue_files(thickness_files: list, z_base_files: list, grounded_fraction_files: list) -> None:
    """Ensures that the number of files for thickness, z_base, and grounded fraction match."""
    if len(thickness_files) != len(z_base_files):
        raise ValueError(
            f"Mismatched number of files: {len(thickness_files)} thickness, "
            f"{len(z_base_files)} z_base files"
        )
        
    if grounded_fraction_files and len(grounded_fraction_files) != len(thickness_files):
        raise ValueError(
            f"Mismatched number of files: {len(thickness_files)} thickness, "
            f"{len(grounded_fraction_files)} grounded fraction files"
            )
    return

def load_areacell(areacell_file: Path) -> DataArray:
    """
    Load grid cell area from netCDF file.
    
    Parameters
    ----------
    areacell_file : str
        Path to netCDF file containing grid cell areas
        
    Returns
    -------
    xarray.DataArray
        Grid cell areas in m²
        
    Raises
    ------
    ValueError
        If areacell file cannot be loaded or has wrong dimensions
    """
    from .config import VARNAMES
    
    if not areacell_file.exists():
        raise FileNotFoundError(f"Areacell file not found: {areacell_file}")
    
    with xr.open_dataset(areacell_file) as ds:
        # First try the configured variable name
        configured_var = VARNAMES["cell_area"]
        if configured_var in ds:
            areacell = ds[configured_var].load()
        else:
            raise KeyError(
                f"Areacell variable '{configured_var}' not found in {areacell_file}. "
                "Consider changing cell_area variable name in slepy config."
            )
        # Check that it has spatial dimensions
        required_spatial_dims = {'x', 'y'}
        if not required_spatial_dims.issubset(set(areacell.dims)):
            raise ValueError(
                f"Areacell must have dimensions {required_spatial_dims}. "
                f"Found: {set(areacell.dims)}"
            )
        return areacell


def load_grounded_fraction(grounded_fraction_file: str) -> DataArray:
    """
    Load grounded fraction from netCDF file.
    
    Parameters
    ----------
    grounded_fraction_file : str
        Path to netCDF file containing grounded fractions
        
    Returns
    -------
    xarray.DataArray
        Grounded fractions (0=floating, 1=grounded)
        
    Raises
    ------
    ValueError
        If grounded fraction file cannot be loaded or has wrong dimensions
    """
    from pathlib import Path
    from .config import VARNAMES
    
    grounded_path = Path(grounded_fraction_file)
    if not grounded_path.exists():
        raise ValueError(f"Grounded fraction file not found: {grounded_fraction_file}")
    
    with xr.open_dataset(grounded_fraction_file) as ds:
        configured_var = VARNAMES["grounded_fraction"]
        if configured_var in ds:
            grounded_fraction = ds[configured_var].load()
        else:
            raise KeyError(
                f"Grounded fraction variable '{configured_var}' not found in {grounded_fraction_file}. "
                "Consider changing grounded_fraction variable name in slepy config."
            )
            
        # Check that it has required dimensions
        required_dims = {'x', 'y', 'time'}
        if not required_dims.issubset(set(grounded_fraction.dims)):
            raise ValueError(
                f"Grounded fraction must have dimensions {required_dims}. "
                f"Found: {set(grounded_fraction.dims)}"
            )
        
        # Validate range (should be between 0 and 1)
        if grounded_fraction.min() < 0 or grounded_fraction.max() > 1:
            raise ValueError(
                f"Grounded fraction values must be between 0 and 1. "
                f"Found range: {grounded_fraction.min():.3f} to {grounded_fraction.max():.3f}"
            )
            
        return grounded_fraction
