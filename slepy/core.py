"""
Handles sea level equivalent calculation and ensemble processing.
"""
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional, Literal, Union
from xarray import DataArray, Dataset
from dask.distributed import progress

from .config import DENSITIES, OCEAN_AREA, VARNAMES, DASK_CONFIG, CHUNKS
from .utils import validate_input_data


class SLECalculator:
    """
    Calculator for sea level contribution from ice sheet data.
    
    This class implements the methodology from Goelzer et al. (2020) for 
    calculating sea level contribution from ice sheet thickness and bed 
    elevation data.
    
    Parameters
    ----------
    rho_ice : float, optional
        Density of ice in kg/m³
    rho_ocean : float, optional  
        Density of seawater in kg/m³
    rho_water : float, optional
        Density of freshwater in kg/m³
    ocean_area : float, optional
        Surface area of the ocean in m²
    quiet : bool, optional
        Whether to suppress all output and progress bars (default: False)
    areacell : xarray.DataArray, optional
        Pre-calculated grid cell areas in m². If provided, bypasses automatic area calculation
    dask_config : dict, optional
        Dask configuration parameters
        
    References
    ----------
    Goelzer et al. (2020): https://doi.org/10.5194/tc-14-833-2020
    """
    
    def __init__(
        self,
        pole: Optional[Literal[-1, 1]] = None,  # +1 = Arctic, -1 = Antarctic
        quiet: bool = False,                    # Suppress output and progress bars
        parallel: bool = True
    ):
        self.areacell = None   # Cache for grid cell area
        self.pole = pole            
        self.quiet = quiet
        self.parallel = parallel

        # Get parameters from config
        self.rho_ice = DENSITIES["ice"]
        self.rho_ocean = DENSITIES["ocean"] 
        self.rho_water = DENSITIES["water"]
        self.ocean_area = OCEAN_AREA
        self.varnames = VARNAMES
        self.dask_config = DASK_CONFIG
        self.chunks = CHUNKS

        # Dask cluster and client
        self._cluster = None
        self._client = None
        
        # Validate parameters
        if self.rho_ice <= 0:                            
            raise ValueError("Ice density must be positive")
        if self.rho_ocean <= 0:
            raise ValueError("Ocean density must be positive")
        if self.rho_water <= 0:
            raise ValueError("Water density must be positive")
        if self.ocean_area <= 0:
            raise ValueError("Ocean area must be positive")
            
        # Always set up dask cluster
        self._setup_dask_cluster()
            
    def _setup_dask_cluster(self):
        """Set up dask cluster for parallel computation with memory limits."""
        try:
            from dask.distributed import get_client
            self._client = get_client()
        except ValueError:
            from dask.distributed import Client, LocalCluster
            self._cluster = LocalCluster(**self.dask_config)
            self._client = Client(self._cluster)
            
    def close(self):
        """Clean up dask cluster resources."""
        if self._client is not None:
            self._client.close()
            self._client = None
            
        if self._cluster is not None:
            self._cluster.close()
            self._cluster = None

    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up dask resources."""
        self.close()
        
    def calculate_sle(
        self, 
        thickness: DataArray, 
        z_base: DataArray,
        grounded_fraction: Optional[DataArray] = None,
        cell_area: Optional[DataArray] = None,
        sum: bool = True,   # Sum over spatial dimensions
    ) -> DataArray:
        """
        Calculate sea level contribution from ice thickness and bed elevation.
        
        Parameters
        ----------
        thickness : xarray.DataArray
            Ice sheet thickness with dimensions (x, y, time)
        z_base : xarray.DataArray  
            Bed elevation with dimensions (x, y, time)
        grounded_fraction : xarray.DataArray, optional
            Pre-calculated grounded fractions (0=floating, 1=grounded). If provided, 
            bypasses automatic floatation criteria calculation. Values should be between 0 and 1.
        cell_area : xarray.DataArray, optional
            Grid cell areas in m². If provided, bypasses automatic area calculation.
        sum: bool, optional
            Whether to sum over spatial dimensions (default: True)
            
        Returns
        -------
        xarray.DataArray
            Sea level contribution grid with dimensions (x, y, time)
        """
        # Validate input data
        validate_input_data(thickness, z_base)
            
        # Fill NaNs in thickness
        thickness = thickness.fillna(0)
        
        # If cell area is provided, load into SLECalculator cache. Otherwise, calculate from grid spacing.
        if self.areacell:
            pass  # already cached
        elif cell_area:
            self.areacell = cell_area
        else:
            self.areacell = self._calculate_areacell(thickness)

        # Get grounded fraction - use provided values or calculate from floatation criteria
        if grounded_fraction is None:
            grounded_fraction = self._calculate_grounded_mask(thickness, z_base).astype(float)
        
        # Calculate all components lazily
        sle_af = self._calculate_volume_above_floatation(thickness, z_base, grounded_fraction)
        del grounded_fraction
        sle_pov = self._calculate_potential_ocean_volume(z_base)
        del z_base
        sle_den = self._calculate_density_correction(thickness)
        del thickness

        # Sum all components together
        sle = sle_af + sle_pov + sle_den
        sle.name = "sle"
        sle.attrs.update({
            "long_name": "Sea level contribution",
            "units": "m",
            "methodology": "Goelzer et al. (2020)",
        })
        
        if sum:
            return sle.sum(dim=["x", "y"])  # Sum over spatial dimensions
        return sle

    def process_ensemble(
            self,
            ensemble_dir: Union[str,Path],
            basins_file: Optional[Union[str,Path]] = None,
    ):
        """
        Apply calculate_sle to an ensemble of ice sheet model runs.

        Parameters
        ----------
        ensemnble_dir : str|Path
            Directory containing netCDF files
        basins_file : Optional[str|Path]
            Basin mask netCDF file for regional SLE timeseries

        Returns
        -------
        xarray.DataArray
            Ensemble sea level contribution timeseries with dimensions (run, time[, basin])
        """

        if not self._client:
            raise RuntimeError("Dask client not initialized. Call `SLECalculator()` first.")

        # Validate ensemble directory and get list of files
        ensemble_dir = Path(ensemble_dir)
        if not ensemble_dir.is_dir():
            raise ValueError(f"Ensemble directory does not exist: {ensemble_dir}")
        files = sorted(ensemble_dir.glob("*.nc"))

        # load basin mask into memory
        basins = self._load_basins(basins_file) if basins_file else None

        # Load cell area into cache
        if self.areacell is None:
            first_file = files[0]
            with xr.open_dataset(first_file, engine="netcdf4") as ds:
                thickness = ds[self.varnames["thickness"]]
            self.areacell = self._calculate_areacell(thickness)
            del thickness

        timeseries = []
        print("Loading ensemble data:")
        for i, file in enumerate(files, 1):

            # Print progress if not quiet
            if not self.quiet:
                print(f"    Run {i}/{len(files)}: {file.stem}")
            
            thickness, bed_elevation, grounded_fraction = self._load_run_data(file)
            # Calculate sea level contribution lazily
            sle_grid = self.calculate_sle(thickness, bed_elevation, grounded_fraction, sum=False)
            del thickness, bed_elevation, grounded_fraction

            # Before summing over spatial dimensions, apply basin mask (if provided)
            if basins is not None:
                ts = sle_grid.groupby(basins).sum(dim=["x", "y"])
            else:
                ts = sle_grid.sum(dim=["x", "y"])
            del sle_grid
            timeseries.append(ts)
            
        # Combine into ensemble with aligned time dimensions
        run_labels = range(1, len(timeseries) + 1)
        ensemble = self._concat_timeseries(timeseries, run_labels)
        
        if not self.quiet:
            print("Calculating sea level equivalent...")
            dashboard_link = self._client.dashboard_link
            print(f"📊 Dask dashboard: {dashboard_link}")
            ensemble = ensemble.persist()
            progress(ensemble)
        ensemble = ensemble.compute()

        # Print final status
        if not self.quiet:
            print(f"Completed processing {len(timeseries)} ensemble runs")

        return ensemble
    
    def _concat_timeseries(self, timeseries, run_labels):
        """
        Align time dimensions and concatenate timeseries with different time lengths.
        
        This method creates a union of all time coordinates and reindexes each
        timeseries to this common time grid, filling missing values with NaN.
        
        Parameters
        ----------
        timeseries_list : list of xarray.DataArray
            List of timeseries DataArrays with potentially different time dimensions
        run_labels : range or list
            Labels for the run dimension
            
        Returns
        -------
        xarray.DataArray
            Concatenated timeseries with aligned time dimensions
        """
            
        # Get all unique time coordinates
        times = np.array([])
        for ts in timeseries:
            times = np.append(times, ts.time.values)
        unique_times = sorted(np.unique(times))
        
        # Reindex each timeseries to the union time grid
        aligned_timeseries = []
        for ts in timeseries:
            # Reindex to union time coordinates, filling missing values with NaN
            aligned_ts = ts.reindex(time=unique_times, fill_value=np.nan)
            aligned_timeseries.append(aligned_ts)
        
        # Now concatenate along run dimension
        ensemble = xr.concat(aligned_timeseries, dim="run")
        ensemble = ensemble.assign_coords(run=run_labels)

        return ensemble

    def _load_run_data(
            self,
            file: Union[str,Path],
    ):
        """
        Loads thickness, bed elevation, and optionally grounded fraction data for a single run,
        using specified variable names and chunking from config.
        """
        
        # Open file and load variables
        with xr.open_dataset(file, engine="netcdf4") as ds:
            thickness = ds[self.varnames["thickness"]]
            z_base = ds[self.varnames["bed_elevation"]]

            # Load grounded fraction if available
            grounded_fraction = None
            grounded_fraction_name = self.varnames["grounded_fraction"]
            if grounded_fraction_name in ds:
                grounded_fraction = ds[grounded_fraction_name]
            
        # Chunk data for parallel processing
        thickness = thickness.chunk(self.chunks)
        z_base = z_base.chunk(self.chunks)
        if grounded_fraction is not None:
            grounded_fraction = grounded_fraction.chunk(self.chunks)

        return thickness, z_base, grounded_fraction

    def _load_basins(self, mask_file: Path) -> DataArray:
        """Load basin mask for regional analysis."""
        with xr.open_dataset(mask_file) as ds:
            return ds[self.varnames["basin"]]

    def _calculate_areacell(self, data: Union[DataArray, Dataset]) -> Union[DataArray, float]:
        """Get area of each grid cell in m². Assumes even-gridded data. If pole is set, applies
        scale factor for Polar Stereographic projection and returns DataArray. Otherwise, returns
        float."""
        x = data.x
        
        # Handle single pixel case
        if len(x) < 2:
            dx = 1.0  # Default pixel spacing for single pixel
        else:
            dx = float(x[1] - x[0])  # Assumes regularly spaced grid
        
        # Import here to avoid circular imports
        k = 1   # default scale factor if no pole is specified
        if self.pole:
            from .utils import scale_factor
            k = scale_factor(data, sgn=self.pole)  # sgn=-1 -> South Polar Stereographic
        
        # Grid cell area
        areacell = dx**2 / k**2
        
        return areacell
    
    def _calculate_grounded_mask(
        self,
        thickness: DataArray,
        z_base: DataArray,
        ) -> DataArray:
        """Generate grounded mask based on floatation criteria."""
        return (thickness > -z_base * self.rho_ocean / self.rho_ice)
        
    def _calculate_volume_above_floatation(
        self, 
        thickness: DataArray, 
        z_base: DataArray, 
        grounded_fraction: Optional[DataArray] = None,
    ) -> DataArray:
        """Calculate sea level contribution from volume above floatation."""
        
        # Get grounded fraction - use provided values or calculate from floatation criteria
        if grounded_fraction is None:
            grounded_fraction = self._calculate_grounded_mask(thickness, z_base).astype(float)
        
        # Calculate volume above floatation for all points
        # This includes both grounded and floating areas
        v_af = (
            thickness + 
            np.minimum(z_base, 0) * self.rho_ocean / self.rho_ice
        ) * self.areacell * grounded_fraction
        
        # Sea level contribution (relative to first time step)
        sle_af = -(v_af - v_af.isel(time=0)) * (self.rho_ice / self.rho_ocean) / self.ocean_area
        
        return sle_af

    def _calculate_potential_ocean_volume(
        self, 
        z_base: DataArray, 
    ) -> DataArray:
        """Calculate sea level contribution from potential ocean volume."""
        v_pov = (-z_base).clip(min=0) * self.areacell
        sle_pov = -(v_pov - v_pov.isel(time=0)) / self.ocean_area
        
        return sle_pov
        
    def _calculate_density_correction(
        self, 
        thickness: DataArray, 
    ) -> DataArray:
        """Calculate density correction."""
        v_den = thickness * (
            self.rho_ice / self.rho_water - self.rho_ice / self.rho_ocean
        ) * self.areacell
        
        sle_den = -(v_den - v_den.isel(time=0)) / self.ocean_area
        
        return sle_den