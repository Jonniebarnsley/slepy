"""
Handles sea level equivalent calculation and ensemble processing.
"""
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional, Literal, Union
from xarray import DataArray, Dataset

from .config import DENSITIES, OCEAN_AREA, DEFAULT_VARNAMES, DEFAULT_DASK_CONFIG, DEFAULT_CHUNKS

class SLECalculator:
    """
    Calculator for sea level contribution from ice sheet data.
    
    This class implements the methodology from Goelzer et al. (2020) for 
    calculating sea level contribution from ice sheet thickness and bed 
    elevation data.
    
    Parameters
    ----------
    quiet : bool, optional
        Whether to suppress all output and progress bars (default: False)
    parallel : bool, optional
        Whether to use dask for parallel processing (default: False)
    pole : int, optional
        Pole of the polar stereographic grid: 1 for North Pole, -1 for South Pole.
        If not provided, no scale factor is applied (default: None)

        
    References
    ----------
    Goelzer et al. (2020): https://doi.org/10.5194/tc-14-833-2020
    """
    
    def __init__(
        self,
        quiet: bool = False,                    # Suppress output and progress bars
        parallel: bool = False,                 # Use dask for parallel processing
        pole: Optional[Literal[-1, 1]] = None,  # +1 = Arctic, -1 = Antarctic
        ice_density: Optional[float] = None,    # kg/m³
        ocean_density: Optional[float] = None,  # kg/m³
        ocean_area: Optional[float] = None,     # m²
        varnames: Optional[dict] = None,        # variable names in netCDF files
        dask_config: Optional[dict] = None,     # dask cluster config
        chunks: Optional[dict] = None,          # chunk sizes for dask
    ):
        # User options            
        self.quiet = quiet
        self.parallel = parallel
        self.pole = pole

        # Parameters
        self.ice_density = ice_density if ice_density else DENSITIES["ice"]
        self.ocean_density = ocean_density if ocean_density else DENSITIES["ocean"]
        self.freshwater_density = DENSITIES["water"]
        self.ocean_area = ocean_area if ocean_area else OCEAN_AREA

        # Variable names
        self.varnames = DEFAULT_VARNAMES.copy()
        if varnames:
            self.varnames.update(varnames)

        # Dask options
        self.dask_config = DEFAULT_DASK_CONFIG.copy()
        if dask_config:
            self.dask_config.update(dask_config)
        self.chunks = DEFAULT_CHUNKS.copy()
        if chunks:
            self.chunks.update(chunks)
        self._cluster = None
        self._client = None

        # Cache for grid cell area
        self.cell_area = None

        # Validate parameters
        if self.ice_density <= 0:                            
            raise ValueError("Ice density must be positive")
        if self.ocean_density <= 0:
            raise ValueError("Ocean density must be positive")
        if self.freshwater_density <= 0:
            raise ValueError("Water density must be positive")
        if self.ocean_area <= 0:
            raise ValueError("Ocean area must be positive")

        # Set up dask cluster if required
        if self.parallel:
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
        bed_elevation: DataArray,
        grounded_fraction: Optional[DataArray] = None,
        cell_area: Optional[DataArray] = None,
        sum: bool = True,
    ) -> DataArray:
        """
        Calculate sea level contribution from ice thickness and bed elevation.
        
        Parameters
        ----------
        thickness : xarray.DataArray
            Ice sheet thickness with dimensions (x, y, time)
        bed_elevation : xarray.DataArray  
            Bed elevation with dimensions (x, y, time)
        grounded_fraction : xarray.DataArray, optional
            Grounded ice fraction with dimensions (x, y, time). If not provided,
            grounded mask is calculated from floatation criteria.
        cell_area : xarray.DataArray, optional
            Grid cell areas in m². If not provided, calculated from grid spacing.
        sum: bool, optional
            Whether to sum over spatial dimensions (default: True)
            
        Returns
        -------
        xarray.DataArray
            Sea level contribution grid with dimensions (x, y, time)
        """
            
        # If cell area is provided, load into cache. Otherwise, calculate from grid spacing.
        if self.cell_area is not None:
            pass  # already cached
        elif cell_area:
            self.cell_area = cell_area
        else:
            self.cell_area = self._calculate_cell_area(thickness)

        # If grounded fraction not provided, calculate from floatation criteria
        if grounded_fraction is None:
            grounded_fraction = self._create_grounded_mask(thickness, bed_elevation).astype(float)
        
        # Calculate all components
        sle_af = self._calculate_volume_above_floatation(thickness, bed_elevation, grounded_fraction)
        sle_pov = self._calculate_potential_ocean_volume(bed_elevation)
        sle_den = self._calculate_density_correction(thickness)

        # Sum all components together
        sle = sle_af + sle_pov + sle_den
        sle.name = "sle"
        sle.attrs.update({
            "long_name": "Sea level equivalent",
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
    ) -> Dataset:
        """
        Apply calculate_sle to an ensemble of ice sheet model runs.

        Parameters
        ----------
        ensemble_dir : str|Path
            Directory containing netCDF files
        basins_file : Optional[str|Path]
            Basin mask netCDF file for regional SLE timeseries

        Returns
        -------
        xarray.DataArray
            Ensemble sea level contribution timeseries with dimensions (run, time[, basin])
        """

        ensemble_dir = Path(ensemble_dir)
        if not ensemble_dir.is_dir():
            raise ValueError(f"Ensemble directory does not exist: {ensemble_dir}")
        files = sorted(ensemble_dir.glob("*.nc"))
        if len(files) == 0:
            raise ValueError(f"No netCDF files found in ensemble directory: {ensemble_dir}")
        basins = self._load_basins(basins_file) if basins_file else None

        timeseries = []
        if not self.quiet:
            if self.parallel:
                print("Loading ensemble data:")
            else:
                print("Computing sea level equivalent for file:")

        for i, file in enumerate(files, 1):
            if not self.quiet:
                print(f"    Run {i}/{len(files)}: {file.stem}")
            
            thickness, bed_elevation, grounded_fraction = self._load_run_data(file)
            sle_grid = self.calculate_sle(thickness, bed_elevation, grounded_fraction, sum=False)
            del thickness, bed_elevation, grounded_fraction

            # Before summing over spatial dimensions, apply basin mask (if provided)
            if basins is not None:
                ts = sle_grid.groupby(basins).sum()
            else:
                ts = sle_grid.sum(dim=["x", "y"])
            timeseries.append(ts)
            
        # Combine into ensemble with aligned time dimensions
        run_labels = range(1, len(timeseries) + 1)
        ensemble = self._concat_timeseries(timeseries, run_labels)
         
        if not self.parallel:
            return ensemble

        ensemble = ensemble.persist()
        if self.quiet:
            return ensemble.compute()
        
        # If parallel and not quiet, show progress bar
        from dask.distributed import progress
        print("Calculating sea level equivalent...")
        dashboard_link = self._client.dashboard_link
        print(f"📊 Dask dashboard: {dashboard_link}")
        progress(ensemble)
        ensemble = ensemble.compute()
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
            bed_elevation = ds[self.varnames["bed_elevation"]]

            # Load grounded fraction if available
            grounded_fraction = None
            grounded_fraction_name = self.varnames["grounded_fraction"]
            if grounded_fraction_name in ds:
                grounded_fraction = ds[grounded_fraction_name]

            # Load cell area if available and not already cached
            cell_area_name = self.varnames["cell_area"]
            if self.cell_area is None and cell_area_name in ds:
                self.cell_area = ds[cell_area_name]
            
        # Chunk data for parallel processing
        if self.parallel:
            thickness = thickness.chunk(self.chunks)
            bed_elevation = bed_elevation.chunk(self.chunks)
            if grounded_fraction is not None:
                grounded_fraction = grounded_fraction.chunk(self.chunks)

        # Fill NaNs in thickness
        thickness = thickness.fillna(0)

        return thickness, bed_elevation, grounded_fraction

    def _load_basins(self, mask_file: Path) -> DataArray:
        """Load basin mask for regional analysis."""
        with xr.open_dataset(mask_file) as ds:
            return ds[self.varnames["basin"]]

    def _calculate_cell_area(self, data: Union[DataArray, Dataset]) -> Union[DataArray, float]:
        """Get area of each grid cell in m². Assumes even-gridded data. If pole is set, applies
        scale factor for Polar Stereographic projection and returns DataArray. Otherwise, returns
        float."""
        x = data.x
        y = data.y
        
        # Handle single pixel case
        if len(x) < 2:
            dx = 1.0  # Default pixel spacing for single pixel
        else:
            dx = float(x[1] - x[0])  # Assumes regularly spaced grid
        
        if len(y) < 2:
            dy = 1.0
        else:
            dy = float(y[1] - y[0])
        
        # Import here to avoid circular imports
        k = 1   # default scale factor if no pole is specified
        if self.pole:
            from .utils import scale_factor
            k = scale_factor(data, sgn=self.pole)  # sgn=-1 -> South Polar Stereographic
        
        # Grid cell area
        cell_area = dx*dy / k**2
        
        return cell_area
    
    def _create_grounded_mask(
        self,
        thickness: DataArray,
        bed_elevation: DataArray,
        ) -> DataArray:
        """Generate grounded mask based on floatation criteria."""
        return (thickness > -bed_elevation * self.ocean_density / self.ice_density)
        
    def _calculate_volume_above_floatation(
        self, 
        thickness: DataArray, 
        bed_elevation: DataArray, 
        grounded_fraction: Optional[DataArray] = None,
    ) -> DataArray:
        """Calculate sea level contribution from volume above floatation."""
        
        # Get grounded fraction - use provided values or calculate from floatation criteria
        if grounded_fraction is None:
            grounded_fraction = self._create_grounded_mask(thickness, bed_elevation).astype(float)
        
        # Calculate volume above floatation for all points
        # This includes both grounded and floating areas
        v_af = (
            thickness + 
            np.minimum(bed_elevation, 0) * self.ocean_density / self.ice_density
        ) * self.cell_area * grounded_fraction
        
        # Sea level contribution (relative to first time step)
        sle_af = -(v_af - v_af.isel(time=0)) * (self.ice_density / self.ocean_density) / self.ocean_area
        
        return sle_af

    def _calculate_potential_ocean_volume(self, bed_elevation: DataArray) -> DataArray:
        """Calculate sea level contribution from potential ocean volume."""
        # If bed elevation has no time dimension, POV contribution is zero
        if 'time' not in bed_elevation.dims:
            return 0
        
        v_pov = (-bed_elevation).clip(min=0) * self.cell_area
        sle_pov = -(v_pov - v_pov.isel(time=0)) / self.ocean_area
        
        return sle_pov
        
    def _calculate_density_correction(self, thickness: DataArray) -> DataArray:
        """Calculate density correction."""
        v_den = thickness * (
            self.ice_density / self.freshwater_density - self.ice_density / self.ocean_density
        ) * self.cell_area
        
        sle_den = -(v_den - v_den.isel(time=0)) / self.ocean_area
        
        return sle_den