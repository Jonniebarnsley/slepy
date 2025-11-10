# slepy

*/ˈslɛpi/*

A Python library for calculating sea level equivalent (SLE) from ice sheet model output using the methodology from [Goelzer et al. (2020)](https://doi.org/10.5194/tc-14-833-2020).

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Jonniebarnsley/slepy.git slepy-project
cd slepy-project

# 2. (Optional) Customize defaults by editing config.yaml

# 3. Create and activate virtual environment
python -m venv slepy-env
source slepy-env/bin/activate  # On Windows: slepy-env\Scripts\activate

# 4. Install slepy and dependencies
pip install -e .
```

**Note:** After installation, the `slepy` command will be available in your terminal.

## Quick Start

slepy provides tools for calculating sea level equivalent both for a single model run and for model ensembles. It requires netcdf data for ice thickness with dimensions `(x, y, time)` and bed elevation with dimensions `(x, y)` or `(x, y, time)`. The ensemble processor assumes a directory structure similar to:
```
data/
├── run01.nc
├── run02.nc
└── ...
```

### Command Line Interface

```bash
# Basic usage
slepy data/ output.nc

# With basin mask for regional analysis
slepy data/ output.nc --mask basins.nc

# Custom options
slepy --parallel --quiet data/ output.nc
```

### Python API

```python
from slepy import SLECalculator

# Simple calculation on xarray DataArray objects
with SLECalculator() as calc:
    sle = calc.calculate_sle(thickness_da, z_base_da)

# Ensemble processing data directories
with SLECalculator() as calc:
    results = calc.process_ensemble("data/", mask_file="basins.nc")
```

## Advanced Usage

### Custom Variable Names

By default, the library expects specific variable names in your netCDF files. However, you can customize these in `config.yaml` to match your data:

#### Default Variable Names
- `thickness`: Ice thickness
- `Z_base`: Bed elevation 
- `grounded_fraction`: Grounded fraction (0=floating, 1=grounded)
- `basin`: Basin mask for regional analysis

## Requirements

- Python ≥ 3.9
- numpy ≥ 1.21.0
- xarray ≥ 2022.3.0  
- dask[distributed] ≥ 2022.3.0
- netcdf4 ≥ 1.5.0

## References

Goelzer, H., Coulon, V., Pattyn, F., de Boer, B., and van de Wal, R.: Brief communication: On calculating the sea-level contribution in marine ice-sheet models , The Cryosphere, 14, 833–840, https://doi.org/10.5194/tc-14-833-2020, 2020.

## License

MIT License - see LICENSE file for details.
