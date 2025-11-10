"""
Command-line interface for slepy.
"""
import sys
import argparse
from pathlib import Path
from typing import Literal
from xarray import DataArray

from .core import SLECalculator


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Calculate sea level contribution from ice sheet model ensemble",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  slepy thickness/ z_base/ output.nc
  
  # With basin mask
  slepy thickness/ z_base/ output.nc --mask basins.nc
  
  # With custom area file
  slepy thickness/ z_base/ output.nc --areacell areas.nc
  
  # With grounded fraction directory
  slepy thickness/ z_base/ output.nc --grounded-fraction-dir grounded_fraction/
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "ensemble_dir",
        type=Path,
        help="Directory containing netCDF files"
    )
    parser.add_argument(
        "output_file",
        type=Path, 
        help="Output netCDF file path"
    )
    
    # Optional arguments
    parser.add_argument(
        "-M", "--mask",
        type=Path,
        help="Basin mask netCDF file for regional analysis"
    )
    parser.add_argument(
        "--pole",
        type=int,
        choices=[-1, 1],
        default=None,
        help="Pole of the polar stereographic grid: 1 for North Pole, -1 for South Pole"
    )
    parser.add_argument(
        "-o", "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists"
    )
    parser.add_argument(
        "-p", "--parallel",
        action="store_true",
        help="Use dask for parallel processing"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all output and progress bars"
    )
    return parser

def validate_arg_paths(args) -> None:
    """Validate that provided paths exist and are correct."""
    if not args.ensemble_dir.exists():
        raise FileNotFoundError(f"Directory not found: {args.ensemble_dir}")

    if args.output_file.suffix != ".nc":
        raise ValueError("Output file must have .nc extension")
    
    if args.output_file.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_file.name} already exists. Use --overwrite to overwrite.")
    
    if args.mask and not args.mask.exists():
        raise FileNotFoundError(f"Mask file not found: {args.mask}")

def save_results(data: DataArray, output_file: Path, overwrite: bool = False) -> None:
    """
    Save results to netCDF file.
    
    Parameters
    ----------
    data : xarray.DataArray
        Data to save
    output_file : Path or str
        Output file path
    overwrite : bool, optional
        Whether to overwrite existing files
    """
    
    if output_file.suffix != ".nc":
        raise ValueError("Output file must have .nc extension")
        
    if output_file.exists() and not overwrite:
        raise FileExistsError(
            f"{output_file.name} already exists. Use overwrite=True to overwrite."
        )
        
    if overwrite and output_file.exists():
        output_file.unlink()
        
    # Convert to dataset and save
    ds = data.to_dataset(name="sle")
    ds.attrs.update({
        "title": "Sea level contribution",
        "methodology": "Goelzer et al. (2020)",
        "software": "slepy",
    })
    
    ds.to_netcdf(output_file)

def main(args=None):
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(args)
    validate_arg_paths(args)


    # Initialize calculator
    calc = SLECalculator(pole=args.pole, parallel=args.parallel, quiet=args.quiet)

    # Print inputs summary   
    if not args.quiet:
        print("Processing ensemble...")
        print(f"Ensemble dir: {args.ensemble_dir}")
        if args.mask:
            print(f"Basin mask: {args.mask}")
        if args.pole:
            print(f"Polar Stereographic grid: {'North' if args.pole == 1 else 'South'}")
        print(f"Output: {args.output_file}")

    # Calculate SLE 
    sle = calc.process_ensemble(
        ensemble_dir=args.ensemble_dir,
        basins_file=args.mask,
    )
    calc.close() # clean up SLECalculator resources

    # Save to file
    save_results(sle, args.output_file, overwrite=args.overwrite)
    
    # Print results summary
    if not args.quiet:
        print(f"✓ Results saved to {args.output_file}")
        print(f"  Ensemble size: {sle.sizes['run']} runs")
        if 'basin' in sle.dims:
            print(f"  Basins: {sle.sizes['basin']} basins")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
