"""Data processor module for TEMPO Analyzer."""

import xarray as xr
import numpy as np
from pathlib import Path
import logging

try:
    import pandas as pd
except ImportError:
    pass

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles processing of TEMPO data (averaging, FNR calculation)."""
    
    @staticmethod
    def process_dataset(file_paths: list[Path]) -> xr.Dataset:
        """
        Load multiple NetCDF files, aggregate them, and calculate FNR.
        
        Args:
            file_paths: List of paths to .nc files
            
        Returns:
            Processed xarray Dataset with hourly averages and FNR
        """
        if not file_paths:
            return None
            
        datasets = []
        try:
            # Load all datasets, extracting hour from filename
            for p in file_paths:
                try:
                    ds = xr.open_dataset(p)
                    
                    # Extract hour from filename pattern: tempo_YYYY-MM-DD_HH.nc
                    fname = p.stem  # e.g., tempo_2025-12-15_17
                    parts = fname.split('_')
                    if len(parts) >= 3:
                        hour = int(parts[-1])  # Last part is hour
                        ds = ds.assign_coords(HOUR=hour)
                    
                    datasets.append(ds)
                except Exception as e:
                    logger.warning(f"Failed to open {p}: {e}")
            
            if not datasets:
                return None
                
            # Combine along time dimension
            combined = xr.concat(datasets, dim='TSTEP')
            
            # Calculate hourly means using the HOUR coordinate
            if 'HOUR' in combined.coords:
                ds_avg = combined.groupby('HOUR').mean(dim='TSTEP', skipna=True, keep_attrs=True)
            else:
                # Fallback: just average everything
                ds_avg = combined.mean(dim='TSTEP', skipna=True, keep_attrs=True)
            
            # Calculate FNR (HCHO / NO2)
            # Filter low NO2 to avoid division by zero or noise
            ds_avg['FNR'] = xr.where(
                (ds_avg['NO2_TropVCD'] > 1e-12) & (ds_avg['HCHO_TotVCD'] > -9e30),
                ds_avg['HCHO_TotVCD'] / ds_avg['NO2_TropVCD'],
                np.nan
            )
            
            # Load data into memory so we can close source files
            ds_avg.load()
            
            return ds_avg
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise
        finally:
            # Close all opened datasets
            for ds in datasets:
                try:
                    ds.close()
                except Exception:
                    pass

    @staticmethod
    def save_processed(dataset: xr.Dataset, output_path: Path):
        """Save processed dataset to NetCDF."""
        dataset.to_netcdf(output_path)
