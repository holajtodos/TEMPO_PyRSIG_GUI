"""RSIG Downloader module for TEMPO data with parallel downloading."""

import asyncio
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging
import time
import gc
import traceback

try:
    from pyrsig import RsigApi
except ImportError:
    RsigApi = None

from .status import StatusManager
from .constants import DEFAULT_BBOX

logger = logging.getLogger(__name__)

# Configure parallel download settings
MAX_CONCURRENT_DOWNLOADS = 4  # Number of parallel downloads
DOWNLOAD_TIMEOUT = 180.0      # Timeout per granule in seconds


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format.
    
    Returns:
        < 60s: "45s"
        1m - 60m: "2m 30s"
        > 1h: "1h 15m"
    """
    if seconds < 0:
        return "calculating..."
    
    seconds = int(seconds)
    
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m {s}s" if s else f"{m}m"
    else:
        h, remainder = divmod(seconds, 3600)
        m = remainder // 60
        return f"{h}h {m}m" if m else f"{h}h"





class RSIGDownloader:
    """Handles downloading TEMPO data from EPA RSIG API with parallel execution."""
    
    def __init__(self, workdir: Path, max_concurrent: int = MAX_CONCURRENT_DOWNLOADS, api_key: str = ""):
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.max_concurrent = max_concurrent
        self.api_key = api_key  # Configured API key (empty = anonymous)
        
        # Shared state for progress tracking
        self._completed = 0
        self._total = 0
        self._lock = asyncio.Lock()
        
    async def download_granules(self, 
                              dates: list[str], 
                              hours: list[int], 
                              bbox: list[float], 
                              dataset_name: str,
                              max_cloud: float = 0.5, 
                              max_sza: float = 70.0,
                              status: StatusManager = None) -> list[Path]:
        """
        Download TEMPO granules using daily batch requests with controlled parallelism.
        
        This approach uses daily batches (instead of per-hour) to reduce server load,
        while still allowing parallel processing of multiple days for speed.
        
        Args:
            dates: List of date strings (YYYY-MM-DD)
            hours: List of UTC hours (0-23)
            bbox: [west, south, east, north]
            dataset_name: Name of the dataset (for file naming)
            max_cloud: Maximum cloud fraction (0-1)
            max_sza: Maximum solar zenith angle (deg)
            status: StatusManager for UI updates
        
        Returns:
            List of paths to downloaded .nc files
        """
        if RsigApi is None:
            if status: status.emit("error", "pyrsig library not installed!")
            return await self._simulate_download(dates, hours, dataset_name, status)
        
        dataset_dir = self.workdir
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine hour range for daily requests
        min_hour = min(hours)
        max_hour = max(hours)
        
        self._total = len(dates)  # Counting days
        self._completed = 0
        
        if status:
            status.emit("info", f"🚀 Daily batch download: {self._total} days × {len(hours)} hours, {self.max_concurrent} workers")
        
        logger.info(f"[BATCH] Downloading {self._total} days with {self.max_concurrent} parallel workers")
        
        
        # Use configured API key or anonymous
        api_key = self.api_key if self.api_key else "anonymous"
        
        start_time = time.time()
        
        # Create semaphore for controlled parallelism
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def download_day_worker(d_str: str, worker_id: int) -> list[Path]:
            """Worker that downloads one day with its own API session."""
            async with semaphore:
                import tempfile
                import shutil
                
                # Each worker gets its own temp directory and API session
                temp_dir = tempfile.mkdtemp(prefix=f"rsig_w{worker_id}_")
                
                try:
                    # Create API session for this worker
                    api = RsigApi(bbox=bbox, workdir=temp_dir, grid_kw='1US1', gridfit=True)
                    api.tempo_kw.update({
                        'minimum_quality': 'normal',
                        'maximum_cloud_fraction': max_cloud,
                        'maximum_solar_zenith_angle': max_sza,
                        'api_key': api_key
                    })
                    
                    if status:
                        async with self._lock:
                            progress = self._completed / self._total
                        status.emit("download", f"⬇️ W{worker_id}: {d_str}", progress)
                    
                    # Download entire day in one request
                    result = await self._download_daily_batch(
                        api, d_str, min_hour, max_hour, hours, dataset_dir, status
                    )
                    
                    async with self._lock:
                        self._completed += 1
                    
                    return result if result else []
                    
                except Exception as e:
                    logger.error(f"[BATCH] Worker {worker_id} failed for {d_str}: {e}")
                    if status:
                        status.emit("error", f"❌ W{worker_id}: {d_str} - {e}")
                    async with self._lock:
                        self._completed += 1
                    return []
                    
                finally:
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except:
                        pass
        
        # Launch all day downloads in parallel (semaphore controls actual concurrency)
        tasks = [download_day_worker(d_str, i % self.max_concurrent + 1) for i, d_str in enumerate(dates)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        saved_files = []
        errors = 0
        for result in results:
            if isinstance(result, Exception):
                errors += 1
                logger.error(f"[BATCH] Task exception: {result}")
            elif result:
                saved_files.extend(result)
        
        elapsed = time.time() - start_time
        
        logger.info(f"[BATCH] Complete: {len(saved_files)} files, {errors} errors in {elapsed:.1f}s")
        if status:
            status.emit("ok", f"✅ Downloaded {len(saved_files)} granules from {self._total} days in {format_duration(elapsed)}")
        
        return saved_files
    
    async def _download_daily_batch(
        self,
        api: 'RsigApi',
        d_str: str,
        min_hour: int,
        max_hour: int,
        hours: list[int],
        dataset_dir: Path,
        status: StatusManager
    ) -> list[Path]:
        """Download an entire day's worth of data in one request.
        
        Uses a persistent API session to maintain connection and reduce overhead.
        Splits the returned data into per-hour files for consistent processing.
        """
        logger.info(f"[BATCH] Fetching day {d_str} (hours {min_hour:02d}-{max_hour:02d})")
        
        def _fetch_day():
            """Synchronous fetch for entire day range."""
            d_obj = pd.to_datetime(d_str)
            bdate = d_obj + pd.to_timedelta(min_hour, unit='h')
            edate = d_obj + pd.to_timedelta(max_hour, unit='h') + pd.to_timedelta('59m')
            
            logger.info(f"[BATCH] Requesting NO2: {bdate} to {edate}")
            if status: status.emit("info", f"Requesting NO2: {d_str}")

            no2ds = api.to_ioapi('tempo.l2.no2.vertical_column_troposphere', bdate=bdate, edate=edate)
            
            logger.info(f"[BATCH] Requesting HCHO: {bdate} to {edate}")
            if status: status.emit("info", f"Requesting HCHO: {d_str}")

            hchods = api.to_ioapi('tempo.l2.hcho.vertical_column', bdate=bdate, edate=edate)
            
            return no2ds, hchods
        
        try:
            no2ds, hchods = await asyncio.wait_for(
                asyncio.to_thread(_fetch_day),
                timeout=DOWNLOAD_TIMEOUT * 3  # Longer timeout for daily batches
            )
        except asyncio.TimeoutError:
            logger.error(f"[BATCH] Timeout fetching day {d_str}")
            if status:
                status.emit("error", f"⏱️ Timeout: {d_str}")
            return []
        except Exception as e:
            error_str = str(e)
            if "Unknown file format" in error_str or "NetCDF: Unknown file format" in error_str:
                logger.info(f"[BATCH] No data available for {d_str}")
                if status: status.emit("warning", f"No data for {d_str}")
                return []
            raise
        
        # Split into per-hour files for consistent downstream processing
        saved = []
        
        try:
            # Get timestamps from the response
            if 'TSTEP' in no2ds.dims:
                timestamps = pd.to_datetime(no2ds.TSTEP.values)
            else:
                # Single timestep, just use requested hours
                timestamps = [pd.to_datetime(d_str) + pd.to_timedelta(h, unit='h') for h in hours]
            
            # Group data by hour
            for hour in hours:
                # Check if this hour has data
                hour_data_mask = [t.hour == hour for t in timestamps]
                if not any(hour_data_mask):
                    continue
                
                filename = f"tempo_{d_str}_{hour:02d}.nc"
                filepath = dataset_dir / filename
                
                # Extract data for this hour
                if 'TSTEP' in no2ds.dims:
                    hour_indices = [i for i, m in enumerate(hour_data_mask) if m]
                    if not hour_indices:
                        continue
                    
                    no2_hour = no2ds.isel(TSTEP=hour_indices)
                    hcho_hour = hchods.isel(TSTEP=hour_indices) if 'TSTEP' in hchods.dims else hchods
                else:
                    no2_hour = no2ds
                    hcho_hour = hchods
                
                # Create output dataset
                outds = xr.Dataset(attrs=dict(no2ds.attrs))
                
                if 'LATITUDE' in no2_hour:
                    lat_data = no2_hour['LATITUDE']
                    if 'TSTEP' in lat_data.dims:
                        lat_data = lat_data.isel(TSTEP=0)
                    if 'LAY' in lat_data.dims:
                        lat_data = lat_data.isel(LAY=0)
                    outds.coords['LAT'] = (('ROW', 'COL'), lat_data.values.copy())
                    
                if 'LONGITUDE' in no2_hour:
                    lon_data = no2_hour['LONGITUDE']
                    if 'TSTEP' in lon_data.dims:
                        lon_data = lon_data.isel(TSTEP=0)
                    if 'LAY' in lon_data.dims:
                        lon_data = lon_data.isel(LAY=0)
                    outds.coords['LON'] = (('ROW', 'COL'), lon_data.values.copy())
                
                n_var = no2_hour.get('NO2_VERTICAL_CO', xr.DataArray(np.nan))
                h_var = hcho_hour.get('VERTICAL_COLUMN', xr.DataArray(np.nan))

                if 'LAY' in n_var.dims: n_var = n_var.isel(LAY=0)
                if 'LAY' in h_var.dims: h_var = h_var.isel(LAY=0)
                if 'TSTEP' in n_var.dims: n_var = n_var.mean(dim='TSTEP')
                if 'TSTEP' in h_var.dims: h_var = h_var.mean(dim='TSTEP')

                # Mask fill values (typically -9.999e36) as NaN
                n_var = n_var.where(n_var > -1e30, np.nan)
                h_var = h_var.where(h_var > -1e30, np.nan)

                outds['NO2_TropVCD'] = xr.DataArray(n_var.values.copy(), dims=n_var.dims, attrs=dict(n_var.attrs) if hasattr(n_var, 'attrs') else {})
                outds['HCHO_TotVCD'] = xr.DataArray(h_var.values.copy(), dims=h_var.dims, attrs=dict(h_var.attrs) if hasattr(h_var, 'attrs') else {})
                
                # Check validity
                if outds['NO2_TropVCD'].isnull().all() and outds['HCHO_TotVCD'].isnull().all():
                    continue
                
                # Save file
                await asyncio.to_thread(lambda: outds.to_netcdf(filepath, engine='netcdf4', compute=True))
                outds.close()
                
                fsize = filepath.stat().st_size
                if fsize > 1000:
                    saved.append(filepath)
                    logger.info(f"[BATCH] Saved: {filename} ({fsize/1024:.1f} KB)")
                    if status:
                         # Calculate progress within this batch function is tricky, 
                         # but we can at least show the success message in the log
                         status.emit("download", f"✅ Saved: {filename}", None) 
                else:
                    filepath.unlink()
                    
        finally:
            no2ds.close()
            hchods.close()
        
        return saved
    
    async def _download_single_granule(
        self, 
        d_str: str, 
        hour: int, 
        bbox: list[float],
        max_cloud: float,
        max_sza: float,
        dataset_dir: Path,
        status: StatusManager
    ) -> Path | None:
        """Legacy method - kept for backwards compatibility but not used."""
        # Now handled by _download_daily_batch
        return None
    
    
    async def _save_granule(
        self, 
        ds: xr.Dataset, 
        filepath: Path, 
        filename: str, 
        progress: float,
        status: StatusManager
    ) -> Path | None:
        """Save a downloaded dataset to disk with robust error handling."""
        
        logger.info(f"[SAVE] Saving: {filepath}")
        
        # Delete existing file if present
        if filepath.exists():
            logger.info(f"[SAVE] Deleting existing file: {filepath}")
            if status: status.emit("info", f"🗑️ Overwriting {filename}")
            gc.collect()
            try:
                filepath.unlink()
            except PermissionError:
                await asyncio.sleep(0.5)
                gc.collect()
                try:
                    filepath.unlink()
                except PermissionError as e:
                    logger.error(f"[SAVE] Cannot delete locked file: {e}")
                    if status:
                        status.emit("error", f"🔒 File locked: {filename}")
                    ds.close()
                    return None
        
        # Save to netCDF
        try:
            await asyncio.to_thread(
                lambda: ds.to_netcdf(filepath, engine='netcdf4', compute=True)
            )
            ds.close()
        except PermissionError as e:
            logger.error(f"[SAVE] Permission denied: {e}")
            if status:
                status.emit("error", f"🔒 Permission denied: {filename}")
            ds.close()
            return None
        except Exception as e:
            logger.error(f"[SAVE] Save error: {e}")
            logger.error(f"[SAVE] Traceback:\n{traceback.format_exc()}")
            if status:
                status.emit("error", f"❌ Save failed: {filename} - {e}")
            ds.close()
            return None
        
        # Validate saved file
        if not filepath.exists():
            logger.error(f"[SAVE] File missing after save!")
            if status:
                status.emit("error", f"❌ File disappeared: {filename}")
            return None
        
        fsize = filepath.stat().st_size
        if fsize < 1000:
            logger.warning(f"[SAVE] File too small: {fsize} bytes")
            if status:
                status.emit("error", f"⚠️ File too small: {filename}")
            filepath.unlink()
            return None
        
        logger.info(f"[SAVE] Success: {filename} ({fsize/1024:.1f} KB)")
        if status:
            status.emit("download", f"✅ {filename} ({fsize/1024:.1f} KB)", progress)
        
        return filepath
    
    async def _simulate_download(self, dates, hours, dataset_name, status):
        """Fallback simulation for testing without credentials/deps."""
        dataset_dir = self.workdir
        dataset_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        total = len(dates) * len(hours)
        curr = 0
        
        for d in dates:
            for h in hours:
                curr += 1
                if status: status.emit("download", f"Simulating: {d} @ {h:02d}:00", curr/total)
                await asyncio.sleep(0.3)  # Faster simulation
                # Create dummy file
                p = dataset_dir / f"tempo_{d}_{h:02d}.nc"
                with open(p, 'w') as f: f.write("dummy netcdf")
                saved.append(p)
                
        return saved
