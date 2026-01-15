"""Data exporter module for TEMPO Analyzer."""

import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
import logging
from typing import Optional
from math import radians, sin, cos, sqrt, atan2

from .constants import SITES

logger = logging.getLogger(__name__)

MISSING_VALUE = -999


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in kilometers."""
    R = 6371.0  # Earth radius in kilometers
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def find_n_nearest_cells(lat: float, lon: float, lats_2d: np.ndarray,
                         lons_2d: np.ndarray, n: int) -> list[tuple]:
    """Find the N nearest grid cells to a target location."""
    distances = np.zeros_like(lats_2d)
    rows, cols = lats_2d.shape
    for i in range(rows):
        for j in range(cols):
            distances[i, j] = haversine(lat, lon, lats_2d[i, j], lons_2d[i, j])
    flat_idx = np.argsort(distances, axis=None)[:n]
    row_indices, col_indices = np.unravel_index(flat_idx, distances.shape)
    result = []
    for r, c in zip(row_indices, col_indices):
        result.append((int(r), int(c), float(distances[r, c])))
    return result





def filter_sites_in_bbox(sites: dict, dataset: xr.Dataset) -> dict:
    """Filter sites to only those within dataset's bounding box."""
    if 'LAT' not in dataset.coords:
        return {}
    lats = dataset['LAT'].values
    lons = dataset['LON'].values
    min_lat, max_lat = lats.min(), lats.max()
    min_lon, max_lon = lons.min(), lons.max()
    return {
        site: coords for site, coords in sites.items()
        if (min_lat <= coords[0] <= max_lat) and (min_lon <= coords[1] <= max_lon)
    }


def apply_monthly_hourly_fill(df: pd.DataFrame, value_columns: list) -> pd.DataFrame:
    """Fill NaN values using monthly-hourly climatological means."""
    df_filled = df.copy()
    group_cols = []
    if 'Site' in df.columns:
        group_cols.append('Site')
    if 'Month' in df.columns:
        group_cols.append('Month')
    if 'Hour' in df.columns:
        group_cols.append('Hour')
    if not group_cols:
        return df_filled
    means = df_filled.groupby(group_cols)[value_columns].transform('mean')
    df_filled[value_columns] = df_filled[value_columns].fillna(means)
    return df_filled


class DataExporter:
    """Handles exporting processed data to Excel formats."""

    def __init__(self, output_dir: Path):
        """Initialize exporter with output directory."""
        self.output_dir = output_dir / "exports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_dataset(self,
                      dataset: xr.Dataset,
                      dataset_name: str,
                      export_format: str,
                      num_points: int = 4,
                      utc_offset: float = -6.0,
                      metadata: Optional[dict] = None) -> list[str]:
        """Export dataset in specified format.

        Args:
            dataset: xarray.Dataset with LAT, LON coords
            dataset_name: Base name for output files
            export_format: One of "hourly", "daily"
            utc_offset: Hours offset from UTC for local time (default -6.0)
            metadata: Optional dictionary of dataset metadata (settings, stats)

        Returns:
            List of generated file paths
        """
        if export_format == "hourly":
            return self._export_hourly(dataset, dataset_name, utc_offset, num_points, metadata)
        elif export_format == "daily":
            return self._export_daily(dataset, dataset_name, utc_offset, num_points, metadata)
        else:
            raise ValueError(f"Unknown export format: {export_format}")

    def _create_metadata_df(self, metadata: dict) -> pd.DataFrame:
        """Create a DataFrame from metadata dictionary for export."""
        if not metadata:
            return pd.DataFrame({'Parameter': ['No metadata available'], 'Value': ['']})
        
        rows = []
        # Add basic settings first
        settings_keys = ['max_cloud', 'max_sza', 'date_start', 'date_end', 'day_filter', 'hour_filter']
        for k in settings_keys:
            if k in metadata:
                rows.append({'Parameter': k, 'Value': str(metadata[k])})
        
        # Add any other keys
        for k, v in metadata.items():
            if k not in settings_keys:
                rows.append({'Parameter': k, 'Value': str(v)})
                
        return pd.DataFrame(rows)

    def _get_time_info(self, dataset: xr.Dataset):
        """Extract time dimension and values from dataset."""
        time_dim = 'TSTEP' if 'TSTEP' in dataset.dims else 'HOUR' if 'HOUR' in dataset.dims else None
        if time_dim is None:
            return None, None
        
        if time_dim == 'TSTEP':
            return time_dim, pd.to_datetime(dataset['TSTEP'].values)
        else:
            # HOUR dimension - check if datetime or just integers
            hour_values = dataset['HOUR'].values
            if np.issubdtype(hour_values.dtype, np.datetime64):
                return time_dim, pd.to_datetime(hour_values)
            else:
                # Create datetime index from hour integers (use placeholder date)
                # We'll need actual dates from the dataset or user
                hours = [int(h) for h in hour_values]
                return time_dim, hours  # Return raw hours, handle downstream

    def _export_hourly(self, dataset: xr.Dataset, dataset_name: str,
                       utc_offset: float, num_points: int = 9,
                       metadata: Optional[dict] = None) -> list[str]:
        """Export hourly format - separate file per site with N cells.
        
        Columns: UTC_Time, Local_Time (UTC-X.0), Cell1_NO2...CellN_NO2, Cell1_HCHO...CellN_HCHO
        """
        time_dim, time_values = self._get_time_info(dataset)
        if time_dim is None or 'LAT' not in dataset.coords:
            logger.warning("Missing required coords/dims")
            return []

        lats = dataset['LAT'].values
        lons = dataset['LON'].values

        valid_sites = filter_sites_in_bbox(SITES, dataset)
        if not valid_sites:
            logger.warning("No sites found within dataset bounds")
            return []

        # Handle time values
        if isinstance(time_values, pd.DatetimeIndex):
            utc_times = time_values
            local_times = utc_times + pd.Timedelta(hours=utc_offset)
            utc_col = utc_times
            local_col = local_times
        else:
            # Hours only - create time columns as strings
            hours = time_values
            utc_col = [f"{h:02d}:00 UTC" for h in hours]
            local_hours = [(h + int(utc_offset)) % 24 for h in hours]
            local_col = [f"{h:02d}:00 Local" for h in local_hours]

        generated_files = []

        for site, (t_lat, t_lon) in valid_sites.items():
            # Find cells based on num_points
            cells = find_n_nearest_cells(t_lat, t_lon, lats, lons, num_points)
            
            # Build data dictionary
            data = {
                'UTC_Time': utc_col,
                f'Local_Time (UTC{utc_offset:+.1f})': local_col,
            }

            # Extract NO2 and HCHO interleaved for each cell
            for i, (r, c, dist) in enumerate(cells):
                cell_num = i + 1
                if 'NO2_TropVCD' in dataset:
                    values = dataset['NO2_TropVCD'].isel(ROW=r, COL=c).values.flatten()
                    # Keep NaN as empty, not -999
                    data[f'Cell{cell_num}_NO2'] = values
                if 'HCHO_TotVCD' in dataset:
                    values = dataset['HCHO_TotVCD'].isel(ROW=r, COL=c).values.flatten()
                    data[f'Cell{cell_num}_HCHO'] = values

            # Create hourly data DataFrame
            df = pd.DataFrame(data)
            
            # Create Grid_Info sheet with cell metadata
            grid_info = []
            for i, (r, c, dist) in enumerate(cells):
                cell_num = i + 1
                cell_lat = float(lats[r, c])
                cell_lon = float(lons[r, c])
                grid_info.append({
                    'Cell_ID': f'Cell{cell_num}',
                    'Grid_Lat': cell_lat,
                    'Grid_Lon': cell_lon,
                    'Dist_to_Site_km': dist,
                })
            df_grid = pd.DataFrame(grid_info)
            
            # Save to Excel with multiple sheets
            fname = self.output_dir / f"{site}_{dataset_name}_hourly.xlsx"
            with pd.ExcelWriter(fname, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Hourly_Data', index=False)
                df_grid.to_excel(writer, sheet_name='Grid_Info', index=False)
                if metadata:
                    # Add site-specific stats to metadata
                    meta_df = self._create_metadata_df(metadata)
                    # Calculate missing data stats for this site
                    total_pts = len(df)
                    no2_missing = df.filter(like='_NO2').isna().sum().sum()
                    no2_total = df.filter(like='_NO2').size
                    hcho_missing = df.filter(like='_HCHO').isna().sum().sum()
                    hcho_total = df.filter(like='_HCHO').size
                    
                    stats_rows = pd.DataFrame([
                        {'Parameter': 'Site', 'Value': site},
                        {'Parameter': 'Total_Time_Steps', 'Value': total_pts},
                        {'Parameter': 'NO2_Missing_Pct', 'Value': f"{(no2_missing/no2_total)*100:.1f}%" if no2_total else "0%"},
                        {'Parameter': 'HCHO_Missing_Pct', 'Value': f"{(hcho_missing/hcho_total)*100:.1f}%" if hcho_total else "0%"}
                    ])
                    meta_final = pd.concat([meta_df, stats_rows], ignore_index=True)
                    meta_final.to_excel(writer, sheet_name='Metadata', index=False)
            
            generated_files.append(str(fname))

        return generated_files

    def _export_daily(self, dataset: xr.Dataset, dataset_name: str,
                      utc_offset: float, num_points: int = 8,
                      metadata: Optional[dict] = None) -> list[str]:
        """Export daily format - single file with all sites.
        
        Columns: Date, Site, TMP_NO2_NoFill_Ngridcells, TMP_NO2_NoFill_Ncnt, ...
        Uses hours 08-14 (inclusive) local time.
        Uses configurable num_points (default 8) for cell selection.
        """
        time_dim, time_values = self._get_time_info(dataset)
        if time_dim is None or 'LAT' not in dataset.coords:
            logger.warning("Missing required coords/dims")
            return []

        lats = dataset['LAT'].values
        lons = dataset['LON'].values

        valid_sites = filter_sites_in_bbox(SITES, dataset)
        if not valid_sites:
            logger.warning("No sites found within dataset bounds")
            return []

        # Handle time values
        if isinstance(time_values, pd.DatetimeIndex):
            utc_times = time_values
            local_times = utc_times + pd.Timedelta(hours=utc_offset)
        else:
            # For hour-only data, we can't create proper daily aggregates
            logger.warning("Daily export requires full timestamps, not just hours")
            return []

        all_rows = []

        for site, (t_lat, t_lon) in valid_sites.items():
            # Find cells based on num_points
            cells = find_n_nearest_cells(t_lat, t_lon, lats, lons, num_points)
            
            n_cells = len(cells)
            half_cells = n_cells // 2 if n_cells > 1 else n_cells
            
            # Extract raw data for all cells
            raw_data = {
                'Local_Time': local_times,
                'Date': local_times.date,
                'Hour': local_times.hour,
                'Month': local_times.month,
                'Site': site,
            }
            
            # Extract NO2 and HCHO for each cell
            for i, (r, c, dist) in enumerate(cells):
                if 'NO2_TropVCD' in dataset:
                    raw_data[f'Cell{i}_NO2'] = dataset['NO2_TropVCD'].isel(ROW=r, COL=c).values.flatten()
                if 'HCHO_TotVCD' in dataset:
                    raw_data[f'Cell{i}_HCHO'] = dataset['HCHO_TotVCD'].isel(ROW=r, COL=c).values.flatten()
            
            df_site = pd.DataFrame(raw_data)
            
            # Filter to hours 8-14 local time
            df_filtered = df_site[(df_site['Hour'] >= 8) & (df_site['Hour'] <= 14)].copy()
            
            if df_filtered.empty:
                continue
            
            # Create filled version
            no2_cols = [f'Cell{i}_NO2' for i in range(n_cells) if f'Cell{i}_NO2' in df_filtered.columns]
            hcho_cols = [f'Cell{i}_HCHO' for i in range(n_cells) if f'Cell{i}_HCHO' in df_filtered.columns]
            value_cols = no2_cols + hcho_cols
            
            df_filled = apply_monthly_hourly_fill(df_filtered, value_cols)
            
            # Aggregate by date
            for date, grp in df_filtered.groupby('Date'):
                row = {'Date': date, 'Site': site}
                grp_filled = df_filled[df_filled['Date'] == date]
                
                # Use half cells and full cells based on selected num_points
                for cell_count, label in [(half_cells, str(half_cells)), (n_cells, str(n_cells))]:
                    # NoFill
                    no2_cols_n = [f'Cell{i}_NO2' for i in range(cell_count) if f'Cell{i}_NO2' in grp.columns]
                    hcho_cols_n = [f'Cell{i}_HCHO' for i in range(cell_count) if f'Cell{i}_HCHO' in grp.columns]
                    
                    if no2_cols_n:
                        no2_vals = grp[no2_cols_n].values.flatten()
                        no2_valid = no2_vals[~np.isnan(no2_vals)]
                        row[f'TMP_NO2_NoFill_{label}gridcells'] = np.mean(no2_valid) if len(no2_valid) > 0 else MISSING_VALUE
                        row[f'TMP_NO2_NoFill_{label}cnt'] = len(no2_valid)
                    
                    if hcho_cols_n:
                        hcho_vals = grp[hcho_cols_n].values.flatten()
                        hcho_valid = hcho_vals[~np.isnan(hcho_vals)]
                        row[f'TMP_HCHO_NoFill_{label}gridcells'] = np.mean(hcho_valid) if len(hcho_valid) > 0 else MISSING_VALUE
                        row[f'TMP_HCHO_NoFill_{label}cnt'] = len(hcho_valid)
                    
                    # Fill
                    if no2_cols_n:
                        no2_vals_f = grp_filled[no2_cols_n].values.flatten()
                        no2_valid_f = no2_vals_f[~np.isnan(no2_vals_f)]
                        row[f'TMP_NO2_Fill_{label}gridcells'] = np.mean(no2_valid_f) if len(no2_valid_f) > 0 else MISSING_VALUE
                        row[f'TMP_NO2_Fill_{label}cnt'] = len(no2_valid_f)
                    
                    if hcho_cols_n:
                        hcho_vals_f = grp_filled[hcho_cols_n].values.flatten()
                        hcho_valid_f = hcho_vals_f[~np.isnan(hcho_vals_f)]
                        row[f'TMP_HCHO_Fill_{label}gridcells'] = np.mean(hcho_valid_f) if len(hcho_valid_f) > 0 else MISSING_VALUE
                        row[f'TMP_HCHO_Fill_{label}cnt'] = len(hcho_valid_f)
                
                all_rows.append(row)
        
        if not all_rows:
            return []
        
        # Create final DataFrame and order columns properly
        df_final = pd.DataFrame(all_rows)
        
        # Order: Date, Site, then NoFill columns (smaller cell count first), then Fill columns
        base_cols = ['Date', 'Site']
        other_cols = [c for c in df_final.columns if c not in base_cols]
        
        # Sort columns: NoFill before Fill, smaller cell count first, NO2 before HCHO
        def col_sort_key(col):
            fill_order = 0 if 'NoFill' in col else 1
            no2_order = 0 if 'NO2' in col else 1
            cnt_order = 1 if 'cnt' in col else 0
            # Extract cell count from column name
            import re
            match = re.search(r'(\d+)gridcells', col)
            cell_count = int(match.group(1)) if match else 0
            match2 = re.search(r'(\d+)cnt', col)
            cell_count = int(match2.group(1)) if match2 else cell_count
            return (fill_order, cell_count, no2_order, cnt_order)
        
        other_cols.sort(key=col_sort_key)
        df_final = df_final[base_cols + other_cols]
        df_final = df_final.sort_values(['Date', 'Site']).reset_index(drop=True)
        
        # Save
        fname = self.output_dir / f"{dataset_name}_daily.xlsx"
        with pd.ExcelWriter(fname, engine='openpyxl') as writer:
            df_final.to_excel(writer, sheet_name='Daily_Data', index=False)
            if metadata:
                meta_df = self._create_metadata_df(metadata)
                
                # Calculate stats for daily data
                stats_data = []
                total_rows = len(df_final)
                stats_data.append({'Parameter': 'Total_Site_Days', 'Value': total_rows})
                
                # Check missing data in NoFill columns
                nofill_cols = [c for c in df_final.columns if 'NoFill' in c and 'gridcells' in c]
                for col in nofill_cols:
                    # MISSING_VALUE is -999
                    missing_cnt = (df_final[col] == MISSING_VALUE).sum()
                    stats_data.append({
                        'Parameter': f'{col}_Missing_Pct', 
                        'Value': f"{(missing_cnt/total_rows)*100:.1f}%" if total_rows else "0%"
                    })
                
                stats_df = pd.DataFrame(stats_data)
                meta_final = pd.concat([meta_df, stats_df], ignore_index=True)
                meta_final.to_excel(writer, sheet_name='Metadata', index=False)
        
        return [str(fname)]
