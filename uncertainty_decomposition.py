#!/usr/bin/env python3
"""
Uncertainty Decomposition Analysis
Calculates model uncertainty, scenario uncertainty, and internal variability 
components for climate model projections following Hawkins & Sutton (2009) methodology
"""

import os
import numpy as np
import xarray as xr
import warnings
from datetime import datetime
import time

warnings.filterwarnings('ignore')

# Regional bounds - modify as needed
REGION_BOUNDS = {
    'lat_min': -35.0,
    'lat_max': 40.0,
    'lon_min': -20.0,
    'lon_max': 55.0
}


def create_region_mask(lat, lon):
    """Create mask for specified regional bounds."""
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')
    mask = (lat_grid >= REGION_BOUNDS['lat_min']) & (lat_grid <= REGION_BOUNDS['lat_max']) & \
           (lon_grid >= REGION_BOUNDS['lon_min']) & (lon_grid <= REGION_BOUNDS['lon_max'])
    
    return mask


def debug_print_stats(data_array, name, step=None):
    """Print statistical summary of data array for debugging."""
    if step:
        prefix = f"[Step {step}] "
    else:
        prefix = ""
        
    if isinstance(data_array, xr.DataArray):
        values = data_array.values
    else:
        values = data_array
        
    valid_data = ~np.isnan(values)
    if np.any(valid_data):
        min_val = np.nanmin(values)
        max_val = np.nanmax(values)
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)
        
        print(f"{prefix}{name}: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}, std={std_val:.6f}")
    else:
        print(f"{prefix}{name}: No valid data (all NaN)")


def validate_uncertainty_components(model_unc, scenario_unc, internal_var, total_unc, time_idx):
    """Validate that uncertainty components are correctly calculated."""
    # Check for negative values
    for comp, name in zip(
        [model_unc, scenario_unc, internal_var, total_unc],
        ['Model uncertainty', 'Scenario uncertainty', 'Internal variability', 'Total uncertainty']
    ):
        neg_count = np.sum(comp < 0)
        if neg_count > 0:
            print(f"WARNING: {name} has {neg_count} negative values at time step {time_idx}")
    
    # Verify total uncertainty calculation
    calculated_total = model_unc + scenario_unc + internal_var
    diff = np.abs(calculated_total - total_unc)
    max_diff = np.nanmax(diff)
    
    if max_diff > 1e-10:
        print(f"ERROR: Total uncertainty mismatch at time step {time_idx}, max difference: {max_diff}")
    
    # Calculate and verify fractions
    valid = total_unc > 0
    
    if np.sum(valid) > 0:
        model_frac = np.zeros_like(total_unc)
        scenario_frac = np.zeros_like(total_unc)
        internal_frac = np.zeros_like(total_unc)
        
        model_frac[valid] = model_unc[valid] / total_unc[valid]
        scenario_frac[valid] = scenario_unc[valid] / total_unc[valid]
        internal_frac[valid] = internal_var[valid] / total_unc[valid]
        
        model_perc = 100 * np.nanmean(model_frac[valid])
        scenario_perc = 100 * np.nanmean(scenario_frac[valid])
        internal_perc = 100 * np.nanmean(internal_frac[valid])
        
        print(f"Average contribution at time step {time_idx}:")
        print(f"  Model uncertainty: {model_perc:.2f}%")
        print(f"  Scenario uncertainty: {scenario_perc:.2f}%")
        print(f"  Internal variability: {internal_perc:.2f}%")


def extract_model_name(filename):
    """Extract model name from filename - customize patterns as needed."""
    model_patterns = [
        'ACCESS-CM2', 'ACCESS-ESM1-5', 'CanESM5', 'CESM2', 'GFDL-ESM4', 
        'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KACE-1-0-G', 'NorESM2-LM'
        # Add your model patterns here
    ]
    
    for pattern in model_patterns:
        if pattern in filename:
            return pattern
    
    return None


def process_datasets(file_list, variable_name='rsds', region_only=True):
    """Process NetCDF files and extract data for analysis."""
    print("\n===== PROCESSING INPUT DATASETS =====")
    print(f"Starting to process {len(file_list)} files for variable {variable_name}")
    
    raw_data = {}
    models = set()
    scenarios = set()
    
    for file_idx, file_path in enumerate(file_list, 1):
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            continue
            
        filename = os.path.basename(file_path)
        
        # Extract scenario from filename
        if 'historical' in filename:
            scenario = 'historical'
        elif 'ssp' in filename:
            for part in filename.split('_'):
                if part.startswith('ssp'):
                    scenario = part
                    break
        else:
            print(f"Skipping {filename}: not a historical or scenario file")
            continue
            
        model = extract_model_name(filename)
        if model is None:
            print(f"Skipping {filename}: cannot extract model information")
            continue
        
        print(f"\nProcessing file {file_idx}/{len(file_list)}: {filename}")
        print(f"  Detected model: {model}, scenario: {scenario}")
        
        try:
            # Try different engines to open file
            ds = xr.open_dataset(file_path, decode_times=False)
            
            # Find the target variable
            if variable_name in ds:
                da = ds[variable_name]
            else:
                print(f"  Variable {variable_name} not found in dataset")
                continue
            
            # Standardize dimension names
            lat_dim = None
            lon_dim = None
            time_dim = None
            
            for dim in ds.dims:
                if dim.lower() in ['lat', 'latitude']:
                    lat_dim = dim
                elif dim.lower() in ['lon', 'longitude']:
                    lon_dim = dim
                elif dim.lower() in ['time']:
                    time_dim = dim
            
            if not (lat_dim and lon_dim and time_dim):
                print(f"  Missing required dimensions")
                continue
            
            # Rename dimensions to standard names
            rename_dict = {}
            if lat_dim != 'lat':
                rename_dict[lat_dim] = 'lat'
            if lon_dim != 'lon':
                rename_dict[lon_dim] = 'lon'
            if time_dim != 'time':
                rename_dict[time_dim] = 'time'
                
            if rename_dict:
                da = da.rename(rename_dict)
            
            # Convert longitude if needed (0-360 to -180-180)
            lon_max = float(da.lon.max())
            if lon_max > 180:
                da = da.assign_coords(lon=(da.lon + 180) % 360 - 180)
                da = da.sortby('lon')
            
            # Subsample time if too many steps
            if len(da.time) > 100:
                step = 12  # Keep every 12th time step (annual from monthly)
                da = da.isel(time=slice(0, None, step))
                print(f"  Sampling time steps (every {step}th)")
            
            if model not in raw_data:
                raw_data[model] = {}
            
            # Add regional mask if requested
            if region_only:
                lat_vals = da.lat.values
                lon_vals = da.lon.values
                region_mask = create_region_mask(lat_vals, lon_vals)
                da.attrs['region_mask'] = region_mask
                
            models.add(model)
            scenarios.add(scenario)
            
            raw_data[model][scenario] = da
            
            debug_print_stats(da, f"{model}_{scenario} raw data")
            
            print(f"  Successfully processed {model} {scenario}")
            
        except Exception as e:
            print(f"  Error processing {file_path}: {str(e)}")
    
    print(f"\nProcessed data for {len(models)} models and {len(scenarios)} scenarios")
    
    return raw_data, list(models), list(scenarios)


def regrid_datasets(raw_data, models, scenarios):
    """Regrid all datasets to common grid."""
    print("\n===== REGRIDDING DATASETS TO COMMON GRID =====")
    
    # Find coarsest grid as target
    min_grid_size = float('inf')
    target_model = None
    target_scenario = None
    
    for model in models:
        if model not in raw_data:
            continue
            
        for scenario in scenarios:
            if scenario not in raw_data[model]:
                continue
                
            lat_size = len(raw_data[model][scenario].lat)
            lon_size = len(raw_data[model][scenario].lon)
            grid_size = lat_size * lon_size
            
            if grid_size < min_grid_size:
                min_grid_size = grid_size
                target_model = model
                target_scenario = scenario
    
    if target_model is None:
        print("Error: Could not find target grid")
        return None
        
    print(f"Using grid from {target_model}_{target_scenario} as target")
    target_lat = raw_data[target_model][target_scenario].lat
    target_lon = raw_data[target_model][target_scenario].lon
    
    regridded_data = {}
    
    for model in models:
        if model not in raw_data:
            continue
            
        regridded_data[model] = {}
        
        for scenario in scenarios:
            if scenario not in raw_data[model]:
                continue
                
            source_data = raw_data[model][scenario]
            
            # Check if already on target grid
            if (len(source_data.lat) == len(target_lat) and 
                len(source_data.lon) == len(target_lon) and
                np.allclose(source_data.lat, target_lat) and
                np.allclose(source_data.lon, target_lon)):
                regridded_data[model][scenario] = source_data
                continue
                
            # Regrid using xarray interpolation
            try:
                regridded = source_data.interp(lat=target_lat, lon=target_lon)
                regridded_data[model][scenario] = regridded
                
            except Exception as e:
                print(f"  Error during regridding: {e}")
    
    return regridded_data


def fit_polynomial(data, degree=4, debug_step=None):
    """Fit polynomial to time series to extract forced response."""
    if debug_step:
        print(f"\n----- Polynomial fitting for {debug_step} -----")
    
    time_numeric = np.arange(len(data.time))
    
    nlat = len(data.lat)
    nlon = len(data.lon)
    ntime = len(data.time)
    
    print(f"Fitting polynomial of degree {degree} to data with shape: time={ntime}, lat={nlat}, lon={nlon}")
    
    fitted = np.zeros_like(data.values)
    
    fit_success = 0
    fit_failed = 0
    
    for i in range(nlat):
        for j in range(nlon):
            y = data.isel(lat=i, lon=j).values
            
            if np.all(np.isnan(y)):
                fitted[:, i, j] = np.nan
                continue
            
            valid = ~np.isnan(y)
            if np.sum(valid) <= degree:
                fitted[:, i, j] = np.nan
                fit_failed += 1
                continue
                
            try:
                coeffs = np.polyfit(time_numeric[valid], y[valid], degree)
                fitted[:, i, j] = np.polyval(coeffs, time_numeric)
                fit_success += 1
                
            except Exception:
                fitted[:, i, j] = np.nan
                fit_failed += 1
    
    print(f"Fitting complete: Success: {fit_success}, Failed: {fit_failed}")
    
    fitted_da = xr.DataArray(
        fitted,
        coords={'time': data.time, 'lat': data.lat, 'lon': data.lon},
        dims=['time', 'lat', 'lon']
    )
    
    return fitted_da


def calculate_uncertainty_components(data, models, scenarios, poly_degree=4):
    """Calculate uncertainty components following Hawkins & Sutton (2009)."""
    print("\n===== CALCULATING UNCERTAINTY COMPONENTS =====")
    
    # Get dimensions from first available dataset
    first_model = models[0]
    first_scenario = scenarios[0]
    da = data[first_model][first_scenario]
    
    ntimes = len(da.time)
    nlat = len(da.lat)
    nlon = len(da.lon)
    shape = (ntimes, nlat, nlon)
    
    print(f"Calculating uncertainty for {ntimes} time steps and {nlat}x{nlon} grid")
    
    # Initialize uncertainty arrays
    model_uncertainty = np.full(shape, np.nan)
    scenario_uncertainty = np.full(shape, np.nan)
    internal_variability = np.full(shape, np.nan)
    
    print("\n----- Extracting forced responses -----")
    forced_responses = {}
    residuals = {}
    
    # Extract forced response (polynomial fit) for each model-scenario combination
    for model in models:
        if model not in data:
            continue
            
        print(f"\nProcessing model: {model}")
        forced_responses[model] = {}
        residuals[model] = {}
        
        for scenario in scenarios:
            if scenario not in data[model]:
                continue
                
            da = data[model][scenario]
            
            # Fit polynomial to extract forced response
            forced = fit_polynomial(da, degree=poly_degree, debug_step=f"{model}_{scenario}")
            forced_responses[model][scenario] = forced
            
            # Calculate residuals (internal variability)
            residuals[model][scenario] = da - forced
    
    print("\n----- Calculating uncertainty components -----")
    
    # Calculate components for each time step
    for t in range(ntimes):
        if t % 10 == 0:
            print(f"\nProcessing time step {t+1}/{ntimes}")
            
        # Collect data for this time step
        model_means = {}
        scenario_means = {scenario: [] for scenario in scenarios}
        all_residuals = []
        
        for model in forced_responses:
            model_values = []
            
            for scenario in forced_responses[model]:
                try:
                    value_at_t = forced_responses[model][scenario].isel(time=t).values
                    scenario_means[scenario].append(value_at_t)
                    model_values.append(value_at_t)
                    
                    if scenario in residuals[model]:
                        all_residuals.append(residuals[model][scenario].isel(time=t).values)
                except IndexError:
                    continue
            
            if model_values:
                model_means[model] = np.nanmean(np.stack(model_values), axis=0)
        
        # Calculate model uncertainty (variance across models)
        if len(model_means) > 1:
            try:
                model_values_array = np.stack(list(model_means.values()))
                model_uncertainty[t] = np.nanvar(model_values_array, axis=0, ddof=1)
            except ValueError:
                model_uncertainty[t] = np.nan
        
        # Calculate scenario uncertainty (variance across scenarios)
        scenario_values = []
        for scenario in scenario_means:
            if scenario_means[scenario]:
                try:
                    scenario_values.append(np.nanmean(np.stack(scenario_means[scenario]), axis=0))
                except ValueError:
                    pass
        
        if len(scenario_values) > 1:
            try:
                scenario_uncertainty[t] = np.nanvar(np.stack(scenario_values), axis=0, ddof=1)
            except ValueError:
                scenario_uncertainty[t] = np.nan
        
        # Calculate internal variability (variance of residuals)
        if all_residuals:
            try:
                residuals_array = np.stack(all_residuals)
                internal_variability[t] = np.nanvar(residuals_array, axis=0, ddof=1)
            except ValueError:
                internal_variability[t] = np.nan
        
        # Validate components occasionally
        if t % 20 == 0:
            m_unc = np.nan_to_num(model_uncertainty[t])
            s_unc = np.nan_to_num(scenario_uncertainty[t])
            i_var = np.nan_to_num(internal_variability[t])
            total = m_unc + s_unc + i_var
            
            validate_uncertainty_components(m_unc, s_unc, i_var, total, t)
    
    print("\n----- Creating output dataset -----")
    
    # Create coordinate arrays
    time_coords = da.time
    lat_coords = da.lat
    lon_coords = da.lon
    
    # Fill NaN values with zeros for uncertainty calculations
    model_uncertainty_filled = np.nan_to_num(model_uncertainty)
    scenario_uncertainty_filled = np.nan_to_num(scenario_uncertainty)
    internal_variability_filled = np.nan_to_num(internal_variability)
    
    # Calculate total uncertainty
    total_uncertainty = model_uncertainty_filled + scenario_uncertainty_filled + internal_variability_filled
    
    # Calculate fractional contributions
    model_fraction = np.zeros((ntimes, nlat, nlon))
    scenario_fraction = np.zeros((ntimes, nlat, nlon))
    internal_fraction = np.zeros((ntimes, nlat, nlon))
    
    valid = total_uncertainty > 0
    model_fraction[valid] = model_uncertainty_filled[valid] / total_uncertainty[valid]
    scenario_fraction[valid] = scenario_uncertainty_filled[valid] / total_uncertainty[valid]
    internal_fraction[valid] = internal_variability_filled[valid] / total_uncertainty[valid]
    
    # Create DataArrays
    uncertainty_data = {
        'model_uncertainty': (['time', 'lat', 'lon'], model_uncertainty),
        'scenario_uncertainty': (['time', 'lat', 'lon'], scenario_uncertainty),
        'internal_variability': (['time', 'lat', 'lon'], internal_variability),
        'total_uncertainty': (['time', 'lat', 'lon'], total_uncertainty),
        'model_fraction': (['time', 'lat', 'lon'], model_fraction),
        'scenario_fraction': (['time', 'lat', 'lon'], scenario_fraction),
        'internal_fraction': (['time', 'lat', 'lon'], internal_fraction)
    }
    
    coords = {
        'time': time_coords,
        'lat': lat_coords,
        'lon': lon_coords
    }
    
    ds = xr.Dataset(uncertainty_data, coords=coords)
    
    return ds


def main():
    """Main execution function."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"uncertainty_results_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find NetCDF files - modify pattern as needed
    file_list = [f for f in os.listdir('.') if f.endswith('.nc')]
    
    if not file_list:
        print("No suitable NetCDF files found in current directory.")
        return
    
    print(f"Found {len(file_list)} files for processing")
    
    # Process datasets
    raw_data, models, scenarios = process_datasets(
        file_list, 
        variable_name='rsds',
        region_only=True
    )
    
    if not raw_data:
        print("No data could be processed. Exiting.")
        return
        
    print(f"Processed data for {len(models)} models and {len(scenarios)} scenarios")
    
    # Regrid to common grid
    regridded_data = regrid_datasets(raw_data, models, scenarios)
    
    if not regridded_data:
        print("Error during regridding. Exiting.")
        return
    
    # Calculate uncertainty components
    print("Calculating uncertainty components...")
    try:
        uncertainty_ds = calculate_uncertainty_components(
            regridded_data,
            models,
            scenarios,
            poly_degree=4
        )
        
        # Save results
        output_file = os.path.join(output_dir, 'uncertainty_partitioning.nc')
        print(f"Saving results to {output_file}")
        uncertainty_ds.to_netcdf(output_file)
        
        print("Analysis complete")
        
    except Exception as e:
        print(f"Error during uncertainty calculation: {e}")


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        elapsed = time.time() - start_time
        print(f"Execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
