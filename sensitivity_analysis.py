#!/usr/bin/env python3
"""
Sensitivity Analysis for Solar Radiation Bias Correction
Physics-Constrained Deep Learning Framework for CMIP6 Solar Radiation Over Africa

This module calculates sensitivity coefficients between surface solar radiation biases 
and key atmospheric factors (total cloud cover and clear-sky radiation) across Africa.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def get_time_dim(da):
    """Get the time dimension name for a given DataArray."""
    for dim in ['time', 'valid_time']:
        if dim in da.dims:
            return dim
    raise ValueError("No time dimension found")


def standardize_coords(da):
    """Standardize coordinate names to lat/lon."""
    rename_dict = {
        'latitude': 'lat',
        'longitude': 'lon',
        'valid_time': 'time'
    }
    # Only rename if the coordinate exists
    existing_coords = {k: v for k, v in rename_dict.items() if k in da.dims or k in da.coords}
    if existing_coords:
        da = da.rename(existing_coords)
    return da


def standardize_time(da, time_slice):
    """Standardize time dimension and select time period."""
    time_dim = get_time_dim(da)
    if time_dim != 'time':
        da = da.rename({time_dim: 'time'})
    return da.sel(time=time_slice)


def load_and_process_data():
    """Load and process all datasets."""
    print("Loading datasets...")
    
    # Load datasets with proper time dimension handling
    rsds_model = xr.open_dataset('rsds_mon_one_historical126_192_ave.nc')['rsds'].load()
    rsds_obs = xr.open_dataset('rsds_obsv.nc')['SIS'].load()
    tcc_model = xr.open_dataset('clt_ensemble_mean.nc')['clt'].load()
    tcc_obs = xr.open_dataset('TCC_era5_percent.nc')['tcc'].load()
    rsdscs_model = xr.open_dataset('rsdscs_ensemble_mean.nc')['rsdscs'].load()
    rsdscs_obs = xr.open_dataset('ssrdc_era5_wm2.nc')['ssrdc'].load()
    
    # Print initial shapes and time dimensions
    datasets = [rsds_model, rsds_obs, tcc_model, tcc_obs, rsdscs_model, rsdscs_obs]
    names = ['RSDS model', 'RSDS obs', 'TCC model', 'TCC obs', 'RSDSCS model', 'RSDSCS obs']
    
    print("\nInitial dataset information:")
    for name, ds in zip(names, datasets):
        time_dim = get_time_dim(ds)
        print(f"\n{name}:")
        print(f"Shape: {ds.shape}")
        print(f"Time dimension: {time_dim}")
        print(f"Time range: {ds[time_dim].values[0]} to {ds[time_dim].values[-1]}")
    
    # Define common time period
    time_slice = slice('1983-01-01', '2014-12-31')
    
    # Standardize coordinates and time
    print("\nStandardizing coordinates...")
    datasets = [standardize_coords(ds) for ds in datasets]
    print("\nStandardizing time coordinates...")
    datasets = [standardize_time(ds, time_slice) for ds in datasets]
    
    # Use RSDS model grid as target
    target_grid = datasets[0]
    
    # Print coordinate information
    print("\nTarget grid coordinates:")
    print(f"Latitude range: {target_grid.lat.values.min():.2f} to {target_grid.lat.values.max():.2f}")
    print(f"Longitude range: {target_grid.lon.values.min():.2f} to {target_grid.lon.values.max():.2f}")
    
    # Regrid datasets
    print("\nRegridding datasets...")
    for i in range(1, len(datasets)):
        print(f"\nRegridding {names[i]}...")
        print(f"Original coordinates: {list(datasets[i].dims)}")
        try:
            datasets[i] = datasets[i].interp(lat=target_grid.lat, lon=target_grid.lon)
            print(f"Successfully regridded to shape: {datasets[i].shape}")
        except Exception as e:
            print(f"Error regridding {names[i]}: {str(e)}")
            print(f"Dataset dimensions: {datasets[i].dims}")
            print(f"Dataset coordinates: {list(datasets[i].coords)}")
            raise
    
    # Process units
    print("\nProcessing units...")
    if hasattr(datasets[3], 'units') and datasets[3].attrs.get('units') == '(0 - 1)':
        datasets[3] = datasets[3] * 100
        datasets[3].attrs['units'] = '%'
    
    if hasattr(datasets[5], 'units') and datasets[5].attrs.get('units') == 'J m**-2':
        datasets[5] = datasets[5] / (24 * 3600)
        datasets[5].attrs['units'] = 'W m-2'
    
    # Print final shapes
    print("\nFinal dataset shapes:")
    for name, ds in zip(names, datasets):
        print(f"{name}: {ds.shape}")
    
    return tuple(datasets)


def calculate_biases(model, obs, name=""):
    """Calculate biases between model and observations."""
    print(f"\nCalculating biases for {name}...")
    
    # Convert to numpy arrays for easier handling
    model_data = model.values
    obs_data = obs.values
    
    # Calculate bias
    bias = model_data - obs_data
    
    # Create xarray DataArray with proper coordinates
    bias_da = xr.DataArray(bias, 
                          coords=model.coords,
                          dims=model.dims)
    
    # Calculate statistics safely
    valid_mask = ~np.isnan(bias)
    if np.any(valid_mask):
        min_val = np.nanmin(bias)
        max_val = np.nanmax(bias)
        mean_val = np.nanmean(bias)
        std_val = np.nanstd(bias)
        
        print(f"Bias statistics:")
        print(f"Min: {min_val:.2f}, Max: {max_val:.2f}")
        print(f"Mean: {mean_val:.2f}, Std: {std_val:.2f}")
        print(f"Valid (non-NaN) points: {np.sum(valid_mask)} of {bias.size}")
    else:
        print("Warning: No valid data points found in bias calculation")
    
    return bias_da


def calculate_sensitivity_maps(rsds_bias, tcc_bias, rsdscs_bias):
    """Calculate sensitivity maps using partial correlations."""
    print("Calculating sensitivity maps...")
    
    shape = rsds_bias.shape[1:]
    sensitivity_tcc = np.full(shape, np.nan)
    sensitivity_rsdscs = np.full(shape, np.nan)
    significance_tcc = np.zeros(shape, dtype=bool)
    significance_rsdscs = np.zeros(shape, dtype=bool)
    
    for i in range(shape[0]):
        if i % 10 == 0:
            print(f"Processing latitude {i}/{shape[0]}")
        for j in range(shape[1]):
            rs = rsds_bias[:, i, j]
            tcc = tcc_bias[:, i, j]
            rsdscs = rsdscs_bias[:, i, j]
            
            # Remove NaN values
            valid = ~np.isnan(rs) & ~np.isnan(tcc) & ~np.isnan(rsdscs)
            if np.sum(valid) >= 3:
                rs_valid = rs[valid]
                tcc_valid = tcc[valid]
                rsdscs_valid = rsdscs[valid]
                
                try:
                    # Calculate correlations and p-values for TCC
                    rxy_tcc, p_tcc = stats.pearsonr(rs_valid, tcc_valid)
                    rxz_tcc, _ = stats.pearsonr(rs_valid, rsdscs_valid)
                    ryz_tcc, _ = stats.pearsonr(tcc_valid, rsdscs_valid)
                    
                    # Calculate partial correlation for TCC
                    denom_tcc = np.sqrt((1 - rxz_tcc**2) * (1 - ryz_tcc**2))
                    if denom_tcc > 0:
                        r_tcc = (rxy_tcc - rxz_tcc*ryz_tcc) / denom_tcc
                        sensitivity_tcc[i, j] = r_tcc
                        significance_tcc[i, j] = p_tcc < 0.05  # Mark as significant if p < 0.05
                    
                    # Calculate correlations and p-values for Rs-clear
                    rxy_rsdscs, p_rsdscs = stats.pearsonr(rs_valid, rsdscs_valid)
                    rxz_rsdscs, _ = stats.pearsonr(rs_valid, tcc_valid)
                    ryz_rsdscs, _ = stats.pearsonr(rsdscs_valid, tcc_valid)
                    
                    # Calculate partial correlation for Rs-clear
                    denom_rsdscs = np.sqrt((1 - rxz_rsdscs**2) * (1 - ryz_rsdscs**2))
                    if denom_rsdscs > 0:
                        r_rsdscs = (rxy_rsdscs - rxz_rsdscs*ryz_rsdscs) / denom_rsdscs
                        sensitivity_rsdscs[i, j] = r_rsdscs
                        significance_rsdscs[i, j] = p_rsdscs < 0.05  # Mark as significant if p < 0.05
                        
                except Exception as e:
                    continue
    
    print("\nSensitivity calculation complete")
    print(f"Valid points in TCC sensitivity: {np.sum(~np.isnan(sensitivity_tcc))}")
    print(f"TCC sensitivity range: {np.nanmin(sensitivity_tcc):.3f} to {np.nanmax(sensitivity_tcc):.3f}")
    print(f"Significant points in TCC: {np.sum(significance_tcc)}")
    print(f"Valid points in Rs-clear sensitivity: {np.sum(~np.isnan(sensitivity_rsdscs))}")
    print(f"Rs-clear sensitivity range: {np.nanmin(sensitivity_rsdscs):.3f} to {np.nanmax(sensitivity_rsdscs):.3f}")
    print(f"Significant points in Rs-clear: {np.sum(significance_rsdscs)}")
    
    return sensitivity_tcc, sensitivity_rsdscs, significance_tcc, significance_rsdscs


def plot_sensitivity_maps(sensitivity_tcc, sensitivity_rsdscs, 
                         significance_tcc, significance_rsdscs, 
                         lats, lons, output_path='africa_sensitivity_maps.png'):
    """Create sensitivity maps figure with proper colormapping."""
    print("\nCreating sensitivity maps...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8),
                                  subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Create meshgrid for plotting
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    
    # Set up common parameters
    vmin, vmax = -0.8, 0.8
    cmap = plt.cm.RdBu_r
    
    # Plot TCC sensitivity with masked data
    mask_tcc = ~np.isnan(sensitivity_tcc)
    masked_tcc = np.ma.array(sensitivity_tcc, mask=~mask_tcc)
    cf1 = ax1.pcolormesh(lon_mesh, lat_mesh, masked_tcc,
                         cmap=cmap, vmin=vmin, vmax=vmax,
                         shading='auto', transform=ccrs.PlateCarree())
    ax1.set_title('a) ρ(ΔRs, ΔTCC)', fontsize=12)
    
    # Add stippling for significant points in TCC
    significant_points_tcc = np.where(significance_tcc & mask_tcc)
    ax1.scatter(lon_mesh[significant_points_tcc], lat_mesh[significant_points_tcc],
                color='black', s=1, alpha=0.5, transform=ccrs.PlateCarree())
    
    # Plot Rs-clear sensitivity with masked data
    mask_rsdscs = ~np.isnan(sensitivity_rsdscs)
    masked_rsdscs = np.ma.array(sensitivity_rsdscs, mask=~mask_rsdscs)
    cf2 = ax2.pcolormesh(lon_mesh, lat_mesh, masked_rsdscs,
                         cmap=cmap, vmin=vmin, vmax=vmax,
                         shading='auto', transform=ccrs.PlateCarree())
    ax2.set_title('b) ρ(ΔRs, ΔRs-clear)', fontsize=12)
    
    # Add stippling for significant points in Rs-clear
    significant_points_rsdscs = np.where(significance_rsdscs & mask_rsdscs)
    ax2.scatter(lon_mesh[significant_points_rsdscs], lat_mesh[significant_points_rsdscs],
                color='black', s=1, alpha=0.5, transform=ccrs.PlateCarree())
    
    # Add map features to both plots
    for ax in [ax1, ax2]:
        # Add land feature with white ocean
        ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=100)
        ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='black', linewidth=0.5)
        ax.coastlines(linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.6)
        ax.set_extent([-20, 55, -35, 40], crs=ccrs.PlateCarree())
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
    
    # Add colorbars
    for ax, cf in zip([ax1, ax2], [cf1, cf2]):
        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal',
                         label='',
                         extend='both', pad=0.1)
        cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    
    print(f"Saving figure to {output_path}...")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Plotting complete")


def main():
    """Main execution function."""
    try:
        # Load and process data
        print("Starting data processing...")
        rsds_model, rsds_obs, tcc_model, tcc_obs, rsdscs_model, rsdscs_obs = load_and_process_data()
        
        # Calculate biases
        print("\nCalculating biases...")
        rsds_bias = calculate_biases(rsds_model, rsds_obs, "RSDS")
        tcc_bias = calculate_biases(tcc_model, tcc_obs, "TCC")
        rsdscs_bias = calculate_biases(rsdscs_model, rsdscs_obs, "RSDSCS")
        
        # Calculate sensitivity maps
        print("\nCalculating sensitivity maps...")
        sensitivity_tcc, sensitivity_rsdscs, significance_tcc, significance_rsdscs = calculate_sensitivity_maps(
            rsds_bias, tcc_bias, rsdscs_bias
        )
        
        # Create plots
        print("\nCreating plots...")
        plot_sensitivity_maps(
            sensitivity_tcc, sensitivity_rsdscs,
            significance_tcc, significance_rsdscs,
            rsds_model.lat.values, rsds_model.lon.values
        )
        
        # Save sensitivity coefficients as NetCDF for later use
        print("\nSaving sensitivity coefficients...")
        sensitivity_ds = xr.Dataset({
            'sensitivity_tcc': (['lat', 'lon'], sensitivity_tcc),
            'sensitivity_rsdscs': (['lat', 'lon'], sensitivity_rsdscs),
            'significance_tcc': (['lat', 'lon'], significance_tcc),
            'significance_rsdscs': (['lat', 'lon'], significance_rsdscs)
        }, coords={
            'lat': rsds_model.lat.values,
            'lon': rsds_model.lon.values
        })
        
        sensitivity_ds.to_netcdf('sensitivity_coefficients.nc')
        print("Sensitivity coefficients saved to 'sensitivity_coefficients.nc'")
        
        print("\nSensitivity analysis complete!")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
