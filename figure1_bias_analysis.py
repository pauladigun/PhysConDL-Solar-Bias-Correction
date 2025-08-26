#!/usr/bin/env python3
"""
Figure 1: CMIP6 Model Bias Analysis
Spatial distribution of surface solar radiation bias across CMIP6 models compared to CMSAF-SARAH2 observations
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import gc
import math
from matplotlib.gridspec import GridSpec


def create_land_mask(data):
    """Create a simple land mask using lat/lon coordinates."""
    mask = ((data.lon >= -20) & (data.lon <= 55) & 
            (data.lat >= -35) & (data.lat <= 40))
    return mask


def load_and_process_cmsaf(file_path):
    """Load and process CMSAF data efficiently."""
    ds = xr.open_dataset(file_path)
    data = ds.SIS.sel(time=slice('1983-01-01', '2014-12-31'))
    mean_data = data.mean('time')
    land_mask = create_land_mask(mean_data)
    mean_data = mean_data.where(land_mask)
    ds.close()
    return mean_data


def load_and_process_model(file_path, var_name='rsds'):
    """Load and process model data efficiently."""
    try:
        ds = xr.open_dataset(file_path)
        # Handle different calendars
        if hasattr(ds.time.values[0], 'calendar') and '360' in ds.time.values[0].calendar:
            time_slice = slice('1983-01', '2014-12')
        else:
            time_slice = slice('1983-01-01', '2014-12-31')
        
        data = ds[var_name].sel(time=time_slice)
        mean_data = data.mean('time')
        land_mask = create_land_mask(mean_data)
        mean_data = mean_data.where(land_mask)
        ds.close()
        return mean_data
    except Exception as e:
        print(f"Error in load_and_process_model: {str(e)}")
        return None


def calculate_bias(model_data, obs_data, model_name):
    """Calculate bias between model and observations."""
    try:
        if model_data is None:
            return None
            
        print(f"Processing {model_name}...")
        
        # Regrid observations to model grid
        regridded_obs = obs_data.interp_like(model_data)
        
        # Calculate bias
        bias = model_data - regridded_obs
        
        # Calculate area-weighted mean bias
        weights = np.cos(np.deg2rad(model_data.lat))
        valid_mask = ~np.isnan(bias)
        weighted_mean_bias = float((bias.where(valid_mask) * weights).sum() / 
                                 (weights.where(valid_mask)).sum())
        
        # Calculate significance
        valid_data = bias.values[~np.isnan(bias.values)]
        t_stat, p_value = stats.ttest_1samp(valid_data, 0.0)
        
        return {
            'model': model_name,
            'bias': weighted_mean_bias,
            'significant': p_value < 0.05,
            'p_value': float(p_value),
            'spatial_bias': bias
        }
        
    except Exception as e:
        print(f"Error processing {model_name}: {str(e)}")
        return None


def create_spatial_plots(bias_results):
    """Create spatial plots with bias values in lower left corner."""
    # Calculate layout for spatial plots
    n_models = len([r for r in bias_results if r])
    n_cols = min(5, n_models)
    n_rows = math.ceil(n_models / n_cols)
    
    # Set up figure
    fig = plt.figure(figsize=(20, 12))
    
    # Create gridspec for layout with adjusted spacing
    gs = GridSpec(n_rows, n_cols, figure=fig)
    gs.update(top=0.95, bottom=0.05, left=0.05, right=0.85, wspace=-0.5, hspace=0.2)
    
    # Set fixed min/max values for colorbar
    vmin = -20
    vmax = 20
    
    # Create colormap
    cmap = plt.cm.RdBu_r
    
    # Create spatial plots
    first_im = None
    valid_idx = 0
    
    for result in bias_results:
        if not result:
            continue
            
        row = valid_idx // n_cols
        col = valid_idx % n_cols
        
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        ax.set_extent([-18, 52, -35, 38], crs=ccrs.PlateCarree())
        
        # Plot the data with fixed vmin/vmax
        im = ax.contourf(result['spatial_bias'].lon, result['spatial_bias'].lat,
                        result['spatial_bias'],
                        levels=np.linspace(vmin, vmax, 13),
                        cmap=cmap,
                        transform=ccrs.PlateCarree(),
                        extend='both')
        
        if first_im is None:
            first_im = im
        
        # Add coastlines and borders
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=100)
        
        # Add model name as title
        ax.set_title(result['model'], fontsize=13, fontweight='bold', pad=5)
        
        # Create bias text with value and asterisk if significant
        bias_text = f"{result['bias']:.1f} W/m²"
        if result['significant']:
            bias_text += '*'
            
        # Add bias value in lower left corner with white background for readability
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7)
        
        # Position in lower left corner
        ax.text(-15, -30, bias_text,
                transform=ccrs.PlateCarree(),
                fontsize=10,
                fontweight='bold',
                zorder=101,
                bbox=bbox_props,
                horizontalalignment='left',
                verticalalignment='bottom')
        
        valid_idx += 1
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.8, 0.1, 0.02, 0.8])
    cbar = plt.colorbar(first_im, cax=cbar_ax)
    cbar.set_label('Bias (W/m²)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    return fig


def create_bar_plot(bias_results):
    """Create separate bar plot."""
    # Set up figure
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(111)
    
    # Prepare bar plot data
    models = [r['model'] for r in bias_results if r]
    biases = [r['bias'] for r in bias_results if r]
    significant = [r['significant'] for r in bias_results if r]
    
    # Sort by absolute bias, but keep MME at the end
    special_models = ['MME']
    regular_indices = [i for i, m in enumerate(models) if m not in special_models]
    special_indices = [i for i, m in enumerate(models) if m in special_models]
    
    # Sort regular models by absolute bias
    regular_pairs = [(abs(biases[i]), i) for i in regular_indices]
    regular_pairs.sort()
    sorted_indices = [i for _, i in regular_pairs] + special_indices
    
    models = [models[i] for i in sorted_indices]
    biases = [biases[i] for i in sorted_indices]
    significant = [significant[i] for i in sorted_indices]
    
    # Create bars with different colors for regular models vs MME
    colors = ['skyblue' if model not in special_models else 'lightcoral' 
              for model in models]
    
    bars = ax.bar(range(len(models)), biases, color=colors)
    
    # Add hatching for significant results
    for bar, sig in zip(bars, significant):
        if sig:
            bar.set_hatch('///')
    
    # Customize bar plot
    ax.set_ylabel('Bias (W/m²)', fontsize=12)
    ax.set_xlabel('Models', fontsize=12)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Main execution function."""
    print("Loading CMSAF data...")
    obs_mean = load_and_process_cmsaf('observations.nc')
    
    model_files = [
        'ACCESS-CM2.nc',
        'ACCESS-ESM1-5.nc',
        'AWI-CM-1-1-MR.nc',
        'BCC-CSM2-MR.nc',
        'CanESM5.nc',
        'CESM2.nc',
        'CESM2-WACCM.nc',
        'EC-Earth3.nc',
        'FGOALS-f3-L.nc',
        'FGOALS-g3.nc',
        'GFDL-ESM4.nc',
        'INM-CM4-8.nc',
        'INM-CM5-0.nc',
        'IPSL-CM6A-LR.nc',
        'KACE-1-0-G.nc',
        'MPI-ESM1-2-HR.nc',
        'NorESM2-LM.nc',
        'NorESM2-MM.nc',
        'EC-Earth3-Veg.nc',
        'MME.nc'
    ]
    
    bias_results = []
    
    # Process each model
    for model_file in model_files:
        try:
            if 'MME' in model_file:
                model_name = 'MME'
            else:
                model_name = model_file.replace('.nc', '')
            
            model_mean = load_and_process_model(model_file)
            result = calculate_bias(model_mean, obs_mean, model_name)
            
            if result:
                bias_results.append(result)
            
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {model_file}: {str(e)}")
    
    if bias_results:
        # Create and save spatial plots
        spatial_fig = create_spatial_plots(bias_results)
        plt.savefig('figure1_spatial_biases.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create and save bar plot separately
        bar_fig = create_bar_plot(bias_results)
        plt.savefig('figure1_bar_biases.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print results summary
        print("\nBias Results Summary:")
        print("-" * 70)
        print(f"{'Model':<20} {'Bias (W/m²)':<15} {'Significance':<15}")
        print("-" * 70)
        
        # Sort results by absolute bias
        sorted_results = sorted(bias_results, key=lambda x: abs(x['bias']))
        
        for result in sorted_results:
            sig_str = "significant" if result['significant'] else "not significant"
            print(f"{result['model']:<20} {result['bias']:>6.2f}        {sig_str:<15}")
        
        print(f"\nPlots saved as 'figure1_spatial_biases.png' and 'figure1_bar_biases.png'")
    else:
        print("No valid results to plot.")


if __name__ == "__main__":
    main()
