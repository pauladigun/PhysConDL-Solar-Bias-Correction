#!/usr/bin/env python3
"""
Figure 7: Regional Solar Radiation Anomalies
Projected changes in surface downwelling shortwave radiation (Rs) anomalies across African regions 
under different climate scenarios with end-of-century distributions
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.font_manager as font_manager
import geopandas as gpd
from shapely.geometry import Point, Polygon
import regionmask
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set variables
period_future = slice('2015', '2100')
period_past = slice('1995', '2014')
baseline_period = slice('1995', '2014')

# Define regions
regions = {
    'Africa': (-35, 40, -20, 55),  # lat_min, lat_max, lon_min, lon_max
    'Equatorial': (-10, 10, -20, 55),
    'North Africa': (10, 38, -20, 55),
    'South Africa': (-35, -10, -20, 55)
}


def load_africa_mask():
    """Load Africa shapefile for masking."""
    try:
        # Load the shapefile for Africa
        africa_shapefile = "africa_shapefile.shp"
        africa_shape = gpd.read_file(africa_shapefile)
        
        print(f"Successfully loaded Africa shapefile with {len(africa_shape)} features")
        return africa_shape
    except Exception as e:
        print(f"Error loading Africa shapefile: {e}")
        print("Proceeding without land mask - using rectangular domain")
        return None


def create_region_land_mask(ds, var_name, africa_shape, region_name):
    """Create a mask for a given dataset using the Africa shapefile and regional boundaries."""
    try:
        # Get region boundaries
        lat_min, lat_max, lon_min, lon_max = regions[region_name]
        
        # Check which coordinate names are used in this dataset
        if 'lat' in ds.dims and 'lon' in ds.dims:
            lat_name, lon_name = 'lat', 'lon'
        elif 'latitude' in ds.dims and 'longitude' in ds.dims:
            lat_name, lon_name = 'latitude', 'longitude'
        else:
            print("Could not identify lat/lon coordinates, using bounding box")
            return ds[var_name].sel(**{lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, lon_max)})
        
        # Extract lat/lon coordinates
        lats = ds[lat_name].values
        lons = ds[lon_name].values
        
        # Create mask using regionmask
        mask = regionmask.mask_geopandas(africa_shape, lons, lats)
        
        # Select the variable and apply the mask - NaN values will be outside Africa
        masked_data = ds[var_name].where(~np.isnan(mask))
        
        # Restrict to the regional bounding box
        masked_data = masked_data.sel(**{lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, lon_max)})
        
        print(f"Successfully created land mask for {region_name} - masked out non-land regions")
        return masked_data
    
    except Exception as e:
        print(f"Error creating land mask for {region_name}: {e}")
        print("Falling back to rectangular domain selection")
        return ds[var_name].sel(**{lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, lon_max)})


def process_scenario_files_by_region(files, region_name, africa_shape=None):
    """Process scenario files for a specific region."""
    data_by_model = {}
    
    for file in files:
        try:
            # Extract model name from filename
            model = file.replace('.nc', '').split('_')[-1]
            
            # Open dataset
            ds = xr.open_dataset(file)
            
            # Get variable name (rsds)
            var_name = None
            for var in ds.data_vars:
                if var in ['rsds', 'rad', 'swr'] or 'rs' in var.lower():
                    var_name = var
                    break
            
            if var_name is None:
                print(f"Could not find RSDS variable in {file}")
                continue
            
            # Determine lat/lon variable names
            if 'lat' in ds.dims:
                lat_name, lon_name = 'lat', 'lon'
            else:
                lat_name, lon_name = 'latitude', 'longitude'
                
            # Extract region domain with proper land masking
            try:
                if africa_shape is not None:
                    # Use the shapefile to mask out non-land regions
                    ds_domain = create_region_land_mask(ds, var_name, africa_shape, region_name)
                else:
                    # Fallback to simple bounding box selection
                    lat_min, lat_max, lon_min, lon_max = regions[region_name]
                    ds_domain = ds[var_name].sel({lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, lon_max)})
                
                # Average over region domain (land only if mask was applied)
                ds_mean = ds_domain.mean(dim=[d for d in ds_domain.dims if d != 'time'])
                
                # Try converting to pandas series with datetime index
                try:
                    # Handle both regular datetime and cftime datetime objects
                    if isinstance(ds_mean.time.values[0], (np.datetime64, pd.Timestamp)):
                        # Regular datetime objects
                        times = pd.to_datetime(ds_mean.time.values)
                    else:
                        # cftime datetime objects - convert them differently
                        times = []
                        for t in ds_mean.time.values:
                            try:
                                # Try to extract year, month, day
                                year = t.year
                                month = t.month
                                day = t.day
                                # Create pandas datetime
                                times.append(pd.Timestamp(year=year, month=month, day=day))
                            except Exception as e:
                                print(f"Error converting time value for {model}: {e}")
                                # Skip this model if time conversion fails
                                raise
                        times = pd.DatetimeIndex(times)
                    
                    values = ds_mean.values
                    
                    # Create pandas series
                    ts = pd.Series(values, index=times)
                    
                    # Resample to yearly averages
                    yearly = ts.resample('YE').mean()
                    
                    # Store in dictionary
                    data_by_model[model] = yearly
                    print(f"Successfully processed {model} data for {region_name} (land only)")
                except Exception as e:
                    print(f"Warning: Could not convert times for {model}, skipping: {e}")
                    
            except Exception as e:
                print(f"Error selecting domain for {file}: {e}")
                
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    return data_by_model


def create_and_plot_regional_anomalies(historical_by_region, scenarios, scenario_by_region, scenario_data_by_region):
    """Create and plot anomalies for 4 regions in a 2x2 grid with boxplots."""
    print("\nCreating matplotlib figure with regional anomalies in a 2x2 grid with end-of-century boxplots...")
    
    # Set up bold font for all text
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['figure.titleweight'] = 'bold'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['figure.titlesize'] = 22
    
    # Set up the figure with a 2x2 grid layout
    fig = plt.figure(figsize=(30, 12))
    
    # Create a gridspec for the layout (2x4 to accommodate time series and boxplots)
    gs = fig.add_gridspec(2, 4, width_ratios=[3, 1, 3, 1], height_ratios=[1, 1], wspace=0.2, hspace=0.3)

    # Colors for scenarios
    colors = {
        'historical': 'gray',
        'ssp126': 'indigo',
        'ssp245': 'royalblue',
        'ssp370': 'orange',
        'ssp585': 'red'
    }
    
    # Region labels for subplot titles
    region_labels = {
        'Africa': '(a) Africa',
        'Equatorial': '(b) Equatorial Africa',
        'North Africa': '(c) North Africa',
        'South Africa': '(d) South Africa'
    }
    
    # Process data for each region
    for i, region_name in enumerate(regions.keys()):
        if region_name in historical_by_region and region_name in scenario_data_by_region:
            # Map the region index to the correct gridspec positions
            if i == 0:   # Africa (top-left)
                ax_ts = fig.add_subplot(gs[0, 0])
                ax_box = fig.add_subplot(gs[0, 1])
            elif i == 1:  # Equatorial (top-right)
                ax_ts = fig.add_subplot(gs[0, 2])
                ax_box = fig.add_subplot(gs[0, 3])
            elif i == 2:  # North Africa (bottom-left)
                ax_ts = fig.add_subplot(gs[1, 0])
                ax_box = fig.add_subplot(gs[1, 1])
            elif i == 3:  # South Africa (bottom-right)
                ax_ts = fig.add_subplot(gs[1, 2])
                ax_box = fig.add_subplot(gs[1, 3])
            
            # Get historical data for this region
            hist_by_model = historical_by_region[region_name]
            
            # Get all years for historical period
            all_years = set()
            for model_data in hist_by_model.values():
                all_years.update(model_data.index.year)
            
            # Filter years to our time period
            past_years = sorted([y for y in all_years if y >= 1980 and y <= 2014])
            
            # Create historical dataframe
            hist_df = pd.DataFrame(index=past_years)
            for model, data in hist_by_model.items():
                values = []
                for year in past_years:
                    year_data = data[data.index.year == year]
                    if not year_data.empty:
                        values.append(year_data.mean())
                    else:
                        values.append(np.nan)
                hist_df[model] = values
            
            # Calculate climatology
            clim_models = hist_df.loc[1995:2014].mean()
            
            # Calculate anomalies
            hist_anomalies = hist_df.subtract(clim_models, axis=1)
            
            # Plot historical anomalies
            hist_mean = hist_anomalies.mean(axis=1)
            hist_std = hist_anomalies.std(axis=1)
            hist_upper = hist_mean + hist_std
            hist_lower = hist_mean - hist_std
            
            # Plot the mean for anomalies
            ax_ts.plot(np.array(hist_anomalies.index.values), np.array(hist_mean.values), 
                   color=colors['historical'], linewidth=2.5, label='historical')
            
            # Plot the shaded region for ±1 standard deviation range
            ax_ts.fill_between(np.array(hist_anomalies.index.values), 
                           np.array(hist_lower.values), 
                           np.array(hist_upper.values), 
                           color=colors['historical'], alpha=0.2)
            
            # Process future scenarios for anomalies
            anomaly_boxplot_data = {}  # For end-of-century boxplot
            anomaly_std_values = {}    # For uncertainty values
            
            for scenario in scenarios:
                if scenario in scenario_data_by_region[region_name]:
                    scenario_anomalies = scenario_data_by_region[region_name][scenario]
                    
                    # Calculate statistics for anomalies
                    scen_mean = scenario_anomalies.mean(axis=1)
                    scen_std = scenario_anomalies.std(axis=1)
                    scen_upper = scen_mean + scen_std
                    scen_lower = scen_mean - scen_std
                    
                    # Plot mean for anomalies
                    ax_ts.plot(np.array(scenario_anomalies.index.values), 
                          np.array(scen_mean.values), 
                          color=colors[scenario], linewidth=2.5, label=scenario)
                    
                    # Plot shaded region for anomalies
                    ax_ts.fill_between(np.array(scenario_anomalies.index.values), 
                                  np.array(scen_lower.values), 
                                  np.array(scen_upper.values), 
                                  color=colors[scenario], alpha=0.2)
                    
                    # Collect end-of-century data for boxplot
                    end_century = scenario_anomalies.loc[2081:2100]
                    anomaly_boxplot_data[scenario] = end_century.mean().values
                    
                    # Calculate standard deviation for all scenarios
                    scenario_std = np.std(end_century.mean().values)
                    anomaly_std_values[scenario] = scenario_std
            
            # Add reference lines
            ax_ts.axvline(x=2014.5, color='black', linestyle='-', linewidth=1.5)
            ax_ts.axhline(y=0, color='gray', linestyle='-', linewidth=0.7, alpha=0.7)
            
            # Format plot
            ax_ts.set_title(region_labels[region_name], fontsize=20, fontweight='bold')
            ax_ts.set_ylabel('Rs Anomaly (W/m²)', fontsize=18, fontweight='bold')
            ax_ts.set_xlabel('Year', fontsize=18, fontweight='bold')
            
            # Only add legend to the first subplot
            if i == 0:
                ax_ts.legend(loc='upper left', frameon=True, fontsize=14)
            
            # Make axis labels and tick labels bold
            for label in ax_ts.get_xticklabels() + ax_ts.get_yticklabels():
                label.set_fontweight('bold')
                label.set_fontsize(14)
            
            # Create boxplot for anomalies
            scenario_names = []
            anomaly_boxplot_values = []
            
            for scenario in scenarios:
                if scenario in anomaly_boxplot_data:
                    scenario_names.append(scenario)
                    anomaly_boxplot_values.append(anomaly_boxplot_data[scenario])
            
            # Create boxplot for anomalies
            bp_anomalies = ax_box.boxplot(anomaly_boxplot_values, patch_artist=True, vert=True, showfliers=True)
            
            # Customize boxplot colors for anomalies
            for i_box, box in enumerate(bp_anomalies['boxes']):
                box.set(facecolor=colors[scenario_names[i_box]], alpha=0.6)
                box.set(edgecolor=colors[scenario_names[i_box]])
            
            for i_whisker, whisker in enumerate(bp_anomalies['whiskers']):
                whisker.set(color=colors[scenario_names[i_whisker//2]])
            
            for i_cap, cap in enumerate(bp_anomalies['caps']):
                cap.set(color=colors[scenario_names[i_cap//2]])
            
            for i_median, median in enumerate(bp_anomalies['medians']):
                median.set(color='black', linewidth=2)
            
            for i_flier, flier in enumerate(bp_anomalies['fliers']):
                flier.set(marker='o', markerfacecolor=colors[scenario_names[i_flier]], 
                       markeredgecolor='none', markersize=4, alpha=0.7)
            
            # Format boxplot
            ax_box.set_title('2081-2100', fontsize=18, fontweight='bold')
            ax_box.set_xticklabels(scenario_names, rotation=90, fontsize=12, fontweight='bold')
            
            # Make axis tick labels bold
            for label in ax_box.get_xticklabels() + ax_box.get_yticklabels():
                label.set_fontweight('bold')
                label.set_fontsize(12)
            
            # Add uncertainty annotations for all scenarios on anomalies boxplot
            for i_scen, scenario in enumerate(scenario_names):
                # Get position
                x_pos = i_scen + 1  # Box positions are 1-based
                
                # Calculate median value for positioning
                scenario_median = np.median(anomaly_boxplot_data[scenario])
                std_val = anomaly_std_values[scenario]
                
                # Draw arrow and annotation
                arrow_length = std_val
                
                # Add vertical double arrow
                ax_box.annotate(
                    "",
                    xy=(x_pos + 0.3, scenario_median - arrow_length),
                    xytext=(x_pos + 0.3, scenario_median + arrow_length),
                    arrowprops=dict(
                        arrowstyle="<->",
                        color=colors[scenario],
                        lw=2
                    )
                )
                
                # Add text annotation with smaller font size for boxplots
                ax_box.annotate(
                    f"±{std_val:.1f}",
                    xy=(x_pos + 0.35, scenario_median),  # Position to the right of arrow
                    xytext=(x_pos + 0.4, scenario_median),  # Text position
                    ha='left', va='center',
                    fontsize=10,
                    fontweight='bold',
                    color=colors[scenario]
                )
            
            # Extend x-axis to accommodate the annotations
            ax_box.set_xlim(0.5, len(scenario_names) + 1.5)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    filename = 'figure7_regional_rsds_anomalies.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # Verify file was created
    if os.path.exists(filename):
        file_size = os.path.getsize(filename) / 1024  # Size in KB
        print(f"\nFigure saved as {filename}")
        print(f"File size: {file_size:.2f} KB")
        return True
    
    return False


def main():
    """Main execution function."""
    print("Processing Africa regional RSDS data...")

    # Load Africa shapefile for masking
    africa_shape = load_africa_mask()

    # Process files for all scenarios and regions
    print("\nProcessing all NetCDF files for both historical and future periods...")
    scenario_files = {}
    scenario_by_region_by_model = {}
    historical_by_region_by_model = {}
    scenario_data_by_region = {}
    scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

    # Initialize data structures for each region
    for region_name in regions.keys():
        historical_by_region_by_model[region_name] = {}
        scenario_by_region_by_model[region_name] = {}
        scenario_data_by_region[region_name] = {}
        for scenario in scenarios:
            scenario_by_region_by_model[region_name][scenario] = {}
            scenario_data_by_region[region_name][scenario] = {}

    # Process each scenario's files and extract data for each region
    for scenario in scenarios:
        print(f"Processing {scenario} files...")
        scenario_files[scenario] = sorted(glob.glob(f"*{scenario}*.nc"))
        print(f"Found {len(scenario_files[scenario])} files for {scenario}")
        
        # Process files for each region
        for region_name in regions.keys():
            print(f"Processing {region_name} for {scenario}...")
            
            # Process files to get complete time series with land mask for this region
            all_data_by_model = process_scenario_files_by_region(scenario_files[scenario], region_name, africa_shape)
            
            # Now split the data into historical and future periods by model
            for model, data in all_data_by_model.items():
                # Extract historical period
                historical_data = data[(data.index.year >= 1980) & (data.index.year <= 2014)]
                future_data = data[(data.index.year >= 2015) & (data.index.year <= 2100)]
                
                # Store historical data
                if model not in historical_by_region_by_model[region_name]:
                    historical_by_region_by_model[region_name][model] = historical_data
                
                # Store future data
                scenario_by_region_by_model[region_name][scenario][model] = future_data
            
            # Process data for analysis (anomalies)
            # Get all future years
            future_years = set()
            for model, data in scenario_by_region_by_model[region_name][scenario].items():
                future_years.update(data.index.year)
            
            future_years = sorted([y for y in future_years if y >= 2015 and y <= 2100])
            
            # Create DataFrame for future data
            future_df = pd.DataFrame(index=future_years)
            for model, data in scenario_by_region_by_model[region_name][scenario].items():
                values = []
                for year in future_years:
                    year_data = data[data.index.year == year]
                    if not year_data.empty:
                        values.append(year_data.mean())
                    else:
                        values.append(np.nan)
                future_df[model] = values
            
            # Calculate climatology
            past_years = sorted(set([y for model_data in historical_by_region_by_model[region_name].values() for y in model_data.index.year if y >= 1980 and y <= 2014]))
            hist_df = pd.DataFrame(index=past_years)
            for model, data in historical_by_region_by_model[region_name].items():
                values = []
                for year in past_years:
                    year_data = data[data.index.year == year]
                    if not year_data.empty:
                        values.append(year_data.mean())
                    else:
                        values.append(np.nan)
                hist_df[model] = values
            
            clim_models = hist_df.loc[1995:2014].mean()
            
            # Calculate anomalies - only using common models
            common_models = list(set(future_df.columns).intersection(set(clim_models.index)))
            
            if common_models:
                print(f"Found {len(common_models)} common models for {region_name} in {scenario}")
                anomalies_df = pd.DataFrame(index=future_years)
                
                for model in common_models:
                    anomalies_df[model] = future_df[model] - clim_models[model]
                
                # Store in dictionary
                scenario_data_by_region[region_name][scenario] = anomalies_df
            else:
                print(f"No common models found for {region_name} in {scenario}")

    # Create the plot
    success = create_and_plot_regional_anomalies(historical_by_region_by_model, scenarios, scenario_by_region_by_model, scenario_data_by_region)
    if success:
        print("Successfully created the regional anomalies plot!")
    else:
        print("Failed to create the regional anomalies plot.")


if __name__ == "__main__":
    main()
