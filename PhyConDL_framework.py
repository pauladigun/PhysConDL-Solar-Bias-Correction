#!/usr/bin/env python3
"""
Physics-Constrained Deep Learning Framework for Solar Radiation Bias Correction
This is an incomplete implementation that preserves the physics and architecture but
removes critical components needed for successful execution.
"""

import numpy as np
import xarray as xr
import tensorflow as tf
import os
import gc
import glob
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Configure tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU mode
np.seterr(all='ignore')

# ============================================================================
# LAMBDA PARAMETERS (from paper)
# ============================================================================
LAMBDA_1 = 0.6  # Empirical loss weight
LAMBDA_2 = 0.3  # Physical loss weight
LAMBDA_3 = 0.1  # Regularization weight

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_model_data(rsds_file):
    """Load model RSDS data from NetCDF file."""
    with xr.open_dataset(rsds_file, decode_times=False) as ds:
        if 'rsds' in ds.data_vars:
            rsds_model = ds['rsds'].values
            model_dims = ds['rsds'].dims
            model_coords = {dim: ds[dim].values for dim in model_dims}
            model_attrs = ds['rsds'].attrs
        else:
            var_name = list(ds.data_vars)[0]
            rsds_model = ds[var_name].values
            model_dims = ds[var_name].dims
            model_coords = {dim: ds[dim].values for dim in model_dims}
            model_attrs = ds[var_name].attrs
    
    return rsds_model.astype(np.float32), model_dims, model_coords, model_attrs

def load_physics_constraints(rsdscs_file, clt_file):
    """Load clear-sky radiation and cloud fraction data."""
    with xr.open_dataset(rsdscs_file, decode_times=False) as ds:
        if 'rsdscs' in ds.data_vars:
            rsdscs = ds['rsdscs'].values
            rsdscs_dims = ds['rsdscs'].dims
            rsdscs_coords = {dim: ds[dim].values for dim in rsdscs_dims}
        else:
            var_name = list(ds.data_vars)[0]
            rsdscs = ds[var_name].values
            rsdscs_dims = ds[var_name].dims
            rsdscs_coords = {dim: ds[dim].values for dim in rsdscs_dims}
    
    with xr.open_dataset(clt_file, decode_times=False) as ds:
        if 'clt' in ds.data_vars:
            clt = ds['clt'].values
            clt_dims = ds['clt'].dims
            clt_coords = {dim: ds[dim].values for dim in clt_dims}
        else:
            var_name = list(ds.data_vars)[0]
            clt = ds[var_name].values
            clt_dims = ds[var_name].dims
            clt_coords = {dim: ds[dim].values for dim in clt_dims}
    
    return (rsdscs.astype(np.float32), clt.astype(np.float32), 
            rsdscs_dims, rsdscs_coords, clt_dims, clt_coords)

def load_obs_data():
    """Load SARAH-2.1 observation data efficiently in chunks."""
    try:
        with xr.open_dataset('sis_cmsaf.nc', decode_times=False, chunks={'time': 1}) as ds:
            if 'SIS' in ds.data_vars:
                var_name = 'SIS'
            else:
                var_name = list(ds.data_vars)[0]
            
            obs_dims = ds[var_name].dims
            time_dim = obs_dims[0]
            n_time = ds.dims[time_dim]
            
            obs_lats = ds['lat'].values
            obs_lons = ds['lon'].values
            
            shape = (n_time, len(obs_lats), len(obs_lons))
            rsds_obs = np.zeros(shape, dtype=np.float32)
            
            chunk_size = 12
            for start_idx in range(0, n_time, chunk_size):
                end_idx = min(start_idx + chunk_size, n_time)
                chunk = ds[var_name].isel({time_dim: slice(start_idx, end_idx)}).values
                rsds_obs[start_idx:end_idx] = chunk
                gc.collect()
            
            obs_coords = {}
            for dim in obs_dims:
                if dim == time_dim:
                    obs_coords[dim] = np.arange(n_time)
                else:
                    obs_coords[dim] = ds[dim].values
            
            obs_attrs = ds[var_name].attrs
            
            return rsds_obs, obs_dims, obs_coords, obs_attrs
            
    except Exception as e:
        print(f"Error loading observation data: {e}")
        raise

def regrid_data_to_model(source_data, source_lats, source_lons, target_lats, target_lons):
    """Regrid data to model grid using nearest neighbor."""
    regridded = np.full((source_data.shape[0], len(target_lats), len(target_lons)), 
                        np.nan, dtype=np.float32)
    
    lat_idx = {i: np.argmin(np.abs(source_lats - lat)) for i, lat in enumerate(target_lats)}
    lon_idx = {i: np.argmin(np.abs(source_lons - lon)) for i, lon in enumerate(target_lons)}
    
    for t in range(source_data.shape[0]):
        for i, lat in enumerate(target_lats):
            for j, lon in enumerate(target_lons):
                regridded[t, i, j] = source_data[t, lat_idx[i], lon_idx[j]]
    
    return regridded

# ============================================================================
# STAGE 1: PHYSICS-INFORMED NUDGING
# ============================================================================

def create_nudged_simulation(rsds_model, rsds_obs, rsdscs, clt):
    """
    Stage 1: Physics-informed nudging with cloud-dependent weighting.
    Weight formula: weight = 0.2 + 0.4 × (TCC / 100)
    """
    n_time, n_lat, n_lon = rsds_model.shape
    nudged_rsds = np.copy(rsds_model)
    
    for t in range(n_time):
        clt_t = clt[t].copy()
        clt_t[np.isnan(clt_t)] = 0.0
        
        weight = 0.2 + 0.4 * (clt_t / 100.0)
        
        valid = ~np.isnan(rsds_model[t]) & ~np.isnan(rsds_obs[t])
        if np.sum(valid) > 0:
            nudged_rsds[t][valid] = ((1 - weight[valid]) * rsds_model[t][valid] + 
                                      weight[valid] * rsds_obs[t][valid])
            
            if rsdscs is not None:
                rsdscs_t = rsdscs[t].copy()
                rsdscs_valid = ~np.isnan(rsdscs_t) & valid
                if np.sum(rsdscs_valid) > 0:
                    nudged_rsds[t][rsdscs_valid] = np.minimum(
                        nudged_rsds[t][rsdscs_valid], 
                        rsdscs_t[rsdscs_valid]
                    )
    
    return nudged_rsds

# ============================================================================
# STAGE 2: SPECTRAL CORRECTION
# ============================================================================

def apply_spectral_correction(model_rsds, nudged_rsds):
    """
    Stage 2: Fourier-domain spectral correction.
    FFT_corrected = |FFT_model| × exp(i × phase(FFT_nudged))
    """
    n_time, n_lat, n_lon = model_rsds.shape
    corrected_rsds = np.copy(nudged_rsds)
    
    for t in range(n_time):
        for i in range(n_lat):
            model_row = model_rsds[t, i]
            nudged_row = nudged_rsds[t, i]
            valid_row = ~np.isnan(model_row) & ~np.isnan(nudged_row)
            
            if np.sum(valid_row) < 10:
                continue
            
            model_valid = model_row[valid_row]
            nudged_valid = nudged_row[valid_row]
            
            try:
                model_fft = np.fft.rfft(model_valid)
                nudged_fft = np.fft.rfft(nudged_valid)
                
                model_amp = np.abs(model_fft)
                nudged_phase = np.angle(nudged_fft)
                corrected_fft = model_amp * np.exp(1j * nudged_phase)
                
                corrected_valid = np.fft.irfft(corrected_fft, n=len(model_valid))
                
                corrected_row = corrected_rsds[t, i].copy()
                corrected_row[valid_row] = corrected_valid
                corrected_rsds[t, i] = corrected_row
            except Exception:
                pass
    
    corrected_rsds[corrected_rsds < 0] = 0
    
    return corrected_rsds

# ============================================================================
# STAGE 3: NEURAL NETWORK WITH LAMBDA-WEIGHTED LOSS
# ============================================================================

def build_physics_informed_ml_model_with_lambdas(input_shape, lambda1=0.6, lambda2=0.3, lambda3=0.1, 
                                                  scaling_factor=15.0):
    """
    Build physics-constrained neural network with lambda-weighted loss function.
    
..........
Missing 

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================


# ============================================================================
# EVALUATION AND SAVING
# ============================================================================

def calculate_rmse(predicted, reference, mask=None):
    """Calculate Root Mean Square Error."""
    if mask is None:
        mask = ~np.isnan(predicted) & ~np.isnan(reference)
    
    if np.sum(mask) == 0:
        return np.nan
    
    error = predicted[mask] - reference[mask]
    return np.sqrt(np.mean(error**2))

def save_netcdf(data, coords, attrs, filename, var_name='rsds'):
    """Save corrected data as NetCDF file."""
    ds = xr.Dataset(
        data_vars={var_name: (list(coords.keys()), data, attrs)},
        coords=coords
    )
    
    ds.attrs['description'] = 'Physics-constrained bias-corrected solar radiation'
    ds.attrs['creation_date'] = datetime.now().isoformat()
    ds.attrs['method'] = 'Three-stage cascade with lambda-weighted loss'
    ds.attrs['lambda_parameters'] = f'λ₁={LAMBDA_1}, λ₂={LAMBDA_2}, λ₃={LAMBDA_3}'
    
    ds.to_netcdf(filename)
    print(f"Saved: {filename}")

# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_model(model_name, rsds_file, rsdscs_file, clt_file, obs_data):
    """
    Complete three-stage bias correction pipeline with lambda-weighted loss.
    
    Loss Function: L = λ₁L_empirical + λ₂L_physical + λ₃L_regularization
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING MODEL: {model_name}")
    print(f"{'='*80}")
    
    rsds_obs, obs_dims, obs_coords, obs_attrs = obs_data
    obs_lats = obs_coords['lat']
    obs_lons = obs_coords['lon']
    
    print(f"Loading data...")
    try:
        rsds_model, model_dims, model_coords, model_attrs = load_model_data(rsds_file)
        model_lats = model_coords['lat']
        model_lons = model_coords['lon']
        
        rsdscs, clt, rsdscs_dims, rsdscs_coords, clt_dims, clt_coords = load_physics_constraints(
            rsdscs_file, clt_file
        )
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return None
    
    if (obs_lats.shape[0] != model_lats.shape[0] or obs_lons.shape[0] != model_lons.shape[0]):
        print(f"Regridding observations...")
        rsds_obs_grid = regrid_data_to_model(rsds_obs, obs_lats, obs_lons, model_lats, model_lons)
    else:
        rsds_obs_grid = rsds_obs
    
    min_time = min(rsds_model.shape[0], rsdscs.shape[0], clt.shape[0], rsds_obs_grid.shape[0])
    rsds_model = rsds_model[:min_time]
    rsdscs = rsdscs[:min_time]
    clt = clt[:min_time]
    rsds_obs_grid = rsds_obs_grid[:min_time]
    
    print(f"\nSTAGE 1: Physics-informed nudging...")
    nudged_rsds = create_nudged_simulation(rsds_model, rsds_obs_grid, rsdscs, clt)
    
    print(f"STAGE 2: Spectral correction...")
    spectral_corrected = apply_spectral_correction(rsds_model, nudged_rsds)
    
    print(f"STAGE 3: Neural network with lambda-weighted loss...")
    print(f"  Preparing training data...")
    training_data = prepare_training_data(rsds_model, rsds_obs_grid, rsdscs, clt)
    
    print(f"  Building model with λ₁={LAMBDA_1}, λ₂={LAMBDA_2}, λ₃={LAMBDA_3}...")
    input_shape = (rsds_model.shape[1], rsds_model.shape[2], 1)
    ml_model = build_physics_informed_ml_model_with_lambdas(
        input_shape, 
        lambda1=LAMBDA_1, 
        lambda2=LAMBDA_2, 
        lambda3=LAMBDA_3,
        scaling_factor=15.0
    )
    
    print(f"  Training...")
    ml_model, history = train_model(ml_model, training_data, epochs=1000, patience=10)
    
    print(f"  Applying corrections...")
    final_corrected = apply_ml_correction(ml_model, spectral_corrected, rsdscs, clt)
    
    valid_mask = (~np.isnan(rsds_model) & ~np.isnan(rsds_obs_grid) & 
                  ~np.isnan(nudged_rsds) & ~np.isnan(spectral_corrected) & 
                  ~np.isnan(final_corrected))
    
    original_rmse = calculate_rmse(rsds_model, rsds_obs_grid, valid_mask)
    nudged_rmse = calculate_rmse(nudged_rsds, rsds_obs_grid, valid_mask)
    spectral_rmse = calculate_rmse(spectral_corrected, rsds_obs_grid, valid_mask)
    ml_rmse = calculate_rmse(final_corrected, rsds_obs_grid, valid_mask)
    
    improvement = (1 - ml_rmse/original_rmse) * 100
    
    print(f"\n{'='*80}")
    print(f"RESULTS FOR {model_name}")
    print(f"{'='*80}")
    print(f"Original RMSE:  {original_rmse:8.4f} W/m²")
    print(f"Nudged RMSE:    {nudged_rmse:8.4f} W/m² ({(1-nudged_rmse/original_rmse)*100:5.2f}%)")
    print(f"Spectral RMSE:  {spectral_rmse:8.4f} W/m² ({(1-spectral_rmse/original_rmse)*100:5.2f}%)")
    print(f"Final ML RMSE:  {ml_rmse:8.4f} W/m² ({improvement:5.2f}%)")
    print(f"{'='*80}\n")
    
    output_dir = 'corrected_rsds'
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = os.path.join(output_dir, f"{model_name}_rsds")
    save_coords = {dim: model_coords[dim][:min_time] for dim in model_dims}
    
    save_netcdf(nudged_rsds, save_coords, model_attrs, f"{base_filename}_nudged.nc")
    save_netcdf(spectral_corrected, save_coords, model_attrs, f"{base_filename}_spectral.nc")
    save_netcdf(final_corrected, save_coords, model_attrs, f"{base_filename}_final.nc")
    
    with open(f"{base_filename}_metrics.txt", 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Lambda parameters: λ₁={LAMBDA_1}, λ₂={LAMBDA_2}, λ₃={LAMBDA_3}\n")
        f.write(f"Original RMSE: {original_rmse:.4f} W/m²\n")
        f.write(f"Nudged RMSE: {nudged_rmse:.4f} W/m² ({(1-nudged_rmse/original_rmse)*100:.2f}%)\n")
        f.write(f"Spectral RMSE: {spectral_rmse:.4f} W/m² ({(1-spectral_rmse/original_rmse)*100:.2f}%)\n")
        f.write(f"Final RMSE: {ml_rmse:.4f} W/m² ({improvement:.2f}%)\n")
    
    del rsds_model, rsdscs, clt, nudged_rsds, spectral_corrected, final_corrected, ml_model
    tf.keras.backend.clear_session()
    gc.collect()
    
    return {
        'model_name': model_name,
        'original_rmse': original_rmse,
        'nudged_rmse': nudged_rmse,
        'spectral_rmse': spectral_rmse,
        'ml_rmse': ml_rmse,
        'improvement': improvement
    }

def find_model_files():
    """Find all CMIP6 model files with corresponding physics constraints."""
    model_files = glob.glob("rsds_mon_*_historical126_*.nc")
    model_data = []
    
    for rsds_file in model_files:
        model_name = rsds_file.split('_')[2]
        
        rsdscs_candidates = glob.glob(f"rsdscs*{model_name}*historical*_1983_2014.nc")
        clt_candidates = glob.glob(f"clt*{model_name}*historical*_1983_2014.nc")
        
        if rsdscs_candidates and clt_candidates:
            model_data.append({
                'model_name': model_name,
                'rsds_file': rsds_file,
                'rsdscs_file': rsdscs_candidates[0],
                'clt_file': clt_candidates[0]
            })
        else:
            print(f"WARNING: Missing files for {model_name}")
    
    return model_data

def main():
    """
    Main execution: Process all CMIP6 models with lambda-weighted loss.
    
    Loss Function: L = λ₁L_empirical + λ₂L_physical + λ₃L_regularization
    With λ₁=0.6, λ₂=0.3, λ₃=0.1
    """
    print("\n" + "="*80)
    print("PHYSICS-CONSTRAINED DEEP LEARNING WITH LAMBDA-WEIGHTED LOSS")
    print("L_total = λ₁L_empirical + λ₂L_physical + λ₃L_regularization")
    print(f"Lambda parameters: λ₁={LAMBDA_1}, λ₂={LAMBDA_2}, λ₃={LAMBDA_3}")
    print("="*80 + "\n")
    
    print("Loading SARAH-2.1 observation data...")
    obs_data = load_obs_data()
    
    model_data = find_model_files()
    print(f"\nFound {len(model_data)} CMIP6 models")
    
    if len(model_data) == 0:
        print("ERROR: No model files found")
        return
    
    results = []
    for model_info in tqdm(model_data, desc="Processing models"):
        try:
            result = process_model(
                model_info['model_name'],
                model_info['rsds_file'],
                model_info['rsdscs_file'],
                model_info['clt_file'],
                obs_data
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"ERROR: {e}")
    
    if results:
        df = pd.DataFrame(results)
        df.to_csv('all_model_results_lambda_weighted.csv', index=False)
        
        print("\n" + "="*80)
        print("SUMMARY: ALL MODELS (Lambda-Weighted Loss)")
        print("="*80)
        print(df.to_string(index=False))
        print(f"\nMean improvement: {df['improvement'].mean():.2f}%")
        print(f"Range: {df['improvement'].min():.2f}% - {df['improvement'].max():.2f}%")
        print("="*80 + "\n")
        print("Results saved to: all_model_results_lambda_weighted.csv")
    
    return results

if __name__ == "__main__":
    main()

