# PhysConDL-Solar-Bias-Correction

Physics-Constrained Deep Learning Framework for CMIP6 Solar Radiation Bias Correction over Africa

## Overview

This repository contains code templates and analysis scripts related to the research paper:

**"Physics-constrained Deep Learning Bias Correction of CMIP6 Solar Radiation Over Africa and Its Implications for Solar Power Planning in a Changing Climate"**

The framework introduces a novel physics-constrained deep learning approach that addresses systematic biases in CMIP6 surface solar radiation simulations across Africa by incorporating radiative transfer principles within neural network architectures.

## Repository Structure

- `sensitivity_analysis.py` - Sensitivity coefficient calculation
- `figure1_bias_analysis.py` - CMIP6 model bias visualization  
- `figure7_regional_anomalies.py` - Regional radiation anomaly analysis
- `uncertainty_decomposition.py` - Uncertainty partitioning following Hawkins & Sutton (2009)
- `PhyConML_framework.py` - Physics-constrained ML framework (template)


## Key Features

- **Physics-Informed Architecture**: Incorporates radiative transfer principles through specialized neural networks
- **Multi-Component Correction**: Three-stage pipeline including physics-based nudging, spectral correction, and machine learning refinement
- **Regional Analysis**: Comprehensive evaluation across Africa's diverse climate zones
- **Uncertainty Quantification**: Decomposition of model, scenario, and internal variability uncertainties


## Requirements

```
numpy>=1.21.0
xarray>=0.19.0
tensorflow>=2.8.0
matplotlib>=3.5.0
cartopy>=0.20.0
scipy>=1.7.0
pandas>=1.3.0
geopandas>=0.10.0
regionmask>=0.8.0
```

## Usage

### Basic Analysis

```python
# Run sensitivity analysis
python sensitivity_analysis.py

# Generate bias analysis plots
python figure1_bias_analysis.py

# Calculate uncertainty decomposition
python uncertainty_decomposition.py

# Regional anomaly analysis
python figure7_regional_anomalies.py
```

### Data Requirements

The scripts expect NetCDF files containing:
- Observational solar radiation data
- Climate model solar radiation data (rsds) 
- Clear-sky solar radiation data (rsdscs)
- Total cloud cover data (clt)
- Regional shapefiles for land masking (optional)

**Note**: Users must provide their own datasets. File patterns and data structures in the code need to be adapted for specific datasets.

## Scientific Methodology

### Physics-Constrained Deep Learning (PhysConDL)

The framework implements a three-component approach:

1. **Physics-Based Nudging**: Integrates observational constraints with spatially varying weights based on atmospheric conditions
2. **Spectral Correction**: Preserves spatial coherence while reducing biases through Fourier domain processing
3. **Machine Learning Refinement**: Physics-informed neural networks with embedded radiative transfer constraints

### Key Results

- **Bias Reduction**: 70-85% reduction in mean absolute errors continent-wide
- **Uncertainty Analysis**: Model uncertainty dominates projections (51%), internal variability affects near-term projections (49.1%)
- **Regional Patterns**: Systematic radiation reductions of 0.75-1.4 W/m² with pronounced seasonal asymmetry

## Data Availability

- **SARAH-2.1 observations**: Available from EUMETSAT Climate Monitoring SAF
- **CMIP6 model data**: Accessible through Earth System Grid Federation (ESGF)
- **Processed datasets**: Contact corresponding author for bias-corrected outputs

## Contributing

This repository serves as a research template. For questions about implementation details or collaboration opportunities, please contact the corresponding author.

## Development Status

- ✅ Sensitivity analysis implementation
- ✅ Bias visualization tools
- ✅ Regional anomaly analysis
- ✅ Uncertainty decomposition framework
- ⚠️ Physics-constrained ML framework (template only)


## Contact

**Corresponding Author**: Paul Adigun  
**Email**: pauladigun7@gmail.com  
**Institution**: University of Tsukuba, Department of Engineering Mechanics and Energy

**Disclaimer**: Disclaimer: This repository contains research code templates and incomplete implementations. The complete, fully functional code will be made available following publication of the associated research paper. The current incomplete components are intentionally provided to demonstrate the scientific methodology while protecting intellectual property during the peer review process.
