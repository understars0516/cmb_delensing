# CMB Lensing Simulation with E/B Mode Analysis

A comprehensive Python package for simulating gravitational lensing effects on the Cosmic Microwave Background (CMB) radiation, with automatic conversion of Stokes parameters (Q, U) to E and B polarization modes. This package generates realistic observational data by incorporating instrumental noise and beam effects.

## 🌟 Key Features

- **High-resolution CMB simulation** (nside=2048, ~1.7 arcmin resolution)
- **Gravitational lensing effects** using lenspyx for accurate curved-sky calculations
- **Automatic E/B mode decomposition** from Q/U Stokes parameters
- **Realistic observational modeling** with instrumental noise and beam smoothing
- **Flexible cosmological parameters** (scalar amplitude A_s, tensor-to-scalar ratio r)
- **Comprehensive validation tools** with power spectrum analysis
- **Professional visualization** and comparison plots

## 📋 Project Overview

This simulation package models the complete pipeline from theoretical CMB power spectra to realistic observational data:

### Simulation Pipeline

1. **Input**: Custom CAMB power spectra files (lensed + lensing potential)
2. **Generation**: 
   - Unlensed CMB maps (theoretical reference)
   - Lensed CMB maps (gravitational lensing applied)
3. **Polarization Processing**: Q, U → E, B mode conversion using spin-2 harmonics
4. **Observational Effects**: 
   - Instrumental noise addition to lensed maps
   - Beam smoothing (20 arcmin FWHM Gaussian beam)
5. **Output**: Realistic observational data + theoretical references

### Physical Interpretation

- **Lensed Maps**: Simulate what telescopes actually observe (with lensing + noise + beam)
- **Unlensed Maps**: Theoretical CMB without lensing (pristine reference for comparison)
- **E/B Modes**: Physical polarization components with distinct cosmological signatures

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- Git
- ~16GB RAM recommended for nside=2048 simulations

### Quick Installation

```bash
git clone https://github.com/your-repo/cmb-lensing-simulation
cd cmb-lensing-simulation
make install
```

### Manual Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- **numpy**: Numerical computations
- **healpy**: HEALPix sky map operations  
- **matplotlib**: Visualization and plotting
- **lenspyx**: High-performance CMB lensing (requires ducc0)
- **ducc0**: Underlying spherical harmonic transforms

## 🚀 Quick Start

### 1. Test Installation
```bash
make test
# or
python quick_start.py
```

### 2. Full Simulation
```bash
make run
# or  
python data_sim.py
```

### 3. Verify Results
```bash
make verify
# or
python verify_maps.py
```

## 📊 Observational Parameters

The simulation incorporates realistic instrumental characteristics:

### Noise Model
- **Temperature sensitivity**: 1.5 μK·arcmin
- **Polarization sensitivity**: 2.1 μK·arcmin (E and B modes)
- **Beam FWHM**: 20 arcminutes (Gaussian beam profile)
- **Noise power spectrum**: `N_ℓ = (ΔT)² / B_ℓ²` where `B_ℓ = exp(-ℓ(ℓ+1)θ²/8ln2)`

### Cosmological Parameters
- **Scalar amplitude**: A_s ∈ [2.0, 2.1, 2.2] × 10⁹
- **Tensor-to-scalar ratio**: r ∈ [0.001, 0.002, ..., 0.01]
- **Resolution**: nside = 2048 (12,582,912 pixels)
- **Maximum multipole**: ℓ_max = 3000 (lensing), 4024 (unlensed buffer)

## 📁 Output Structure

Each simulation generates 6 primary sky maps per parameter combination:

### Map Types
```
map_data/
├── Tlen_as_2.100_r_0.005.fits    # Lensed temperature (observed)
├── Tunl_as_2.100_r_0.005.fits    # Unlensed temperature (theoretical)  
├── Elen_as_2.100_r_0.005.fits    # Lensed E-mode (observed)
├── Eunl_as_2.100_r_0.005.fits    # Unlensed E-mode (theoretical)
├── Blen_as_2.100_r_0.005.fits    # Lensed B-mode (observed)
└── Bunl_as_2.100_r_0.005.fits    # Unlensed B-mode (theoretical)
```

### Data Processing
- **Lensed maps**: Include gravitational lensing + instrumental noise + beam smoothing
- **Unlensed maps**: Pristine theoretical CMB for comparison and analysis
- **File format**: HEALPix FITS format (nside=2048, RING ordering)
- **Units**: Temperature in μK, E/B modes in μK

## 🔬 Scientific Applications

### Lensing Analysis
- **B-mode generation**: Gravitational lensing converts E-modes to B-modes
- **Power spectrum modification**: Smoothing of acoustic peaks by lensing
- **Delensing studies**: Remove lensing effects to access primordial B-modes

### Cosmological Constraints  
- **Tensor-to-scalar ratio**: Constrain primordial gravitational waves via B-modes
- **Scalar amplitude**: Study matter density fluctuations via E-mode power
- **Lensing potential**: Probe dark matter distribution through deflection patterns

### Systematic Studies
- **Noise impact**: Compare pristine vs. observational data quality
- **Beam effects**: Quantify resolution limitations on small-scale science
- **Method validation**: Test analysis pipelines with known input cosmology

## 🛠️ Usage Examples

### Basic Simulation
```python
import numpy as np
import healpy as hp
from data_sim import *

# Run simulation for specific parameters
As = [2.1]  # 10^9 A_s
rs = [0.005]  # r value

# Generates all 6 sky maps automatically
# Results saved to map_data/ directory
```

### E/B Mode Conversion
```python
# Convert existing Q, U maps to E, B modes
python convert_qu_to_eb.py -q Qmap.fits -u Umap.fits -o output_prefix

# Batch convert all maps
make convert-batch
```

### Power Spectrum Analysis
```python
import healpy as hp

# Load maps
Tlen = hp.read_map('map_data/Tlen_as_2.100_r_0.005.fits')
Elen = hp.read_map('map_data/Elen_as_2.100_r_0.005.fits')  
Blen = hp.read_map('map_data/Blen_as_2.100_r_0.005.fits')

# Calculate power spectra
cl_teb = hp.anafast([Tlen, Elen, Blen], lmax=3000)
cl_tt, cl_ee, cl_bb, cl_te, cl_tb, cl_eb = cl_teb
```

## 📈 Validation and Quality Control

### Automated Verification
The `verify_maps.py` script performs comprehensive quality checks:

- **File integrity**: Verify all expected output files exist
- **Statistical validation**: Check map statistics against expectations  
- **Power spectrum comparison**: Compare with theoretical ΛCDM predictions
- **Lensing effects**: Quantify B-mode enhancement from gravitational lensing
- **E/B decomposition**: Validate spin-2 harmonic transforms

### Validation Plots
Generated automatically in `validation_plots/`:
- **Power spectra**: TT, EE, BB comparison with theory
- **Map visualizations**: Full-sky and zoomed views
- **Lensing effects**: Difference maps showing lensing impact
- **E/B modes**: Comparison of E and B mode patterns

### Quality Metrics
- **Lensing detection**: B-mode power enhancement > 10%
- **Power spectrum accuracy**: Theory/simulation ratio within 10%
- **Noise validation**: Instrumental noise at expected levels
- **Beam verification**: Correct smoothing scale applied

## 🎯 Advanced Features

### Makefile Automation
```bash
make help              # Show all available commands
make install           # Install dependencies
make test              # Quick test (low resolution)
make run               # Full simulation
make verify            # Validate results
make convert-batch     # Convert all Q,U → E,B
make clean             # Remove output files
make full-workflow     # Complete pipeline
```

### Customization Options

#### Modify Simulation Parameters
Edit `data_sim.py`:
```python
# Resolution and accuracy
nside = 2048           # HEALPix resolution
lmax_len = 3000        # Maximum multipole
epsilon = 1e-6         # Numerical accuracy

# Instrumental parameters  
beam_fwhm_arcmin = 20.0     # Beam size
noise_T_sensitivity = 1.5   # Temperature noise (μK)
noise_E_sensitivity = 2.1   # E-mode noise (μK)
noise_B_sensitivity = 2.1   # B-mode noise (μK)

# Cosmological parameters
As = [2.0, 2.1, 2.2]       # Scalar amplitude range
rs = [0.001, 0.01]         # Tensor-to-scalar ratio range
```

#### Custom Power Spectra
Place CAMB output files in `cls/` directory:
```
cls/
├── cls_As_2.1_r_0.005_lensedCls.dat
└── cls_As_2.1_r_0.005_lenspotentialCls.dat
```

## 📊 Performance Considerations

### Computational Requirements
- **Memory**: ~16GB RAM for nside=2048 full simulation
- **CPU**: Multi-core recommended (lenspyx uses OpenMP)
- **Storage**: ~1GB per parameter combination (6 maps × 200MB each)
- **Runtime**: ~10-30 minutes per simulation (depends on hardware)

### Optimization Tips
- **Lower resolution**: Use nside=512 for quick tests
- **Reduced lmax**: Decrease lmax_len for faster computation  
- **Parallel processing**: Run multiple parameter combinations separately
- **Memory management**: Process one parameter set at a time for large surveys

## 🔧 Troubleshooting

### Common Issues

#### Installation Problems
```bash
# Missing ducc0 dependency
pip install ducc0

# HEALPix installation issues  
conda install -c conda-forge healpy

# lenspyx compilation problems
pip install --no-binary ducc0 ducc0
pip install lenspyx
```

#### Runtime Errors
```bash
# Memory issues
# Reduce nside or use swap space
export OMP_NUM_THREADS=4  # Limit parallel threads

# File not found errors
# Check cls/ directory contains power spectra files
ls cls/cls_As_*_r_*_*.dat
```

#### Validation Failures
- **Low B-mode enhancement**: Check lensing potential amplitude
- **Power spectrum mismatch**: Verify input CAMB files are correct
- **Noise level incorrect**: Check sensitivity parameter units (μK vs K)

## 📚 Scientific Background

### Gravitational Lensing of CMB

The CMB photons travel through the evolving universe and are deflected by gravitational potential wells. This lensing effect:

1. **Smooths acoustic peaks** in temperature and E-mode power spectra
2. **Generates B-mode polarization** from primordial E-modes  
3. **Correlates** CMB temperature/polarization with large-scale structure
4. **Provides window** into dark matter distribution and neutrino masses

### E/B Mode Decomposition

Polarization is naturally decomposed into:
- **E-modes**: Curl-free component (primordial + lensing)
- **B-modes**: Divergence-free component (lensing + tensors + systematic)

The spin-2 decomposition: Q ± iU → ±2Y_ℓm, then:
- E_ℓm = -(a_{2,ℓm} + a_{-2,ℓm})/2  
- B_ℓm = i(a_{2,ℓm} - a_{-2,ℓm})/2

### Observational Strategy

Modern CMB experiments measure temperature and polarization to:
- **Detect primordial gravitational waves** via tensor B-modes
- **Constrain dark energy** through lensing potential
- **Probe inflation** via scalar/tensor ratio r
- **Measure neutrino masses** through lensing effects

## 📖 References

1. **lenspyx**: Carron & Lewis (2017), "Length-six-py: A Python package for computing CMB lensing power spectra"
2. **HEALPix**: Górski et al. (2005), "HEALPix: A Framework for High-Resolution Discretization"  
3. **CMB Lensing**: Lewis & Challinor (2006), "Weak gravitational lensing of the CMB"
4. **CAMB**: Lewis et al. (2000), "Efficient Computation of Cosmic Microwave Background Anisotropies"

