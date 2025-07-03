# MAP Delensing Quick Fix v2

A robust implementation of Maximum A Posteriori (MAP) delensing for Cosmic Microwave Background (CMB) polarization data, with fixes for FFT shape mismatch issues and enhanced numerical stability.

## 🌟 Features

- **Shape-Adaptive Processing**: Automatically detects and handles 256×256 and 512×512 data formats
- **Conservative Algorithm**: Uses only QU polarization data to avoid temperature-related numerical instabilities  
- **Robust Error Handling**: Comprehensive exception handling and graceful degradation
- **5-Iteration Processing**: Performs up to 5 MAP iterations with early stopping on convergence
- **Comprehensive Output**: Generates both lensed vs delensed comparisons and detailed analysis plots
- **Complete Data Export**: Saves all I, Q, U components in both lensed and delensed forms

## 📋 Requirements

### Dependencies
- **Python 3.7+**
- **Lensit**: CMB lensing reconstruction library
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing (for interpolation)
- **Matplotlib**: Plotting and visualization

### Environment Setup
```bash
# Set required environment variable
export LENSIT=/path/to/lensit/installation

# Ensure Python packages are installed
pip install numpy scipy matplotlib
```

## 📁 Input Data Structure

The script expects input data in the following structure:
```
project_root/
├── npys/
│   ├── Ilen.npy    # I Stokes parameter (lensed)
│   ├── Qlen.npy    # Q Stokes parameter (lensed) 
│   └── Ulen.npy    # U Stokes parameter (lensed)
├── results/        # Output directory (must exist)
└── quick_fix_v2.py
```

### Supported Data Formats
- **256×256 pixels**: Automatically uses LD_res=HD_res=8
- **512×512 pixels**: Automatically uses LD_res=HD_res=9

## 🚀 Usage

### Basic Execution
```bash
python quick_fix_v2.py
```

### Expected Output
The script will display progress information:
```
🔧 MAP Delensing Quick Fix v2
==================================================
Loading data...
✓ Data shape: (512, 512)
✓ Using resolution parameters: LD_res=HD_res=9 (for 512×512)
Preprocessing data...
...
✓ Quick Fix v2 processing completed successfully!
```

## 📊 Output Files

### Visualization Results
- **`results/quick_fix_v2_comparison.png`**: Side-by-side comparison of lensed vs delensed Q and U polarization maps
- **`results/quick_fix_v2_analysis.png`**: Technical analysis including:
  - Reconstructed lensing potential φ
  - Deflection amplitude |∇φ|
  - Q and U polarization differences with RMS statistics

### Numerical Data
All data saved as NumPy arrays in `results/` directory:

**Original (Lensed) Data:**
- `I_lensed.npy`: I Stokes parameter
- `Q_lensed.npy`: Q Stokes parameter  
- `U_lensed.npy`: U Stokes parameter

**Processed (Delensed) Data:**
- `I_delensed.npy`: Delensed I Stokes parameter
- `Q_delensed.npy`: Delensed Q Stokes parameter
- `U_delensed.npy`: Delensed U Stokes parameter

**Reconstruction Results:**
- `reconstructed_phi.npy`: Lensing potential φ map
- `reconstructed_plm.npy`: Lensing potential spherical harmonic coefficients

## ⚙️ Algorithm Configuration

### Conservative Parameters
The script uses conservative settings for enhanced stability:

```python
# Experiment Configuration
exp_config = 'Planck'                    # Conservative Planck settings
ellmax = min(ellmax, 1500)               # Reduced maximum multipole
ellmax_sky = 2000                        # Sky reconstruction limit
ellmax_qlm = 1500                        # Lensing reconstruction limit

# Noise Enhancement for Stability  
sN_uKamin *= 2.0                         # Doubled temperature noise
sN_uKaminP *= 2.0                        # Doubled polarization noise

# Iteration Settings
max_iterations = 5                       # Up to 5 MAP iterations
tolerance = 1e-3                         # Convergence tolerance
```

### Data Preprocessing
- **Automatic Scaling**: Data with RMS > 150 is scaled to reasonable range
- **Outlier Clipping**: 4-sigma outlier removal
- **Amplitude Limiting**: Initial estimates capped at 5×10⁻⁶

## 🔧 Technical Details

### Core Algorithm Components

1. **Spherical Harmonic Libraries**: Three separate libraries for data, sky, and reconstruction multipole ranges
2. **Covariance Calculation**: Diagonal covariance matrix with lensing effects
3. **Quadratic Estimator**: QU-only polarization-based lensing reconstruction
4. **MAP Iterator**: Perturbative maximum a posteriori estimation with BFGS optimization
5. **Delensing**: Simplified deflection reversal using bilinear interpolation

### Lensit Functions Used
- `li.get_config()`: Experiment parameter retrieval
- `li.get_ellmat()`: Flat-sky ellmax matrix generation
- `li.get_fidcls()`: Fiducial power spectrum loading
- `ell_mat.ffs_alm_pyFFTW()`: Spherical harmonic coefficient libraries
- `ffs_cov.ffs_diagcov_alm()`: Covariance matrix setup
- `ffs_ninv_filt_ideal.ffs_ninv_filt()`: Inverse noise filtering
- `chain_samples.get_isomgchain()`: Multigrid preconditioning
- `ffs_iterator_pertMF()`: Perturbative MAP iterator

## 🛠️ Troubleshooting

### Common Issues

**Environment Variable Not Set**
```
❌ Please set LENSIT environment variable
```
**Solution**: `export LENSIT=/path/to/lensit`

**Data Loading Failures**
```
❌ Cannot load data files
```
**Solution**: Ensure `npys/` directory contains `Ilen.npy`, `Qlen.npy`, `Ulen.npy`

**Unsupported Data Shape**
```
❌ Unsupported data shape: (1024, 1024)
```
**Solution**: Resize data to 256×256 or 512×512, or modify resolution parameters

**Iteration Failures**
```
❌ Iteration 1 failed: ffs_displ::Negative value in det k
```
**Solution**: This indicates numerical instability. The script uses previous successful iterations.

### Performance Optimization

- **Memory Usage**: ~8-16 GB RAM recommended for 512×512 data
- **Processing Time**: 15-60 minutes depending on system and convergence
- **Disk Space**: ~200 MB for full output set

## 📚 Scientific Background

This implementation performs CMB delensing using the MAP approach:

1. **Lensing Reconstruction**: Uses quadratic estimators on QU polarization to reconstruct the lensing potential φ
2. **Iterative Refinement**: Applies perturbative MAP iterations to improve the reconstruction
3. **Delensing**: Reverses the lensing effect by applying inverse deflections to the original data

### Key References
- Carron & Lewis (2017): "Maximum a posteriori CMB lensing reconstruction"
- Planck Collaboration (2018): "Planck 2018 results. VIII. Gravitational lensing"

## 📝 License

This project uses the Lensit library and follows its licensing terms. Please cite appropriate references when using this code for scientific research.

## 🤝 Contributing

For bug reports or feature requests, please ensure:
1. Include complete error messages
2. Specify input data characteristics (shape, value ranges)
3. Provide system information (Python version, available memory)

---

**Version**: 2.0  
**Last Updated**: 2024  
**Compatibility**: Lensit, Python 3.7+