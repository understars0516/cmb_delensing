# CMB Delensing Dataset Generation Pipeline - Detailed Documentation

This document provides a comprehensive description of the simulation data generation pipeline for the CMB (Cosmic Microwave Background) delensing project. The entire data processing pipeline consists of four main steps, ultimately used to train deep learning models for CMB delensing tasks.

## Project Overview

The main program of this project is `run.ipynb`, which uses U-Net deep learning architecture to perform CMB map delensing. The complete data generation and processing pipeline includes:

1. **Angular Power Spectrum Generation** - Generate theoretical CMB angular power spectra using CAMB
2. **Sky Map Simulation** - Create sky maps with realistic observational effects based on power spectra
3. **Sky Map Splitting** - Convert full-sky maps into trainable patches
4. **Rotational Data Augmentation** - Apply spherical rotation transformations to reduce splitting errors

---

## Step 1: CMB Angular Power Spectrum Generation (Gen_Power_Spectrum)

### Objective
Generate CMB angular power spectra for different cosmological parameters using CAMB (Code for Anisotropies in the Microwave Background), creating a total of 30 different parameter combinations.

### Core Parameters

#### Cosmological Parameter Ranges:
- **Scalar Amplitude (As)**: [2.0, 2.1, 2.2] × 10⁻⁹
  - Physical meaning: Controls the amplitude of density perturbations
  - Affects the overall amplitude of temperature and polarization power spectra

- **Tensor-to-Scalar Ratio (r)**: [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010]
  - Physical meaning: Ratio of primordial gravitational wave to density perturbation amplitudes
  - Affects the strength of B-mode polarization signals

### Technical Implementation

#### Batch Processing System
- **Automated calculations**: Systematically varies As and r parameter combinations
- **Total calculations**: 3 As values × 10 r values = 30 parameter combinations
- **Parallel processing**: Supports multi-core parallel computation for improved efficiency

#### Example Notebook
The directory contains `get_cls.ipynb` which demonstrates:
- Interactive parameter exploration
- Visualization of power spectra
- Comparison between different parameter combinations
- Quality control and validation

#### Output File Organization
Each parameter combination generates an independent directory:
```
cls/
├── cls_As_2.0_r_0.001/
│   ├── scalCls.dat         # Scalar power spectra
│   ├── tensCls.dat         # Tensor power spectra
│   ├── totCls.dat          # Total power spectra
│   ├── lensedCls.dat       # Lensed power spectra
│   ├── lensedtotCls.dat    # Total lensed power spectra
│   └── lenspotentialCls.dat # Lensing potential power spectra
├── cls_As_2.0_r_0.002/
└── ... (30 total combinations)
```

#### Usage
```bash
python get_cls.py
# or interactive analysis
jupyter notebook get_cls.ipynb
```

---

## Step 2: CMB Sky Map Simulation (Gen_Sky_Map)

### Objective
Based on the angular power spectra generated in Step 1, create high-resolution CMB sky maps containing realistic observational effects including lensed CMB maps, noise, beam effects, and unlensed CMB maps as training labels.

### Technical Specifications

#### Map Resolution Parameters
- **HEALPix NSIDE**: 2048
- **Number of pixels**: 12,582,912 pixels
- **Angular resolution**: ~1.7 arcmin/pixel
- **Maximum multipole**: ℓ_max = 3000 (lensed) / 4024 (unlensed buffer)

#### Sky Map Types
Each parameter combination generates 6 types of sky maps:

**Temperature Maps**:
- `Tlen_as_X.X_r_X.XXX.fits` - Lensed temperature maps (training input)
- `Tunl_as_X.X_r_X.XXX.fits` - Unlensed temperature maps (training target)

**E-mode Polarization**:
- `Elen_as_X.X_r_X.XXX.fits` - Lensed E-mode maps (training input)
- `Eunl_as_X.X_r_X.XXX.fits` - Unlensed E-mode maps (training target)

**B-mode Polarization**:
- `Blen_as_X.X_r_X.XXX.fits` - Lensed B-mode maps (training input)
- `Bunl_as_X.X_r_X.XXX.fits` - Unlensed B-mode maps (training target)

### Observational Effects Modeling

#### Gravitational Lensing Effects
- **Implementation tool**: lenspyx library for curved-sky calculations
- **Physical process**: Use lensing potential to deflect primordial CMB
- **Accuracy**: Maintains complete nonlinear lensing effects

#### Instrumental Noise Model
- **Temperature sensitivity**: 1.5 μK·arcmin
- **Polarization sensitivity**: 2.1 μK·arcmin (E and B modes)
- **Noise type**: Gaussian white noise
- **Noise power spectrum**: N_ℓ = (ΔT)² / B_ℓ²

#### Beam Effects
- **Beam profile**: 20 arcminute Full Width at Half Maximum (FWHM) Gaussian beam
- **Mathematical form**: B_ℓ = exp(-ℓ(ℓ+1)θ²/8ln2)
- **Physical meaning**: Simulates telescope finite angular resolution

#### Example Notebook
The directory contains `data_sim.ipynb` which demonstrates:
- Step-by-step sky map generation process
- Visualization of lensing effects
- Comparison of lensed vs unlensed maps
- Quality control and validation

#### Usage
```bash
python data_sim.py
# or interactive analysis
jupyter notebook data_sim.ipynb
```

---

## Step 3: Sky Map Splitting (Split_Sky_Map)

### Objective
Convert full-sky HEALPix maps (NSIDE=2048) into square patches suitable for deep learning training. Transform one full-sky map into 192 patches of size (512, 512) each, making the data compatible with CNN architectures.

### Technical Specifications

#### Conversion Parameters
- **Input**: Full-sky HEALPix maps (12,582,912 pixels)
- **Output**: 192 square patches of 512×512 pixels each
- **Preservation**: Maintains all original pixel information
- **Total coverage**: Complete sky coverage with optimized patch distribution

#### Key Features
- **Lossless transformation**: No pixel information is lost during splitting
- **Optimized patch size**: 512×512 patches ideal for CNN training
- **Batch processing**: Handles multiple maps simultaneously
- **Memory efficient**: Processes large datasets without excessive RAM usage

### Algorithm Details

#### Pixel Rearrangement Process
1. **HEALPix Index Rearrangement**: Optimizes pixel ordering for patch extraction
2. **Spatial Grouping**: Groups pixels by spatial locality
3. **Patch Formation**: Creates rectangular patches from rearranged pixels
4. **Boundary Handling**: Manages edges and overlap regions

#### Core Arrays
- **`rearr_nside2048.npz`**: HEALPix pixel rearrangement indices (99 MB)
- **`arr_nside2048_192x512x512.npz`**: Mapping from 1D to 2D patch format (3.1 GB)

#### Implementation
```python
# Load rearrangement arrays
rearr = np.load("rearr_nside2048.npz")['data']
arr = np.load("arr_nside2048_192x512x512.npz")['data']

# Transform full-sky map to patches
patches = healpix_map.reshape(-1)[rearr][arr]
# Result: (192, 512, 512) array of patches
```

#### Example Notebook
The directory contains `split_sky_map.ipynb` which demonstrates:
- Visualization of the splitting process
- Comparison of original vs split maps
- Quality control and validation
- Batch processing examples

#### Usage
```bash
python split_sky_map.py
# or interactive analysis
jupyter notebook split_sky_map.ipynb
```

---

## Step 4: Rotational Data Augmentation (Rot_Sky_Map)

### Objective
Apply spherical rotation transformations to CMB sky maps to reduce errors introduced by the splitting process. This step increases training data diversity and helps the model generalize better by learning from different orientations of the same sky patterns.

### Core Innovation: Perfect Pixel Preservation Algorithm

#### Technical Advantages
- **Zero pixel loss**: No pixel information lost during rotation process
- **Perfect recovery**: Ability to completely restore original maps
- **Conflict resolution**: Intelligent handling of pixel mapping conflicts
- **Batch processing support**: Efficient simultaneous rotation of multiple maps

#### Redistribution Algorithm Principles
1. **Conflict detection**: Identify pixels mapping to the same location
2. **Hole identification**: Find empty positions in rotated maps
3. **Smart redistribution**: Reassign conflicting pixels to empty hole positions
4. **Bijective mapping**: Ensure one-to-one correspondence between source and target pixels

### Rotation Parameter Settings

#### Rotation Combinations Used in Main Program
8 predefined rotation angle combinations are defined in `run.ipynb`:
```python
theta_phi_combinations = [
    [60, -30], [60, -60], [120, 30], [120, 60], 
    [240, -30], [240, -60], [240, 30], [240, 60]
]
```

**Currently used**: Combination index 3, i.e., `[120°, 60°]`
- **θ (theta)**: 120° - polar angle rotation
- **φ (phi)**: 60° - azimuthal angle rotation

#### Coordinate System Definition
- **θ (polar angle)**: 
  - Range: 0° ≤ θ ≤ 180°
  - θ = 0°: North pole; θ = 90°: Equator; θ = 180°: South pole
- **φ (azimuthal angle)**:
  - Range: 0° ≤ φ < 360° (or -180° ≤ φ < 180°)
  - φ = 0°: Reference meridian

### Pre-computation Acceleration System

#### Rotation Array Generation
To improve training efficiency, the system pre-computes and saves rotation mapping arrays:
- **Forward rotation arrays**: `rot_arr_{theta}_{phi}.npy`
- **Inverse recovery arrays**: `rerot_arr_{theta}_{phi}.npy`

#### HEALPix Pixel Rearrangement
Combined with special pixel rearrangement system:
- **Rearrangement array**: `rearr_nside2048.npy` - HEALPix pixel index rearrangement
- **Mapping array**: `arr_nside2048_192x512x512.npy` - Mapping from 1D HEALPix to 2D images

#### Data Processing Pipeline
```python
# Complete pipeline for applying rotation transformations
x_rotated = x_batch.reshape(-1)[rearr][rot_arr][arr]
y_rotated = y_batch.reshape(-1)[rearr][rot_arr][arr]
```

#### Example Notebook
The directory contains `rot_map.ipynb` which demonstrates:
- Step-by-step rotation process
- Visualization of rotation effects
- Perfect recovery verification
- Performance benchmarking

#### Usage
```bash
python run.py
# or interactive analysis
jupyter notebook rot_map.ipynb
```

### Performance Optimization

#### Speed Comparison
- **Standard rotation method**: Single map ~2.5 seconds
- **Our method (first time)**: Single map ~3.0 seconds
- **Our method (pre-computed)**: Single map ~0.01 seconds

#### Memory Usage
- **Rotation array storage**: ~32MB for NSIDE=2048
- **Temporary storage**: Minimal additional memory footprint
- **Scalability**: Linear scaling with number of pixels

---

## Final Training Data Format

### Dataset Structure
After the four-step processing pipeline, the final training data format is:

```
train_dataset/
├── Tlen_5maps.npy    # Lensed T-mode maps (input)
├── Tunl_5maps.npy    # Unlensed T-mode maps (target output)
├── Elen_5maps.npy    # Lensed E-mode maps (input)
├── Eunl_5maps.npy    # Unlensed E-mode maps (target output)
├── Blen_5maps.npy    # Lensed B-mode maps (input)
└── Bunl_5maps.npy    # Unlensed B-mode maps (target output)
```

### Data Dimensions
- **Number of maps**: Each file contains 5 sets of map data
- **Size per set**: 192 patches of 512×512 pixels each
- **Total samples**: 960 training samples (5×192)
- **Data type**: float32, saving memory and improving training speed

### Training/Validation Split
- **Training set**: First 4 map sets (768 samples)
- **Validation set**: 5th map set (192 samples)
- **Test set**: Full validation set (960 samples) for final evaluation
- **Batch size**: 16 (distributed training across multiple GPUs)

---

## Data Pipeline Summary

1. **CAMB angular power spectrum calculation** → Generate theoretical power spectra for 30 parameter combinations
2. **lenspyx sky map simulation** → Generate 6 types of high-resolution sky maps per parameter set, including noise and beam effects
3. **HEALPix splitting** → Convert full-sky maps into 192 patches of 512×512 pixels suitable for CNN training
4. **Perfect rotation augmentation** → Apply spherical rotations to reduce splitting errors and increase data diversity
5. **Deep learning training** → U-Net model learns delensing mapping relationships

This complete data processing pipeline ensures:
- **Physical accuracy**: Uses state-of-the-art cosmological calculation tools
- **Observational realism**: Includes complete instrumental effects and noise models
- **Training efficiency**: Optimized data formats and pre-computation acceleration
- **Error mitigation**: Rotation augmentation reduces systematic errors from splitting
- **Scientific value**: Provides practical delensing tools for CMB cosmology research

## Quick Start Guide

### Prerequisites
- Python 3.7+
- Required libraries: numpy, healpy, lenspyx, matplotlib, tensorflow
- CAMB executable
- Sufficient disk space (~100GB for complete dataset)
- 16+ GB RAM recommended

### Running the Complete Pipeline
1. **Generate power spectra**: `cd Gen_Power_Spectrum && python get_cls.py`
2. **Simulate sky maps**: `cd Gen_Sky_Map && python data_sim.py`
3. **Split maps**: `cd Split_Sky_Map && python split_sky_map.py`
4. **Apply rotations**: `cd Rot_Sky_Map && python run.py`
5. **Train model**: `jupyter notebook run.ipynb`

### Interactive Analysis
Each step includes Jupyter notebooks for interactive exploration:
- `Gen_Power_Spectrum/get_cls.ipynb`
- `Gen_Sky_Map/data_sim.ipynb`
- `Split_Sky_Map/split_sky_map.ipynb`
- `Rot_Sky_Map/rot_map.ipynb`
