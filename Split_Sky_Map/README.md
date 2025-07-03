# HEALPix Split Sky Map

This project provides tools for splitting HEALPix sky maps into rectangular patches and reconstructing them back to the original HEALPix format. This is particularly useful for applying convolutional neural networks or other image processing techniques to astronomical sky maps.

## Overview

HEALPix (Hierarchical Equal Area isoLatitude Pixelization) is a scheme for discretizing the sphere that is commonly used in astrophysics and cosmology. However, the native HEALPix format is not directly compatible with standard image processing techniques that expect rectangular grids. This project addresses this by:

1. **Splitting**: Converting a HEALPix map into multiple rectangular patches
2. **Reconstructing**: Converting the processed patches back to the original HEALPix format

## Files Description

### Data Files

- **`arr_nside2048_192x512x512.npy`**: Index array for splitting HEALPix maps
  - Shape: (192, 512, 512)
  - Contains indices that map HEALPix pixels to 192 separate 512×512 rectangular patches
  - Used to transform HEALPix maps with nside=2048 into patch format

- **`rearr_nside2048.npy`**: Index array for reconstruction
  - Contains indices to reconstruct the original HEALPix map from the split patches
  - Used to transform processed patches back to HEALPix format

### Code Files

- **`split_data.ipynb`**: Jupyter notebook demonstrating the splitting and reconstruction process
  - Shows how to load and use the index arrays
  - Provides visualization of the transformation process
  - Includes verification that the reconstruction is lossless

## Usage

### Prerequisites

```bash
pip install numpy healpy matplotlib
```

### Running the Demonstration

1. Ensure you have the required `.npy` files in the same directory
2. Open and run `split_data.ipynb` in Jupyter notebook
3. The notebook will:
   - Load a test HEALPix map (using pixel indices as values)
   - Split it into 192 patches of 512×512 pixels
   - Reconstruct the original map
   - Display visualizations to verify the process

### Key Parameters

- **nside**: 2048 (HEALPix resolution parameter)
  - Corresponds to ~1.7 arcminute pixel resolution
  - Results in 12×nside² = 50,331,648 total pixels
- **Number of patches**: 192
- **Patch size**: 512×512 pixels
- **Total patch pixels**: 192×512×512 = 50,331,648 (matches HEALPix pixel count)

## Technical Details

### HEALPix Format
- **Resolution**: nside=2048
- **Total pixels**: 12×2048² = 50,331,648
- **Pixel size**: ~1.7 arcminutes
- **Coverage**: Full sky

### Patch Format
- **Number of patches**: 192
- **Patch dimensions**: 512×512
- **Total patch pixels**: 50,331,648 (lossless transformation)
- **Data type**: Integer indices

### Transformation Process

1. **Forward transformation** (HEALPix → Patches):
   ```python
   arr_mmap = original_healpix_map[arr]
   ```

2. **Inverse transformation** (Patches → HEALPix):
   ```python
   reconstructed_map = patches.reshape(-1)[rearr]
   ```

## Applications

This splitting scheme is particularly useful for:

- **Machine Learning**: Applying CNNs to astronomical sky maps
- **Image Processing**: Using standard computer vision techniques on sky data
- **Data Analysis**: Processing large sky surveys in manageable chunks
- **Cosmological Studies**: Analyzing CMB (Cosmic Microwave Background) data

## Verification

The notebook includes verification steps to ensure the transformation is lossless:
- Computes the difference between original and reconstructed maps
- Displays Mollweide projections for visual comparison
- Shows individual patches to verify spatial structure

## Notes

- The transformation is completely lossless (bijective)
- The splitting preserves spatial locality as much as possible
- The index arrays are pre-computed for efficiency
- Compatible with HEALPix maps of any physical units (the indices work for any nside=2048 map)

## License

This project is part of the CMB Delensing model development. 