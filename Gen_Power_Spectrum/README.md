# CAMB Power Spectrum Batch Calculator

This program automatically generates CMB power spectra using CAMB (Code for Anisotropies in the Microwave Background) for multiple combinations of cosmological parameters.

## Overview

The script performs systematic calculations by varying two key cosmological parameters:
- **As (Scalar Amplitude)**: The amplitude of scalar perturbations at the pivot scale k = 0.05 Mpc⁻¹
- **r (Tensor-to-Scalar Ratio)**: The ratio of tensor to scalar perturbation amplitudes

## Features

- **Automated Parameter Variation**: Systematically varies As and r parameters across predefined ranges
- **Batch Processing**: Runs 30 CAMB calculations automatically (3 As values × 10 r values)
- **Organized Output**: Results are saved in separate directories with descriptive names
- **Progress Tracking**: Real-time progress updates and error reporting
- **Error Handling**: Tracks successful and failed runs
- **Clean File Management**: Automatically creates output directories

## Parameter Ranges

### Scalar Amplitude (As)
- **Values**: 2.0×10⁻⁹, 2.1×10⁻⁹, 2.2×10⁻⁹
- **Physical meaning**: Controls the amplitude of density fluctuations

### Tensor-to-Scalar Ratio (r)
- **Values**: 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010
- **Physical meaning**: Ratio of gravitational wave to density fluctuation amplitudes

## Prerequisites

1. **CAMB Installation**: Ensure CAMB is compiled and the executable is located at `./camb/camb`
2. **Parameter File**: A valid `params.ini` file must be present in the working directory
3. **Python Environment**: Python 3.x with standard libraries (os, sys, numpy)

## Directory Structure

```
project/
├── get_cls.py          # Main script
├── params.ini          # CAMB parameter file (template)
├── camb/              
│   └── camb           # CAMB executable
└── cls/               # Output directory (created automatically)
    ├── cls_As_2.0_r_0.001/
    ├── cls_As_2.0_r_0.002/
    ├── ...
    └── cls_As_2.2_r_0.010/
```

## Usage

### Basic Usage
```bash
python get_cls.py
```

### What the Script Does

1. **Parameter Modification**: For each parameter combination, the script:
   - Modifies `scalar_amp(1)` in `params.ini`
   - Modifies `initial_ratio(1)` in `params.ini`
   - Sets `output_root` to a descriptive directory name

2. **CAMB Execution**: Runs `./camb/camb params.ini` for each parameter set

3. **Output Organization**: Results are automatically saved to directories named:
   ```
   cls/cls_As_2.0_r_0.001/
   cls/cls_As_2.1_r_0.005/
   etc.
   ```

## Output Files

Each run generates standard CAMB output files in its respective directory:

- `scalCls.dat` - Scalar power spectra
- `vecCls.dat` - Vector power spectra  
- `tensCls.dat` - Tensor power spectra
- `totCls.dat` - Total power spectra
- `lensedCls.dat` - Lensed power spectra
- `lensedtotCls.dat` - Total lensed power spectra
- `lenspotentialCls.dat` - Lensing potential power spectra

## Example Output

```
Starting CAMB calculations with 30 parameter combinations...
============================================================
Running 1/30: As=2.0×10^-9, r=0.001
Completed run 1: As=2.0×10^-9, r=0.001
----------------------------------------
Running 2/30: As=2.0×10^-9, r=0.002
Completed run 2: As=2.0×10^-9, r=0.002
----------------------------------------
...
============================================================
All CAMB calculations completed!
Total runs: 30
Successful runs: 30
Failed runs: 0
Results saved in cls/ directory with naming convention: cls_As_X.X_r_X.XXX/
============================================================
```

## Customization

### Modifying Parameter Ranges

Edit the parameter arrays in `get_cls.py`:

```python
# Scalar amplitude values (in units of 10^-9)
scalar_amp_values = [2.0e-9, 2.1e-9, 2.2e-9]

# Tensor-to-scalar ratio values
initial_ratio_values = [0.001, 0.002, 0.003, 0.004, 0.005, 
                       0.006, 0.007, 0.008, 0.009, 0.01]
```

### Adding Additional Parameters

To modify other CAMB parameters, extend the `modify_params_ini()` function:

```python
def modify_params_ini(scalar_amp, initial_ratio):
    # Add more parameter modifications here
    if line.strip().startswith('hubble'):
        lines[i] = f'hubble         = {your_hubble_value}\n'
```

## Error Handling

The script includes basic error handling:
- Tracks CAMB execution success/failure
- Reports failed runs at completion
- Continues execution even if individual runs fail
- Creates necessary directories automatically

## Performance Notes

- **Execution Time**: Each CAMB run typically takes 1-10 minutes depending on parameters
- **Total Runtime**: Complete batch (~30 runs) may take 30 minutes to 5 hours
- **Disk Space**: Each run generates ~1-10 MB of output files
- **Memory Usage**: Minimal Python memory footprint; CAMB memory usage varies

## Troubleshooting

### Common Issues

1. **CAMB Not Found**: Ensure `./camb/camb` exists and is executable
2. **Permission Errors**: Check write permissions for output directories
3. **Parameter File Issues**: Verify `params.ini` format and required parameters
4. **Memory Issues**: Monitor system memory during large batch runs

### Debug Mode

For debugging, you can add verbose output by modifying the print statements or adding:

```python
print(f"Modified params.ini with As={scalar_amp:.1e}, r={initial_ratio:.3f}")
```

## Scientific Applications

This tool is useful for:
- **Cosmological Parameter Studies**: Systematic exploration of parameter space
- **Model Comparison**: Generating predictions for different inflationary models  
- **Observational Constraints**: Creating theoretical predictions for data analysis
- **Sensitivity Analysis**: Understanding parameter impact on observables

## References

- [CAMB Documentation](https://camb.info/)
- [CAMB GitHub Repository](https://github.com/cmbant/CAMB)
- [Cosmological Parameters Review](https://pdg.lbl.gov/2020/reviews/rpp2020-rev-bbang-cosmology.pdf)

## License

This script is provided as-is for scientific research purposes. Please cite CAMB appropriately in any publications using results generated by this tool. 