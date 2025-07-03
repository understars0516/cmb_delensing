import os, sys
import numpy as np

# Define parameter values
# As: scalar amplitude at k=0.05 Mpc^-1
scalar_amp_values = [2.0e-9, 2.1e-9, 2.2e-9]
# r: tensor-to-scalar ratio
initial_ratio_values = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]

def modify_params_ini(scalar_amp, initial_ratio):
    """Modify parameters in params.ini file"""
    # Read original params.ini file
    with open('params.ini', 'r') as f:
        lines = f.readlines()
    
    # Modify parameters
    for i, line in enumerate(lines):
        if line.strip().startswith('scalar_amp(1)'):
            lines[i] = f'scalar_amp(1)             = {scalar_amp:.1e}\n'
        elif line.strip().startswith('initial_ratio(1)'):
            lines[i] = f'initial_ratio(1)          = {initial_ratio:.3f}\n'
        elif line.strip().startswith('output_root'):
            lines[i] = f'output_root = ./cls/cls_As_{scalar_amp*1e9:.1f}_r_{initial_ratio:.3f}\n'
    
    # Write back to file
    with open('params.ini', 'w') as f:
        f.writelines(lines)

def run_camb_and_save(run_number, scalar_amp, initial_ratio):
    """Run CAMB and save results"""
    as_value = scalar_amp * 1e9  # Convert to 10^-9 units for display
    print(f"Running {run_number}/30: As={as_value:.1f}×10^-9, r={initial_ratio:.3f}")
    
    # Create cls directory if it doesn't exist
    cls_dir = f"cls/cls_As_{as_value:.1f}_r_{initial_ratio:.3f}"
    if not os.path.exists(cls_dir):
        os.makedirs(cls_dir)
    
    # Run CAMB
    result = os.system("./camb/camb params.ini")
    
    if result == 0:
        print(f"Completed run {run_number}: As={as_value:.1f}×10^-9, r={initial_ratio:.3f}")
    else:
        print(f"Error in run {run_number}: CAMB execution failed")
    
    return result

# Main loop
run_number = 0
total_runs = len(scalar_amp_values) * len(initial_ratio_values)
failed_runs = 0

print(f"Starting CAMB calculations with {total_runs} parameter combinations...")
print("="*60)

for scalar_amp in scalar_amp_values:
    for initial_ratio in initial_ratio_values:
        run_number += 1
        
        # Modify parameter file
        modify_params_ini(scalar_amp, initial_ratio)
        
        # Run CAMB and save results
        result = run_camb_and_save(run_number, scalar_amp, initial_ratio)
        if result != 0:
            failed_runs += 1
        
        print("-" * 40)

print("="*60)
print(f"All CAMB calculations completed!")
print(f"Total runs: {total_runs}")
print(f"Successful runs: {total_runs - failed_runs}")
print(f"Failed runs: {failed_runs}")
print(f"Results saved in cls/ directory with naming convention: cls_As_X.X_r_X.XXX/")
print("="*60)