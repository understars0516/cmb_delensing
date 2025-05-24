import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack, ndimage, signal
from astropy.io import fits
import os, sys

def load_cmb_map(file_path):
    try:
        with fits.open(file_path) as hdul:
            cmb_map = hdul[0].data
    except:
        try:
            cmb_map = np.load(file_path)
        except:
            raise ValueError(f"Cannot load CMB map from {file_path}")
    
    return cmb_map

def get_cmb_power_spectrum(cmb_map, pixel_size_arcmin=1.0):
    ny, nx = cmb_map.shape
    fft_map = fftpack.fft2(cmb_map)
    power_2d = np.abs(fft_map)**2
    
    kx = 2 * np.pi * fftpack.fftfreq(nx)
    ky = 2 * np.pi * fftpack.fftfreq(ny)
    
    pixel_size_rad = pixel_size_arcmin * (np.pi / (180 * 60))
    kx /= pixel_size_rad
    ky /= pixel_size_rad
    
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_grid = np.sqrt(kx_grid**2 + ky_grid**2)
    
    ell_grid = k_grid
    ell_max = int(np.max(ell_grid))
    ell_bins = np.arange(0, ell_max + 1)
    
    cl = np.zeros(len(ell_bins))
    counts = np.zeros(len(ell_bins))
    
    for i in range(ny):
        for j in range(nx):
            ell_idx = int(np.round(ell_grid[i, j]))
            if ell_idx < len(ell_bins):
                cl[ell_idx] += power_2d[i, j]
                counts[ell_idx] += 1
    valid_idx = counts > 0
    cl[valid_idx] /= counts[valid_idx]
    cl /= (nx * ny)
    
    return ell_bins, cl

def cmb_wiener_filter(lensed_map, cmb_theory_cl, noise_level=0.0, pixel_size_arcmin=1.0):
    ny, nx = lensed_map.shape
    # Convert to Fourier space
    lensed_fft = fftpack.fft2(lensed_map)
    kx = 2 * np.pi * fftpack.fftfreq(nx)
    ky = 2 * np.pi * fftpack.fftfreq(ny)
    pixel_size_rad = pixel_size_arcmin * (np.pi / (180 * 60))
    kx /= pixel_size_rad
    ky /= pixel_size_rad
    
    # Create grid of k values
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_grid = np.sqrt(kx_grid**2 + ky_grid**2)
    ell_grid = k_grid
    wiener_filter = np.ones_like(ell_grid)
    ell_theory = np.arange(len(cmb_theory_cl))
    cl_interp = np.interp(ell_grid.flatten(), ell_theory, cmb_theory_cl).reshape(ell_grid.shape)
    if noise_level > 0:
        noise_per_pixel = noise_level / pixel_size_arcmin
        noise_power = (noise_per_pixel**2) * np.ones_like(ell_grid)
    else:
        noise_power = 1e-10 * np.ones_like(ell_grid)  # Small value to avoid division by zero
    
    wiener_filter = cl_interp / (cl_interp + noise_power)
    filtered_fft = lensed_fft# * wiener_filter
    filtered_map = fftpack.ifft2(filtered_fft).real
    
    return filtered_map

def quadratic_estimator(lensed_map, cmb_theory_cl=None, noise_level=0.0, pixel_size_arcmin=1.0):
    ny, nx = lensed_map.shape
    if cmb_theory_cl is None:
        _, cmb_theory_cl = get_cmb_power_spectrum(lensed_map, pixel_size_arcmin)
    filtered_map = cmb_wiener_filter(lensed_map, cmb_theory_cl, noise_level, pixel_size_arcmin)
    ky, kx = np.meshgrid(
        fftpack.fftfreq(ny),
        fftpack.fftfreq(nx),
        indexing='ij'
    )
    map_fft = fftpack.fft2(filtered_map)
    grad_x_fft = 1j * kx * map_fft
    grad_y_fft = 1j * ky * map_fft
    grad_x = fftpack.ifft2(grad_x_fft).real
    grad_y = fftpack.ifft2(grad_y_fft).real
    grad_x = signal.detrend(grad_x, axis=0)
    grad_x = signal.detrend(grad_x, axis=1)
    grad_y = signal.detrend(grad_y, axis=0)
    grad_y = signal.detrend(grad_y, axis=1)
    filtered_map2 = cmb_wiener_filter(lensed_map, cmb_theory_cl, noise_level*1.5, pixel_size_arcmin)
    qe_term_x = grad_x * filtered_map2
    qe_term_y = grad_y * filtered_map2
    qe_term_x_fft = fftpack.fft2(qe_term_x)
    qe_term_y_fft = fftpack.fft2(qe_term_y)
    norm_factor = np.ones_like(kx)
    ell_grid = np.sqrt(kx**2 + ky**2) * nx
    norm_factor = 1.0 / (1.0 + (ell_grid/500)**4)
    norm_factor[0, 0] = 0  # Remove DC component
    phi_x_fft = qe_term_x_fft * norm_factor
    phi_y_fft = qe_term_y_fft * norm_factor
    k_sq = kx**2 + ky**2
    k_sq[0, 0] = 1.0  # Avoid division by zero
    
    phi_fft = (phi_x_fft * kx + phi_y_fft * ky) / k_sq
    phi_fft[0, 0] = 0  # Remove DC component
    
    phi_map = fftpack.ifft2(phi_fft).real
    displacement_x_fft = 1j * kx * phi_fft
    displacement_y_fft = 1j * ky * phi_fft
    
    displacement_x = fftpack.ifft2(displacement_x_fft).real
    displacement_y = fftpack.ifft2(displacement_y_fft).real
    
    phi_map = ndimage.gaussian_filter(phi_map, sigma=1.0)
    displacement_x = ndimage.gaussian_filter(displacement_x, sigma=1.0)
    displacement_y = ndimage.gaussian_filter(displacement_y, sigma=1.0)
    
    return phi_map, displacement_x, displacement_y

def delens_cmb_map(lensed_map, displacement_x, displacement_y, iterations=3):
    current_map = lensed_map.copy()
    
    for i in range(iterations):
        y, x = np.meshgrid(np.arange(lensed_map.shape[0]), np.arange(lensed_map.shape[1]), indexing='ij')
        
        sample_points_y = y - displacement_y
        sample_points_x = x - displacement_x
        
        delensed_map = ndimage.map_coordinates(current_map, [sample_points_y.flatten(), sample_points_x.flatten()], order=3, mode='wrap').reshape(lensed_map.shape)
        
        current_map = delensed_map
    
    return delensed_map



def main():
    fields = ['T']
    for field in fields:
        len_maps = np.load("train_datasets/%slen_5maps.npy"%field)
        unl_maps = np.load("train_datasets/%sunl_5maps.npy"%field)
        datas = []
        for j in range(5):
            print(j)
            len_map = np.squeeze(len_maps[j])
            unl_map = np.squeeze(unl_maps[j])
    
            delensed_maps = []
            for i in range(192):
                lensed_map = len_map[i, :, :]
                unlensed_map = unl_map[i, :, :]
        
                ell_u, cl_u = get_cmb_power_spectrum(unlensed_map)
                ell_l, cl_l = get_cmb_power_spectrum(lensed_map)
    
                phi_map, displacement_x, displacement_y = quadratic_estimator(lensed_map, cmb_theory_cl=cl_u, noise_level=0)
        
                delensed_map = delens_cmb_map(lensed_map, displacement_x, displacement_y, iterations=6)
                delensed_maps.append(delensed_map)
            datas.append(np.array(delensed_maps))

        np.save("results_QE/%s_QE_5maps.npy"%field, np.array(datas))
    

if __name__ == "__main__":
    main()
