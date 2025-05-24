import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import pspy, pixell, os, sys
from pspy.so_config import DEFAULT_DATA_DIR
pixell.colorize.mpl_setdefault("planck")

def sensity(ls, net, theta):
    return net*net/np.exp(-1*ls*(ls+1)*theta*theta/8/np.log(2))


def get_noise(cl_tt_noise, cl_ee_noise, cl_bb_noise):
    nside = 2048
    ell_max = 2*nside - 1
    cl_noises = [cl_tt_noise, cl_ee_noise, cl_bb_noise,
                 np.zeros_like(cl_tt_noise),
                 np.zeros_like(cl_tt_noise),
                 np.zeros_like(cl_tt_noise)]
    alm_noise = hp.synalm(cl_noises, lmax=ell_max, new=True)

    n_T, n_Q, n_U = hp.alm2map(alm_noise, nside=nside, lmax=ell_max, pol=True)


    return n_T, n_Q, n_U


nside = 2048
ell_max = 3*nside - 1
ells = np.arange(ell_max + 1)
theta = 7.9/60*np.pi/180
noise_T = sensity(ells, 1.5/1e3, theta)
noise_E = sensity(ells, 2.1/1e3, theta)
noise_B = sensity(ells, 2.1/1e3, theta)
n_T, n_Q, n_U = get_noise(noise_T, noise_E, noise_B)



nside_low = 4
nside_high = 2048
target_pixel = int(sys.argv[1])

ra_center, dec_center = hp.pix2ang(nside_low, target_pixel, lonlat=True)

Tlen_map = hp.smoothing(np.load("../npys/Tlen_w_-1.000_r_0.005.npy") + n_T, fwhm=theta)
Tunl_map = np.load("../npys/Tunl_w_-1.000_r_0.005.npy")

Qlen_map = hp.smoothing(np.load("../npys/Qlen_w_-1.000_r_0.005.npy") + n_Q, fwhm=theta)
Qunl_map = np.load("../npys/Qunl_w_-1.000_r_0.005.npy")

Ulen_map = hp.smoothing(np.load("../npys/Ulen_w_-1.000_r_0.005.npy") + n_U, fwhm=theta)
Uunl_map = np.load("../npys/Uunl_w_-1.000_r_0.005.npy")



titles = ['Lensed T Patch', 'Lensed Q Patch', 'Lensed U Patch', 'Unlensed T Patch', 'Unlensed Q Patch', 'Unlensed U Patch']

datas = [Tlen_map, Qlen_map, Ulen_map, Tunl_map, Qunl_map, Uunl_map]

plt.figure(figsize=(11, 8)) # Adjust figsize to your needs
for i in range(1, 7):
    ax = plt.subplot(2, 3, i)
    hp.gnomview(
    datas[i-1],
    rot=(ra_center, dec_center, 0),
    reso=hp.nside2resol(nside_high, arcmin=True),         # 分辨率（角分/像素）
    xsize=512,        # 图像宽度
    ysize=512,        # 图像高度（与宽度相同）
    title=titles[i-1],
    notext=False,
    cbar=True, 
    return_projected_map=False,
    sub=(2, 3, i))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
plt.savefig("patch_show.pdf", format='pdf', bbox_inches='tight', pad_inches=0, dpi=70)
plt.show()
