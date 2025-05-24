import healpy as hp
import numpy as np
from healpy import Rotator
import matplotlib.pyplot as plt
import lenspyx
import os


# rotate map
def rotate_map(mmap, ra, dec):
    theta, phi = hp.pix2ang(2048, np.arange(len(mmap)))
    rot_theta, rot_phi = Rotator(deg=True, rot=[ra, dec])(theta, phi)
    rot_index = hp.ang2pix(2048, rot_theta, rot_phi)
    rot_map = mmap[rot_index]

    return rot_map

# noise sensity
def sensity(ls, net, theta):
    return net*net/np.exp(-1*ls*(ls+1)*theta*theta/8/np.log(2))

# get noise I/Q/U map
def get_noise(cl_tt_noise, cl_ee_noise, cl_bb_noise, nside):
    ell_max = 2*nside - 1
    cl_noises = [cl_tt_noise, cl_ee_noise, cl_bb_noise,
                 np.zeros_like(cl_tt_noise),
                 np.zeros_like(cl_tt_noise),
                 np.zeros_like(cl_tt_noise)]
    alm_noise = hp.synalm(cl_noises, lmax=ell_max, new=True)

    n_T, n_Q, n_U = hp.alm2map(alm_noise, nside=nside, lmax=ell_max, pol=True)
    return n_T, n_Q, n_U

# add beam effect
def sm_map(T, Q, U, theta):
    T = hp.smoothing(T, fwhm=theta).astype('float32')
    Q = hp.smoothing(Q, fwhm=theta).astype('float32')
    U = hp.smoothing(U, fwhm=theta).astype('float32')

    return T, Q, U

T_map_lens = []
T_map_unls = []

E_map_lens = []
E_map_unls = []

B_map_lens = []
B_map_unls = []



# I/Q/U to T/E/B
nside = 2048
lmax=2000; lmax_len = 2000
ell_max = 3*nside - 1
ells = np.arange(ell_max + 1)
theta = 7.9/60*np.pi/180
noise_T = sensity(ells, 1.5/1e3, theta)
noise_E = sensity(ells, 2.1/1e3, theta)
noise_B = sensity(ells, 2.1/1e3, theta)
ws = [-1.025, -1, -0.975][0:1]
rs = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01][0:1]
for w in ws:
    for r in rs:
        n_T, n_Q, n_U = get_noise(noise_T, noise_E, noise_B, nside) # add noise

        Tlen_temp = hp.read_map("map_data/Tlen_w_%0.3f_r_%0.3f.fits"%(w, r)).astype('float32') + n_T.astype('float32')
        Tunl = hp.read_map("map_data/Tunl_w_%0.3f_r_%0.3f.fits"%(w, r)).astype('float32')
        
        Qlen_temp = hp.read_map("map_data/Qlen_w_%0.3f_r_%0.3f.fits"%(w, r)).astype('float32') + n_Q.astype('float32')
        Qunl = hp.read_map("map_data/Qunl_w_%0.3f_r_%0.3f.fits"%(w, r)).astype('float32')
        
        Ulen_temp = hp.read_map("map_data/Ulen_w_%0.3f_r_%0.3f.fits"%(w, r)).astype('float32') + n_U.astype('float32')
        Uunl = hp.read_map("map_data/Uunl_w_%0.3f_r_%0.3f.fits"%(w, r)).astype('float32')

        Tlen, Qlen, Ulen = sm_map(Tlen_temp, Qlen_temp, Ulen_temp, theta) # add beam effect

        # I/Q/U to T/E/B
        geom_info = ('healpix', {'nside':nside})
        gl_geom = lenspyx.get_geom(geom_info)

        elm_len, blm_len = gl_geom.map2alm_spin([Qlen, Ulen], 2, lmax_len, lmax_len, nthreads=os.cpu_count())
        tlm_len = gl_geom.map2alm(Tlen, lmax_len, lmax_len, nthreads=os.cpu_count())

        elm_unl, blm_unl = gl_geom.map2alm_spin([Qunl, Uunl], 2, lmax_len, lmax_len, nthreads=os.cpu_count())
        tlm_unl = gl_geom.map2alm(Tunl, lmax_len, lmax_len, nthreads=os.cpu_count())

        # alm to map
        Tunl_map = hp.alm2map(tlm_unl, 2048)
        Eunl_map = hp.alm2map(elm_unl, 2048)
        Bunl_map = hp.alm2map(blm_unl, 2048)
        
        Tlen_map = hp.alm2map(tlm_len, 2048)
        Elen_map = hp.alm2map(elm_len, 2048)
        Blen_map = hp.alm2map(blm_len, 2048)

        T_map_lens.append(Tlen_map.astype('float32'))
        E_len_lens.append(Elen_map.astype('float32'))
        B_len_lens.append(Blen_map.astype('float32'))

        T_map_unls.append(Tunl_map.astype('float32'))
        E_len_unls.append(Eunl_map.astype('float32'))
        B_len_unls.append(Bunl_map.astype('float32'))


### rot the T/E/B map
thetas = [-180, -120, -60, 0, 60, 120][0:1]
phis = [-60, -30, 0, 30, 60][0:1]
arr = np.load("../rearr_data/arr_nside2048_192x512x512.npy").astype('int')

for theta in thetas:
    for phi in phis:
        Tlens_rot = []; Tunls_rot = []
        Elens_rot = []; Eunls_rot = []
        Blens_rot = []; Bunls_rot = []
        for i in range(len(T_map_lens)):
            Tlens_rot.append(rotate_map(T_map_lens[i], 0, dec)[arr])
            Tunls_rot.append(rotate_map(T_map_unls[i], 0, dec)[arr])
            
            Elens_rot.append(rotate_map(E_map_lens[i], 0, dec)[arr])
            Eunls_rot.append(rotate_map(E_map_unls[i], 0, dec)[arr])
            
            Blens_rot.append(rotate_map(B_map_lens[i], 0, dec)[arr])
            Bunls_rot.append(rotate_map(B_map_unls[i], 0, dec)[arr])
        np.save("rot_map_data/Tlens_nside_2048_30_rot_theta-%d_phi-%d.npy"%(theta, phi), Tlens_rot)
        np.save("rot_map_data/Tunlens_nside_2048_30_rot_theta-%d_phi-%d.npy"%(theta, phi), Tunlens_rot)

        np.save("rot_map_data/Elens_nside_2048_30_rot_theta-%d_phi-%d.npy"%(theta, phi), Elens_rot)
        np.save("rot_map_data/Eunlens_nside_2048_30_rot_theta-%d_phi-%d.npy"%(theta, phi), Eunlens_rot)

        np.save("rot_map_data/Blens_nside_2048_30_rot_theta-%d_phi-%d.npy"%(theta, phi), Blens_rot)
        np.save("rot_map_data/Bunlens_nside_2048_30_rot_theta-%d_phi-%d.npy"%(theta, phi), Bunlens_rot)

