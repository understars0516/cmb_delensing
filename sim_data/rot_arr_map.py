import healpy as hp
import numpy as np
from healpy import Rotator
import matplotlib.pyplot as plt

def rotate_map(mmap, ra, dec):
    theta, phi = hp.pix2ang(2048, np.arange(len(mmap)))
    rot_theta, rot_phi = Rotator(deg=True, rot=[ra, dec])(theta, phi)
    rot_index = hp.ang2pix(2048, rot_theta, rot_phi)
    rot_map = mmap[rot_index]

    return rot_map

TQU = 'T'
Tlen = np.load("map_data/%slen_nside2048_30.npy"%TQU)
Tunlen = np.load("map_data/%sunlen_nside2048_30.npy"%TQU)

arr = np.load("../data/arr_nside2048_192x512x512.npy").astype('int')

thetas = [-180, -120, -60, 0, 60, 120]
phis = [-60, -30, 0, 30, 60]

for theta in thetas:
    for phi in phis:
        lens = []; unlens = []
        for i in range(30):
            lens.append(rotate_map(len[i], 0, dec)[arr])
            unlens.append(rotate_map(unlen[i], 0, dec)[arr])
        lens = np.array(lens)
        unlens = np.array(unlens)
        np.save("rot_map_data/%slens_nside_2048_30_rot_theta-%d_phi-%d.npy"%(TQU, theta, phi), lens)
        np.save("rot_map_data/%sunlens_nside_2048_30_rot_theta-%d_phi-%d.npy"%(TQU, theta, phi), unlens)

