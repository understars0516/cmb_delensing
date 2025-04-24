import os
import pylab as pl
import numpy as np
import lenspyx
import healpy as hp
from lenspyx.utils import camb_clfile
from lenspyx.utils_hp import synalm, almxfl, alm2cl
from lenspyx import synfast
import matplotlib.pyplot as plt


def to_eb(Qlen, Ulen, Qunl, Uunl, nside):

    alm_len_p2, alm_len_m2 = hp.map2alm_spin([Qlen, Ulen], spin=2)
    alm_unl_p2, alm_unl_m2 = hp.map2alm_spin([Qunl, Uunl], spin=2)

    Elen_alm = -(alm_len_p2 + alm_len_m2)/2
    Blen_alm = 1j*(alm_len_p2 - alm_len_m2)/2

    Eunl_alm = -(alm_unl_p2 + alm_unl_m2)/2
    Bunl_alm = 1j*(alm_unl_p2 - alm_unl_m2)/2

    print(Elen_alm.shape)


    Elen_map, Blen_map = hp.alm2map_spin([Elen_alm, Blen_alm], nside=nside, spin=2, lmax=3*nside-1)

    Eunl_map, Bunl_map = hp.alm2map_spin([Eunl_alm, Bunl_alm], nside=nside, spin=2, lmax=3*nside-1)

    return Elen_map, Eunl_map, Blen_map, Bunl_map




i = 0
ws = [-1.025, -1, -0.975]
rs = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
linears = [2]
splt = int(len(ws)*len(rs)*len(linears))
lenmaps = []; unlenmaps = []; lenmap_arrs = []; unlenmap_arrs = []
nside = 2048; cls_path = './camb/data/'

path = './map_data/'
lmax_len = 3000 # desired lmax of the lensed field.
dlmax = 1024  # lmax of the unlensed fields is lmax + dlmax.  (some buffer is required for accurate lensing at lmax)
epsilon = 1e-6 # target accuracy of the output maps (execution time has a fairly weak dependence on this)
lmax_unl, mmax_unl = lmax_len + dlmax, lmax_len + dlmax
lmax_len, mmax_len = lmax_unl,  mmax_unl
Tlens = []; Qlens = []; Ulens = []
Tunls = []; Qunls = []; Uunls = []

for w in ws:
    for r in rs:
            w = -1
            r = 0.5
            lensCL = camb_clfile(os.path.join(cls_path, 'Lens_lensedCls.dat'))
            unlensCL = camb_clfile(os.path.join(cls_path, 'Lens_lenspotentialCls.dat'))

            print(unlensCL['tt'].shape, lmax_unl)
            tlm_unl = synalm(unlensCL['tt'], lmax=lmax_unl, mmax=mmax_unl)
            elm_unl = synalm(unlensCL['ee'], lmax=lmax_unl, mmax=mmax_unl)
            blm_unl = synalm(unlensCL['bb'], lmax=lmax_unl, mmax=mmax_unl)

            tlm_len = synalm(lensCL['tt'], lmax=lmax_len, mmax=mmax_len)
            elm_len = synalm(lensCL['ee'], lmax=lmax_len, mmax=mmax_len)
            blm_len = synalm(lensCL['bb'], lmax=lmax_len, mmax=mmax_len)

            plm = synalm(unlensCL['pp'], lmax=lmax_unl, mmax=mmax_unl)

            dlm = almxfl(plm, np.sqrt(np.arange(lmax_unl + 1, dtype=float) * np.arange(1, lmax_unl + 2)), None, False)

            geom_info = ('healpix', {'nside':1024}) # here we will use an Healpix grid with nside 2048
            geom = lenspyx.get_geom(geom_info)
            Tunl = lenspyx.alm2lenmap(tlm_unl, dlm*0, geometry=geom_info, verbose=1)
            Qunl, Uunl = lenspyx.alm2lenmap_spin([elm_unl, blm_unl], dlm*0, 2, geometry=geom_info, verbose=1)
            Tlen = lenspyx.alm2lenmap(tlm_unl, dlm, geometry=geom_info, verbose=1)
            Qlen, Ulen = lenspyx.alm2lenmap_spin(elm_unl, dlm, 2, geometry=geom_info, verbose=1)

            hp.write_map("map_data/Tlen_w_%0.3f_r_%0.3f.fits"%(w, r), Tlen, overwrite=True)
            hp.write_map("map_data/Tunl_w_%0.3f_r_%0.3f.fits"%(w, r), Tunl, overwrite=True)
            hp.write_map("map_data/Qlen_w_%0.3f_r_%0.3f.fits"%(w, r), Qlen, overwrite=True)
            hp.write_map("map_data/Qunl_w_%0.3f_r_%0.3f.fits"%(w, r), Qunl, overwrite=True)
            hp.write_map("map_data/Ulen_w_%0.3f_r_%0.3f.fits"%(w, r), Ulen, overwrite=True)
            hp.write_map("map_data/Uunl_w_%0.3f_r_%0.3f.fits"%(w, r), Uunl, overwrite=True)
