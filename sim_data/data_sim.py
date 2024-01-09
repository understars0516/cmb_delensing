import os
import pylab as pl
import numpy as np
import lenspyx
import healpy as hp
from lenspyx.utils import camb_clfile
from lenspyx.utils_hp import synalm, almxfl, alm2cl
from lenspyx import synfast



def to_eb(Qlen, Ulen, Qunl, Uunl, nside):

    alm_len_p2, alm_len_m2 = hp.map2alm_spin([Qlen, Ulen], spin=2)
    alm_unl_p2, alm_unl_m2 = hp.map2alm_spin([Qunl, Uunl], spin=2)

    Elen_alm = -(alm_len_p2 + alm_len_m2)/2
    Blen_alm = 1j*(alm_len_p2 - alm_len_m2)/2

    Eunl_alm = -(alm_unl_p2 + alm_unl_m2)/2
    Bunl_alm = 1j*(alm_unl_p2 - alm_unl_m2)/2

    Elen_map, Blen_map = hp.alm2map_spin([Elen_alm, Blen_alm], nside=nside, spin=0, lmax=3*nside-1)

    Eunl_map, Bunl_map = hp.alm2map_spin([Eunl_alm, Bunl_alm], nside=nside, spin=0, lmax=3*nside-1)

    return Elen_map, Eunl_map, Blen_map, Bunl_map




i = 0
ws = [-1.025, -1, -0.975][1:2]
rs = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01][0:1]
linears = [2]
splt = int(len(ws)*len(rs)*len(linears))
lenmaps = []; unlenmaps = []; lenmap_arrs = []; unlenmap_arrs = []
nside = 2048; cls_path = './camb/'
print(100*"*")
print("number of maps: %d"%splt)
print(100*"*")

path = './map_data/'
lmax_len = 3000 # desired lmax of the lensed field.
dlmax = 1024  # lmax of the unlensed fields is lmax + dlmax.  (some buffer is required for accurate lensing at lmax)
epsilon = 1e-6 # target accuracy of the output maps (execution time has a fairly weak dependence on this)
lmax_unl, mmax_unl = lmax_len + dlmax, lmax_len + dlmax
lmax_len, mmax_len = lmax_unl,  mmax_unl
Tlens = []; Qlens = []; Ulens = []
Tunls = []; Qunls = []; Uunls = []
for w in ws: 
    for linear in linears:
        for r in rs:
            lensCL = camb_clfile(os.path.join(cls_path, 'Lens_w_%0.3f_r_%0.3f_lensedCls.dat'%(w, r)))
            unlensCL = camb_clfile(os.path.join(cls_path, 'Lens_w_%0.3f_r_%0.3f_lenspotentialCls.dat'%(w, r)))

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
            
            Tlens.append(Tlen); Qlens.append(Qlen); Ulens.append(Ulen)
            Tunls.append(Tunl); Qunls.append(Uunl); Uunls.append(Uunl)

            Elen, Eunl, Blen, Bunl = to_eb(Qlen, Qunl, Ulen, Uunl, nside)
            if True:
                np.save(path + "Tlen_w_%0.3f_r_%0.3f.fits"%(w, r), Tlen)
                np.save(path + "Qlen_w_%0.3f_r_%0.3f.fits"%(w, r), Qlen)
                np.save(path + "Ulen_w_%0.3f_r_%0.3f.fits"%(w, r), Ulen)

                np.save(path + "Tunl_w_%0.3f_r_%0.3f.fits"%(w, r), Tunl)
                np.save(path + "Qunl_w_%0.3f_r_%0.3f.fits"%(w, r), Qunl)
                np.save(path + "Uunl_w_%0.3f_r_%0.3f.fits"%(w, r), Uunl)


np.save("map_data/Tlen_nside2048_%d.npy"%splt, Tlens)
np.save("map_data/Qlen_nside2048_%d.npy"%splt, Qlens)
np.save("map_data/Ulen_nside2048_%d.npy"%splt, Ulens)

np.save("map_data/Tunl_nside2048_%d.npy"%splt, Tunls)
np.save("map_data/Qunl_nside2048_%d.npy"%splt, Qunls)
np.save("map_data/Uunl_nside2048_%d.npy"%splt, Uunls)

#tlm_unl, elm_len, blm_len = hp.map2alm([Tunl, Qunl, Uunl], lmax=mmax_unl, pol = True)
#tlm_len, elm_len, blm_len = hp.map2alm([Tlen, Qlen, Ulen], lmax=mmax_len, pol = True)

#Tlen = hp.read_map("Tlen_w_-1.000_r_0.001.fits")
#Tunl = hp.read_map("Tunl_w_-1.000_r_0.001.fits")

#Qlen = hp.read_map("Qlen_w_-1.000_r_0.001.fits")
#Qunl = hp.read_map("Qunl_w_-1.000_r_0.001.fits")

#Ulen = hp.read_map("Ulen_w_-1.000_r_0.001.fits")
#Uunl = hp.read_map("Uunl_w_-1.000_r_0.001.fits")


#hp.mollview(Tlen, title='lensed T map')
#hp.mollview(Tunl, title='unlensed T map')
#hp.mollzoom(Tlen - Tunl, title='lensed T - unlensed T')

#hp.mollview(Qlen, title='lensed Q map')
#hp.mollview(Qunl, title='unlensed Q map')
#hp.mollzoom(Qlen - Qunl, title='lensed Q - unlensed Q')

#hp.mollview(Ulen, title='lensed U map')
#hp.mollview(Uunl, title='unlensed U map')
#hp.mollzoom(Ulen - Uunl, title='lensed U - unlensed U')

#pl.show()
#pl.close()



#Tlen = np.load("Tlen_w_%0.3f_r_%0.3f.npy"%(w, r))
#Tunl = np.load("Tunl_w_%0.3f_r_%0.3f.npy"%(w, r))
#Qlen = np.load("Qlen_w_%0.3f_r_%0.3f.npy"%(w, r))
#Qunl = np.load("Qunl_w_%0.3f_r_%0.3f.npy"%(w, r))
#Ulen = np.load("Ulen_w_%0.3f_r_%0.3f.npy"%(w, r))
#Uunl = np.load("Uunl_w_%0.3f_r_%0.3f.npy"%(w, r))

geom_info = ('healpix', {'nside':nside})
geom = lenspyx.get_geom(geom_info)
elm_len, blm_len = geom.map2alm_spin([Qlen, Ulen], 2, lmax_len, lmax_len, nthreads=os.cpu_count())
print(1000*"-")
tlm_len = geom.map2alm(Tlen, lmax_len, lmax_len, nthreads=os.cpu_count())
            
elm_unl, blm_unl = geom.map2alm_spin([Qunl, Uunl], 2, lmax_unl, lmax_unl, nthreads=os.cpu_count())
tlm_unl = geom.map2alm(Tlen, lmax_unl, lmax_unl, nthreads=os.cpu_count())

fig,axes = pl.subplots(2, 1)
clbb_len = alm2cl(blm_len, blm_len, lmax_len, lmax_len, lmax_len) # Same as hp.alm2cl
clbb_unl = alm2cl(blm_unl, blm_unl, lmax_unl, lmax_unl, lmax_unl) # Same as hp.alm2cl
ls = np.arange(2, lmax_len + 1)
pl.sca(axes[0])
pl.loglog(ls, clbb_unl[ls], label='unlens BB c2c', c='r')
pl.loglog(ls, unlensCL['bb'][ls], label='unlens BB init', c='b')
pl.loglog(ls, clbb_len[ls], label='lens BB c2c', c='r')
pl.loglog(ls, lensCL['bb'][ls], label='lens BB init', c='b')
pl.legend()
pl.sca(axes[1])
cltt_len = alm2cl(tlm_len, tlm_len, lmax_len, lmax_len, lmax_len)
clee_len = alm2cl(elm_len, elm_len, lmax_len, lmax_len, lmax_len)
cltt_unl = alm2cl(tlm_unl, tlm_unl, lmax_unl, lmax_unl, lmax_unl)
clee_unl = alm2cl(elm_unl, elm_unl, lmax_unl, lmax_unl, lmax_unl)
pl.loglog(ls, cltt_len[ls]/ unlensCL['tt'][ls], label='TT')
pl.loglog(ls, lensCL['tt'][ls]/ unlensCL['tt'][ls], c='k')
pl.loglog(ls, clee_len[ls]/ unlensCL['ee'][ls], label='EE')
pl.loglog(ls, lensCL['ee'][ls]/ unlensCL['ee'][ls], c='k', ls='--')
pl.ylim(0.8, 1.4)
pl.legend()
pl.show()




if False:
    hp.mollview(Tunl, norm='hist', title='Tunl')
    hp.mollview(Qunl, norm='hist', title='Qunl')
    hp.mollview(Tunl, norm='hist', title='Uunl')
    hp.mollview(Tlen, norm='hist', title='Tlen')
    hp.mollview(Tlen, norm='hist', title='Qlen')
    hp.mollview(Tlen, norm='hist', title='Ulen')
    hp.mollview(Tunl - Tlen, norm='hist', title='Tunl - Tlen')
    hp.mollview(Qunl - Qlen, norm='hist', title='Qunl - Qlen')
    hp.mollview(Uunl - Ulen, norm='hist', title='Uunl - Ulen')

    pl.show()


#gl_geom = lenspyx.get_geom(geom_info)
#elm_len, blm_len = gl_geom.map2alm_spin([Qlen, Ulen], 2, lmax_len, lmax_len, nthreads=os.cpu_count())
#tlm_len = gl_geom.map2alm(Tlen, lmax_len, lmax_len, nthreads=os.cpu_count()) 



# plots against pred

#clbb_len = alm2cl(blm_len, blm_len, lmax_len, lmax_len, lmax_len) # lensed   BB  alm   -->   lensed   BB cl
#clbb_unl = alm2cl(blm_unl, blm_unl, lmax_unl, lmax_unl, lmax_unl) # unlensed BB  alm   -->   unlensed BB cl

#clee_len = alm2cl(elm_len, elm_len, lmax_len, lmax_len, lmax_len)
#clee_unl = alm2cl(elm_len, elm_len, lmax_len, lmax_len, lmax_len)

#cltt_len = alm2cl(tlm_len, tlm_len, lmax_len, lmax_len, lmax_len)
#cltt_unl = alm2cl(tlm_unl, tlm_unl, lmax_unl, lmax_unl, lmax_unl)


#ls = np.arange(2, 3000 + 1)

#fig,axes = pl.subplots(3, 1)

#pl.sca(axes[0])
#pl.loglog(ls, clbb_unl[ls], label='unlensed BB map2cl', c='b', linestyle='--')   # plot unlensed BB cl
#pl.plot(ls, unlensCL['bb'][ls], label='unlensed BB cl', c='b')                   # compare unlensed BB cl

#pl.loglog(ls, clbb_len[ls], label='lensed BB map2cl', c='r', linestyle='--')     # plot lensed   BB cl
#pl.plot(ls, lensCL['bb'][ls], label='lensed BB cl', c='r')                       # compare lensed BB cl
#pl.legend()


#pl.sca(axes[1])
#pl.loglog(ls, clee_len[ls], label='lensed EE map2cl', c='b', linestyle='--')     # plot lensed   BB cl
#pl.plot(ls, lensCL['ee'][ls], label='lensed EE cl', c='b')                       # compare lensed BB cl

#pl.loglog(ls, clee_len[ls], label='lensed EE map2cl', c='r', linestyle='--')     # plot lensed   BB cl
#pl.plot(ls, lensCL['ee'][ls], label='lensed EE cl', c='r')                       # compare lensed BB cl
#pl.legend()



#pl.sca(axes[2])
#pl.loglog(ls, cltt_len[ls], label='lensed TT map2cl', c='b', linestyle='--')     # plot lensed   BB cl
#pl.plot(ls, lensCL['tt'][ls], label='lensed TT cl', c='b')                       # compare lensed BB cl

#pl.loglog(ls, cltt_len[ls], label='lensed TT map2cl', c='r', linestyle='--')     # plot lensed   BB cl
#pl.plot(ls, lensCL['tt'][ls], label='lensed TT cl', c='r')                       # compare lensed BB cl
#pl.legend()

#pl.show()
#pl.sca(axes[1])
#cltt_len = alm2cl(tlm_len, tlm_len, lmax_len, lmax_len, lmax_len)
#clee_len = alm2cl(elm_len, elm_len, lmax_len, lmax_len, lmax_len)
##pl.plot(ls, cltt_len[ls]/ unlensCL['tt'][ls], label='TT')
#pl.plot(ls, lensCL['tt'][ls]/ unlensCL['tt'][ls], c='k')
#pl.plot(ls, clee_len[ls]/ unlensCL['ee'][ls], label='EE')
#pl.plot(ls, lensCL['ee'][ls]/ unlensCL['ee'][ls], c='k', ls='--')
#pl.ylim(0.8, 1.4)
#pl.legend()
#pl.show()
