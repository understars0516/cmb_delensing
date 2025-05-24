import os
import pylab as pl
import numpy as np
import lenspyx
import healpy as hp
from lenspyx.utils import camb_clfile
from lenspyx.utils_hp import synalm, almxfl, alm2cl
from lenspyx import synfast
import matplotlib.pyplot as plt
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

def sensity(ls, net):
    return net*net/np.exp(-1*ls*(ls+1)*theta*theta/8/np.log(2))


def sensity_value(ls, net, fsky, ndet, Y, deltaT, theta):
    return ((net*np.sqrt(4*np.pi*fsky))/(np.sqrt(ndet*Y*deltaT)))

i = 0
ws = [-1.025, -1, -0.975][0:1]
rs = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01][0:1]
linears = [2]
splt = int(len(ws)*len(rs)*len(linears))
lenmaps = []; unlenmaps = []; lenmap_arrs = []; unlenmap_arrs = []
nside = 2048; cls_path = '/home/nisl/Works/CMB_Delensing/Gen_Data/camb/cls/'
print(100*"*")
print("number of maps: %d"%splt)
print(100*"*")

path = '/home/nisl/Data/CMB_DeLensing/fits/'

lmax_len = 4000 # desired lmax of the lensed field.
dlmax = 1024  # lmax of the unlensed fields is lmax + dlmax.  (some buffer is required for accurate lensing at lmax)
epsilon = 1e-6 # target accuracy of the output maps (execution time has a fairly weak dependence on this)
lmax_unl, mmax_unl = lmax_len + dlmax, lmax_len + dlmax
lmax_len, mmax_len = lmax_unl,  mmax_unl
w = -1; r = 0.005
lensCL = camb_clfile(os.path.join(cls_path, 'Lens_w_%0.3f_r_%0.3f_lensedCls.dat'%(w, r)))
unlensCL = camb_clfile(os.path.join(cls_path, 'Lens_w_%0.3f_r_%0.3f_lenspotentialCls.dat'%(w, r)))

field0 = 'TT'
field='EE'
field1='BB'

ls = np.arange(2, nside)
factor = ls*(ls+1)/2/np.pi
net = 2.1; fsky = 0.8; ndet = 1020; Y = 1; deltaT = 24*3600;
theta = 7.9/60*np.pi/180

net = sensity_value(ls, net, fsky, ndet, Y, deltaT, theta)

noise_T = sensity(ls, 1.5/1e3)
noise_P = sensity(ls, 2.1/1e3)

#print(noise_E[-10:])
fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(2, 1, hspace=0, wspace=0, width_ratios=[1], height_ratios=[1.5, 1])
ax1, ax2 = gs.subplots(sharex='col', sharey='row')
# ax1.plot(ls, factor*lensCL['tt'][ls], label = r"$C_\ell^{%s, lens}$"%field, c='blue')
# ax1.plot(ls, factor*unlensCL['tt'][ls], label = r"$C_\ell^{%s, unlens}$"%field, c='orange')
# ax1.plot(ls, noise, label = r"$Noise$", c='orange')
ax1.loglog(ls, factor*lensCL['%s'%field0.lower()][ls], label = r"$C_\ell^{%s, lens}$"%field0, c='red')
ax1.loglog(ls, factor*unlensCL['%s'%field0.lower()][ls], label = r"$C_\ell^{%s, unlens}$"%field0, c='salmon', linestyle='--')


ax1.loglog(ls, factor*lensCL['%s'%field.lower()][ls], label = r"$C_\ell^{%s, lens}$"%field, c='green')
ax1.loglog(ls, factor*unlensCL['%s'%field.lower()][ls], label = r"$C_\ell^{%s, unlens}$"%field, c='lightgreen', linestyle='--')


ax1.loglog(ls, factor*(lensCL['%s'%field1.lower()][ls] + unlensCL['%s'%field1.lower()][ls]), label = r"$C_\ell^{%s, lens}$"%field1, c='blue')
ax1.loglog(ls, factor*unlensCL['%s'%field1.lower()][ls], label = r"$C_\ell^{%s, unlens}$"%field1, c='lightblue', linestyle='--')

ax1.loglog(ls, factor*noise_T, label = r"$N_\ell^{TT, noise}$", c='black')#, linestyle='-')
ax1.loglog(ls, factor*noise_P, label = r"$N_\ell^{EE/BB, noise}$", c='lightgrey')#, linestyle='--')

ax1.set_xlim(1.5, 2000)
ax1.set_xticks([2, 10, 100, 500, 2000])
ax1.set_xticklabels(['2', '10', '100', '500', '2000'])
ax1.set_ylim(1e-9, 2e4)
ax1.set_yticks([1e-8, 1e-6, 1e-4, 1e-2,  1e0, 1e2, 1e4])
ax1.set_yticklabels([r'$10^{-8}$', r'$10^{-6}$', r'$10^{-4}$',  r'$10^{-2}$', r'$10^{0}$', r'$10^2$', r'$10^4$'])
ax1.set_ylabel(r"$\ell(\ell+1)C^{XX}_\ell/2\pi[\mu K]$")
ax1.legend(loc='best', ncol=4)


ax_top = ax1.twiny()
ax_top.spines.top.set_position(("axes", 1))  # 调整上部axis的位置
ax_top.set_xscale("log")
ax_top.set_xlim(1.5, 2000)
ax_top.set_xticks([2, 10, 100, 500, 2000])
ax_top.set_xticklabels([r'$90^\circ$', r'$18^\circ$', r'$1.8^\circ$', r'$0.36^\circ$', r'$0.09^\circ$'])
ax_top.set_xlabel("Angular scale")
ax_top.spines['left'].set_visible(False)
ax_top.spines['bottom'].set_visible(False)
ax_top.spines['right'].set_visible(False)

ax2.loglog(ls, unlensCL['%s'%field0.lower()][ls]/lensCL['%s'%field0.lower()][ls], label = r"$C_\ell^{%s, unlens}/C_\ell^{%s, lens}$"%(field0, field0), color='red', linestyle='--')
ax2.loglog(ls, unlensCL['%s'%field.lower()][ls]/lensCL['%s'%field.lower()][ls], label = r"$C_\ell^{%s, unlens}/C_\ell^{%s, lens}$"%(field, field), color='green', linestyle='--')
ax2.loglog(ls, unlensCL['%s'%field1.lower()][ls]/lensCL['%s'%field1.lower()][ls], label = r"$C_\ell^{%s, unlens}/C_\ell^{%s, lens}$"%(field1, field1), color='blue', linestyle='--')
#ax2.loglog(ls, ls/ls, color='gray', label=r'$Ratio=1$')

ax2.minorticks_off()
ax2.set_xlabel(r"$\rm{Multipole~moment,}~\ell$")
ax2.set_ylabel(r"$Ratio$")
ax2.set_ylim(1e-4, 1e3)
ax2.set_yticks([1e-4, 1e-2, 1e0, 1e2])
ax2.set_yticklabels([r'$10^{-4}$', r'$10^{-2}$',  r'$10^{0}$', r'$10^{2}$'])
ax2.set_xlim(1.5, 2000)
ax2.set_xticks([2, 10, 100, 500, 2000])
ax2.set_xticklabels(['2', '10', '100', '500', '2000'])
ax2.set_ylabel(r"$Ratio$")
ax2.legend(ncol=2, loc='lower left')


for ax in fig.get_axes():
    ax.label_outer()
plt.savefig("TT_EE_BB_cl_ratio.pdf", bbox_inches='tight', pad_inches=0, dpi=300, format='pdf')
plt.show()

