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

def map2cl(mmap, lmax=2000):
    return hp.anafast(mmap, lmax=lmax)

rearr = np.load("../rearr_nside/rearr_nside2048.npy").astype("int")

num = 3
start = 192*num
end = 192*(num+1)

test_T = np.squeeze(np.load("results_TEB/T_test_epochs_2000.npy")).reshape(-1)[rearr]
label_T = np.squeeze(np.load("results_TEB/T_label_epochs_2000.npy")).reshape(-1)[rearr]
pred_T = np.squeeze(np.load("results_TEB/T_pred_epochs_2000.npy")).reshape(-1)[rearr]
#QE_T = np.squeeze(np.load("results_QE/T_QE_5maps.npy"))[3].reshape(-1)[rearr]
QE_T = hp.read_map("delensed_TEB.fits", field=0)

test_E = np.squeeze(np.load("results_TEB/E_test_epochs_4000.npy"))[start:end].reshape(-1)[rearr]
label_E = np.squeeze(np.load("results_TEB/E_label_epochs_4000.npy"))[start:end].reshape(-1)[rearr]
pred_E = np.squeeze(np.load("results_TEB/E_pred_epochs_4000.npy"))[start:end].reshape(-1)[rearr]
#QE_E = np.squeeze(np.load("results_QE/E_QE_5maps.npy"))[3].reshape(-1)[rearr]
QE_E = hp.read_map("delensed_TEB.fits", field=1)


test_B = np.squeeze(np.load("results_TEB/B_test_epochs_6000.npy"))[start:end].reshape(-1)[rearr]
label_B = np.squeeze(np.load("results_TEB/B_label_epochs_6000.npy"))[start:end].reshape(-1)[rearr]
pred_B = np.squeeze(np.load("results_TEB/B_pred_epochs_6000.npy"))[start:end].reshape(-1)[rearr]
#QE_B = np.squeeze(np.load("results_QE/B_QE_5maps.npy"))[3].reshape(-1)[rearr]
QE_B = hp.read_map("delensed_TEB.fits", field=2)

test_T_cl = map2cl(test_T)
label_T_cl = map2cl(label_T)
pred_T_cl = map2cl(pred_T)
QE_T_cl = map2cl(QE_T)

test_E_cl = map2cl(test_E)
label_E_cl = map2cl(label_E)
pred_E_cl = map2cl(pred_E)
QE_E_cl = map2cl(QE_E)

test_B_cl = map2cl(test_B)
label_B_cl = map2cl(label_B)
pred_B_cl = map2cl(pred_B)
QE_B_cl = map2cl(QE_B)



ell = np.arange(len(test_T_cl))[2:]
factor = ell*(ell+1)

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(1, 1, 1)
#gs = fig.add_gridspec(2, 1, hspace=0, wspace=0, width_ratios=[1], height_ratios=[1.5, 1])
#ax1, ax2 = gs.subplots(sharex='col', sharey='row')

#ax1.loglog(ell, factor*label_T_cl[2:], label = r"$C_\ell^{TT, unlens}$", c='darkred')
#ax1.loglog(ell, factor*pred_T_cl[2:], label = r"$C_\ell^{TT_{UNet++}, delens}$", c='red', linestyle=(0, (1, 5)),linewidth=1)
#ax1.loglog(ell, factor*QE_T_cl[2:], label = r"$C_\ell^{TT_{QE}, delens}$", c='lightcoral',linestyle=(0, (3, 10, 1, 10)), linewidth=1)

#ax1.loglog(ell, factor*label_E_cl[2:], label = r"$C_\ell^{EE, unlens}$", c='darkblue')
#ax1.loglog(ell, factor*pred_E_cl[2:], label = r"$C_\ell^{EE_{UNet++}, delens}$", c='blue', linestyle=(0, (1, 5)),linewidth=1)
#ax1.loglog(ell, factor*QE_E_cl[2:], label = r"$C_\ell^{EE_{QE}, delens}$", c='lightblue',linestyle=(0, (3, 10, 1, 10)), linewidth=1)

#ax1.loglog(ell, factor*label_B_cl[2:], label = r"$C_\ell^{BB, unlens}$", c='darkgreen')
#ax1.loglog(ell, factor*pred_B_cl[2:], label = r"$C_\ell^{BB_{UNet++}, delens}$", c='green', dashes=(10, 5))
#ax1.loglog(ell, factor*QE_B_cl[2:], label = r"$C_\ell^{BB_{QE}, delens}$", c='lightgreen',linestyle=(0, (3, 10, 1, 10)), linewidth=1)



# 自定义线型样式
solid = (0, ())            # 实线
dash1 = (0, (20, 10))       # 长虚线
dot_dash = (0, (3, 20, 3, 10))  # 点划线
short_dash = (0, (20, 20))   # 短虚线

linewidth = 1
#ax1.loglog(ell, factor*test_T_cl[2:], label=r"$C_\ell^{TT, lens}$", c='darkred', linestyle=dash1, linewidth=linewidth)
ax1.loglog(ell, factor*label_T_cl[2:], label=r"$C_\ell^{TT, unlens}$", c='darkred', linestyle=solid, linewidth=linewidth)
ax1.loglog(ell, factor*pred_T_cl[2:], label=r"$C_\ell^{TT_{UNet++}, delens}$", c='red', linestyle=dot_dash, linewidth=linewidth)
ax1.loglog(ell, factor*QE_T_cl[2:], label=r"$C_\ell^{TT_{QE}, delens}$", c='pink', linestyle=dash1, linewidth=linewidth)

# E-E plots
#ax1.loglog(ell, factor*(test_E_cl[2:]+test_B_cl[2:]), label=r"$C_\ell^{EE, lens}$", c='gray', linestyle=dot_dash, linewidth=linewidth)
ax1.loglog(ell, factor*label_E_cl[2:], label=r"$C_\ell^{EE, unlens}$", c='darkblue', linestyle=solid, linewidth=linewidth)
ax1.loglog(ell, factor*pred_E_cl[2:], label=r"$C_\ell^{EE_{UNet++}, delens}$", c='blue', linestyle=dot_dash, linewidth=linewidth)
ax1.loglog(ell, factor*QE_E_cl[2:], label=r"$C_\ell^{EE_{QE}, delens}$", c='lightblue', linestyle=dash1, linewidth=linewidth)

# B-B plots
#ax1.loglog(ell, factor*test_B_cl[2:], label=r"$C_\ell^{BB, lens}$", c='lightgrey', linestyle=dot_dash, linewidth=linewidth)
ax1.loglog(ell, factor*label_B_cl[2:], label=r"$C_\ell^{BB, unlens}$", c='darkgreen', linestyle=solid, linewidth=linewidth)
ax1.loglog(ell, factor*pred_B_cl[2:], label=r"$C_\ell^{BB_{UNet++}, delens}$", c='green', linestyle=dot_dash, linewidth=linewidth)
ax1.loglog(ell, factor*QE_B_cl[2:], label=r"$C_\ell^{BB_{QE}, delens}$", c='lightgreen', linestyle=dash1, linewidth=linewidth)



ax1.set_xlim(1.5, 2000)
ax1.set_xticks([2, 10, 100, 500, 2000])
ax1.set_xticklabels(['2', '10', '100', '500', '2000'])
#ax1.set_ylim(1e-9, 5e2)
#ax1.set_yticks([1e-8, 1e-6, 1e-4, 1e-2,  1e0, 1e2])
#ax1.set_yticklabels([r'$10^{-8}$', r'$10^{-6}$', r'$10^{-4}$',  r'$10^{-2}$', r'$10^{0}$', r'$10^2$'])
ax1.set_ylabel(r"$\ell(\ell+1)C^{XX}_\ell/2\pi[\mu K^2]$")
ax1.legend(loc='best', ncol=3)
ax1.set_xlabel(r"$\rm{Multipole~moment,}~\ell$")


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

# ax2.plot(ls, cltt_len[ls]/ unlensCL['tt'][ls], color = 'pink', label = r"$C_\ell^{%s, unlens}/C_\ell^{%s, lens}$"%(field, field))
#ax2.loglog(ls, unlensCL['%s'%field.lower()][ls]/lensCL['%s'%field.lower()][ls], label = r"$C_\ell^{%s, unlens}/C_\ell^{%s, lens}$"%(field, field), color='red')
#ax2.loglog(ls, ls/ls, color='gray', label=r'$Ratio=1$')
#ax2.loglog(ls, unlensCL['%s'%field1.lower()][ls]/lensCL['%s'%field1.lower()][ls], label = r"$C_\ell^{%s, unlens}/C_\ell^{%s, lens}$"%(field1, field1), color='pink')

#ax2.set_xlabel(r"$\rm{Multipole~moment,}~\ell$")
#ax2.set_ylabel(r"$Ratio$")
#ax2.set_ylim(1e-5, 1e3)
#ax2.set_yticks([1e-4, 1e-2, 1e0, 1e2])
#ax2.set_yticklabels([r'$10^{-4}$', r'$10^{-2}$',  r'$10^{0}$', r'$10^{2}$'])
#ax2.set_xlim(1.5, 2000)
#ax2.set_xticks([2, 10, 100, 500, 2000])
#ax2.set_xticklabels(['2', '10', '100', '500', '2000'])
#ax2.set_ylabel(r"$Ratio$")
#ax2.legend(ncol=2, loc='lower left')


for ax in fig.get_axes():
    ax.label_outer()
plt.savefig("result_cls_single.pdf", bbox_inches='tight', pad_inches=0, dpi=100, format='pdf')
plt.show()

