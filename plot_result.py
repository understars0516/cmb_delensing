import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


def plot(filed, cl_test, cl_pred, cl_label, ls, log=True):
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 1, hspace=0, wspace=0, width_ratios=[1], height_ratios=[1.5, 1])
    ax1, ax2 = gs.subplots(sharex='col', sharey='row')
    factor = ls*(ls+1)/2/np.pi
    if log == True:
        ax1.loglog(ls, factor*cl_test, label = r"$C_\ell^{%s, lens}$"%filed, c='red')
        ax1.loglog(ls, factor*cl_pred, label = r"$C_\ell^{%s, unlens}$"%filed, c='blue')
        ax1.loglog(ls, factor*cl_label, label = r"$C_\ell^{%s, delens}$"%filed, c='orange')
        ax1.set_xlim(1.5, 2000)
        ax1.set_xticks([2, 10, 100, 500, 2000])
        ax1.set_xticklabels(['2', '10', '100', '500', '2000'])
        ax1.set_yticks([1e2, 1e3, 1e4])
        ax1.set_yticklabels([r'$10^2$', r'$10^3$', r'$10^4$'])
        ax1.set_ylabel(r"$\ell(\ell+1)C^{%s}_\ell/2\pi[\mu K]$"%filed)
        ax1.legend(loc='best', ncol=3)

    else:
        ax1.plot(ls, factor*cl_test, label = r"$C_\ell^{%s, lens}$"%filed, c='red')
        ax1.plot(ls, factor*cl_pred, label = r"$C_\ell^{%s, unlens}$"%filed, c='blue')
        ax1.plot(ls, factor*cl_label, label = r"$C_\ell^{%s, delens}$"%filed, c='orange')
        ax1.set_xscale("log")
        ax1.set_ylim(-450, 400)
        ax1.set_yticks([-400, -200,  0, 200, 400])
        ax1.set_yticklabels([r'$-400$',r'$-200$', r'$0$', r'$200$', r'$400$'])
        ax1.set_ylabel(r"$\ell(\ell+1)C^{%s}_\ell/2\pi[\mu K]$"%filed)
        ax1.legend(loc='best', ncol=3)

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


    ax2.plot(ls, y2/y_pre, color = 'pink', label = r"$C_\ell^{%s, delens}/C_\ell^{%s, lens}$"%(filed, filed))
    ax2.plot(ls, y3/y_pre, color = 'lightblue', label = r"$C_\ell^{%s, delens}/C_\ell^{%s, unlens}$"%(filed, filed))

    #ax2.plot(ls, cl_pred/cl_label, 'tab:blue', label = r"$C_\ell^{%s%s, delens}/C_\ell^{%s%s, unlens}$"%(filed, filed, filed, filed))
    ax2.plot(ls, ls/ls, 'tab:gray', label = r"$Ratio$", color='gray')
    ax2.set_xlabel(r"$\rm{Multipole~moment,}~\ell$")
    ax2.set_ylabel(r"$Ratio$")
    ax2.set_ylim(-100, 110)
    ax2.set_yticks([-100, -50, 0, 50, 100])
    ax2.set_xlim(1.5, 2000)
    ax2.set_xticks([2, 10, 100, 500, 2000])
    ax2.set_xticklabels(['2', '10', '100', '500', '2000'])
    ax2.set_ylabel(r"$Ratio$")
    ax2.legend(ncol=3, loc='best')



    for ax in fig.get_axes():
        ax.label_outer()
    plt.savefig("result_best_%s.png"%filed, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.show()


lmax = 2048; rot = 0; supervision = 4

prec1 = 4; prec2 = prec1 + 1 
test_T = np.load("result/T_train_mean.npy").reshape(-1)[rearr]
label_T = np.load("result/T_label_mean.npy").reshape(-1)[rearr]
pred_T = np.load("result/T_pred_mean.npy").reshape(-1)[rearr]*prec1/prec2 + label_T/prec2

test_Q = np.load("result/Q_train_mean.npy").reshape(-1)[rearr]
label_Q = np.load("result/Q_label_mean.npy").reshape(-1)[rearr]
pred_Q = np.load("result/Q_pred_mean.npy").reshape(-1)[rearr]*prec1/prec2 + label_Q/prec2

test_U = np.load("result/U_train_mean.npy").reshape(-1)[rearr]
label_U = np.load("result/U_label_mean.npy").reshape(-1)[rearr]
pred_U = np.load("result/U_pred_mean.npy").reshape(-1)[rearr]*prec1/prec2 + label_U/prec2


field = 3 # 0 for TT; 1 for EE; 2 for BB; 3 for TE
cl_test = hp.anafast([test_T, test_Q, test_U], lmax=lmax)[field]
cl_label = hp.anafast([label_T, label_Q, label_U], lmax=lmax)[field]
cl_pred = hp.anafast([pred_T, pred_Q, pred_U], lmax=lmax)[field]

ell = np.arange(2, 2000, dtype=int)


plot('TT', cl_test, cl_pred, cl_label, ell, log=True)
