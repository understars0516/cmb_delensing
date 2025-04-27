# CMB Delensing

---
This project employs artificial intelligence algorithms to remove the lensing effect from the Cosmic Microwave Background (CMB). Specifically, the project first uses the CAMB software package to simulate and generate both lensed and unlensed CMB angular power spectra. Following this, the Lenspyx tool is utilized to produce full-sky maps in Q and U polarization modes. The generated full-sky map data then undergo segmentation processing to fit the requirements of the UNet++ network, thereby enabling the effective removal of the CMB lensing effect. Read the full publication here:  https://arxiv.org/abs/2310.07358



## Installation

Follow these steps to get the project up and running:

- Install $\mathbf{camb}$

```bash
git clone -b CAMB_v0 --single-branch https://github.com/cmbant/CAMB.git
cd CAMB
make clean
make delete
make all
```

- Install $\mathbf{lenspyx}$

```bash
git clone https://github.com/carronj/lenspyx.git
cd lenspyx
pip install --no-binary ducc0 -e ./
```

- Solve dependency

```bash
pip install 'tensorflow[and-cuda]'==2.14.1
pip install camb==1.4.0
pip install lenspyx==2.0.5
pip install matpltolib==3.6.2
pip install scipy==1.9.3
pip install astropy==6.1.7
pip install healpy==1.16.1
```



### Data Preprocessing

Configure the CAMB $\mathbf{params.ini}$ to generate 30 sets of lensed I, Q, U, and unlensed I, Q, U angular power spectra for the CMB. 

get_tensor_cls = T

do_lensing     = T

I: $A_s$=[ $2.0\times 10^{-9},~ 2.1\times 10^{-9},~ 2.2\times 10^{-9}, ~2.3\times 10^{-9}, ~2.4\times 10^{-9}$] , $n_s$ =[$0.94, ~0.95, ~0.96, ~0.97, ~0.98, ~0.99$]

Q, U:  $w$ =[$-1.025$, $-1$, $-0.975$],  $r$ =[ $0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01$]

Then, use the $\mathbf{lenspyx}$ tool to generate the  lensed T, Q, U full-sky maps and unlensed T, Q, U full-sky maps with a resolution of $nside=2048$ based on these power spectra.

Use the [arr_nside2048_192x512x512.npy](https://drive.google.com/file/d/1Q4QOPStMdreQ2Ic0JqNF2ZLRVPnnczMX/view?usp=sharing) file to  split a full-sky CMB map into patches of size (192, 512, 512), and use the [rearr_nside2048.npy](https://drive.google.com/file/d/1WJbkHwrOCrO-HY24FCAqQi2NcWfqc6fU/view?usp=drive_link)  to restore the sky maps.



### Cloning the Repository

Clone the repository locally using Git:

```bash
git clone https://github.com/understars0516/cmb_delensing.git
cd cmb_delensing
python run.py
```





### Result Processing

```python
import numpy as np
import healpy as hp


epochs = 10000
thetas = [-180, -120, -60, 0, 60, 120]
phis = [-60, -30, 0, 30, 60]
nside = 2048
T_pred = np.zeros(hp.nside2npix(nside))
for theta in thetas:
    for phi in phis:
        T = hp.read_map("result/T_rot_theta_%d_phi_%d_train.npy"%(theta, phi))
        T_pred = T_pred + T
        Q = hp.read_map("result/Q_rot_theta_%d_phi_%d_label.npy"%(theta, phi))
        Q_pred = Q_pred + Q
        U = hp.read_map("result/U_rot_theta_%d_phi_%d_pred.npy"%(theta, phi))
        U_pred = U_pred + U
        
T_pred = T_pred/30
Q_pred = Q_pred/30
U_pred = U_pred/30


```



