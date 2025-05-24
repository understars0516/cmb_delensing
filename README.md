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
```bash
# params.ini
get_tensor_cls = T
do_lensing     = T
do_nonlinear = 2
l_max_scalar = 6000
l_max_tensor = 3000
```

I: $$A_s=[2.0\times 10^{-9},2.1\times 10^{-9},2.2\times 10^{-9},2.3\times 10^{-9},2.4\times 10^{-9}]$$, and $$n_s =[0.94, 0.95, 0.96, 0.97, 0.98, 0.99]$$

Q, U:  $$w =[-1.025, -1, -0.975]$$, and $$r =[ 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]$$

Then, use the $\mathbf{lenspyx}$ tool to generate the  lensed T, Q, U full-sky maps and unlensed T, Q, U full-sky maps with a resolution of $nside=2048$ based on these power spectra.

Use the [arr_nside2048_192x512x512.npy](https://drive.google.com/file/d/1Q4QOPStMdreQ2Ic0JqNF2ZLRVPnnczMX/view?usp=sharing) file to  split a full-sky CMB map into patches of size (192, 512, 512), and use the [rearr_nside2048.npy](https://drive.google.com/file/d/1WJbkHwrOCrO-HY24FCAqQi2NcWfqc6fU/view?usp=drive_link)  to restore the sky maps.



### Cloning the Repository

Clone the repository locally using Git:

```bash
git clone https://github.com/understars0516/cmb_delensing.git
cd cmb_delensing
python run.py
```



### ipynb example
We also provide a sample file in the [example.ipynb](https://github.com/understars0516/cmb_delensing/blob/main/example.ipynb) format. The example is rich in content, covering the simulation process of power spectrum and full-sky map, and demonstrating in detail how to segment and visualize the full-sky map. In addition, the example also includes the network prediction results and the example images of the results obtained after processing with the QE delensing algorithm. Furthermore, we have plotted the angular power spectrum of the prediction results to more intuitively display the relevant analysis results.


### run the test datasets
Due to GitHub's size limit on uploaded files, we cannot upload large data files, so we uploaded the relevant data to [Google Drive](https://drive.google.com/drive/folders/1DelPqaa7lX_1CWxzHI10VJnJO5cwJMS0?usp=drive_link). On this cloud storage platform, you can download all program files and supporting data, and these data have been configured properly and can be directly used.
