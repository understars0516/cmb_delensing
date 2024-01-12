# CMB Delensing
---
Read the full publication here:  https://arxiv.org/abs/2310.07358

# Data simulation

### cls simulation with camb
- Download camb from https://github.com/cmbant/CAMB,  and install
- simulate CMB angular power spectrum with different cosmological parameters.

example:
$C_\ell^{TT}$
![TT_cl_ratio](https://github.com/understars0516/cmb_delensing/assets/32385394/f8789e11-4b29-4cd2-a8ca-51bd96a72b8f)

$C_\ell^{EE}$
![EE_cl](https://github.com/understars0516/cmb_delensing/assets/32385394/af135d57-de53-4b49-badb-4515d36571ee)

$C_\ell^{BB}$
![result_best_eb](https://github.com/understars0516/cmb_delensing/assets/32385394/2b5b9c40-cd30-4fa7-a856-2397960b9195)



### map simulation with lenspyx
- Download lenspyx  https://github.com/carronj/lenspyx, and install

example:
T map:
![res_T](https://github.com/understars0516/cmb_delensing/assets/32385394/5a017772-35d3-43ad-af20-456133e697e6)

Q map:
![res_Q](https://github.com/understars0516/cmb_delensing/assets/32385394/9777c9ae-0791-48db-96dd-156d9824567b)

U map:
![res_U](https://github.com/understars0516/cmb_delensing/assets/32385394/ff6d72b1-386c-4825-8692-a408eb948fbb)

# Unet2p:
![unet_2p](https://github.com/understars0516/cmb_delensing/assets/32385394/5b94cd70-e75b-46c3-ac0e-537ccdaf91b4)


# Train Dataset:
![patch_train](https://github.com/understars0516/cmb_delensing/assets/32385394/7063fe35-c678-4b40-a9c2-9aeb3e9cf5a5)

![patch_label](https://github.com/understars0516/cmb_delensing/assets/32385394/93075bb2-060b-41b4-b247-70cd9f308f52)

# Pred Dataset:
![patch_train_test](https://github.com/understars0516/cmb_delensing/assets/32385394/289f0e28-704d-45ac-baad-5c82df95af7c)

![patch_label_test](https://github.com/understars0516/cmb_delensing/assets/32385394/9809c9ec-c3c1-440c-a329-c6b91ae1bbd6)

![patch_train_pred](https://github.com/understars0516/cmb_delensing/assets/32385394/4c3735f9-f53b-4ed4-9572-1babf504715e)

# arr map to images(arr_nside2048_192x512x512.npy), images to map(rearr_nside2048.fits)
https://pan.baidu.com/s/1LW1vVcCRIOpteNgKV2EuEA?pwd=kibp  (code: kibp)

# one universe simulation:
https://pan.baidu.com/s/1liha7rSQSwPgjdU3rHeNsg?pwd=bhdu (code: bhdu)
