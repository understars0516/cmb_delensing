Full-sky maps:
[data_map](https://drive.google.com/drive/folders/17Lk4D_RlJYxAOM3jQmiwM9oqjaHnBxiM?usp=sharing)

Download the lensed and unlensed T, Q, U separately, split them using an [arr_nside2048_192x512x512.npy](https://drive.google.com/file/d/1Q4QOPStMdreQ2Ic0JqNF2ZLRVPnnczMX/view?usp=sharing), and ultimately obtain test data with a shape of (192, 512, 512, 1).

```bash
import healpy as hp
import numpy as np

arr = np.load(arr_nside2048_192x512x512.npy").astype('int')
Tlen = hp.read_map("Tlen_w_-1.000_r_0.010.fits")[arr].reshape(192, 512, 512, 1)
Tunl = hp.read_map("Tunl_w_-1.000_r_0.010.fits")[arr].reshape(192, 512, 512, 1)

```

Rotate the full-sky map according to the angles below.

For I: $A_s$=[ $2.0\times 10^{-9},~ 2.1\times 10^{-9},~ 2.2\times 10^{-9}, ~2.3\times 10^{-9}, ~2.4\times 10^{-9}$] , $n_s$ =[$0.94, 0.95, 0.96, 0.97, 0.98, 0.99$]

For Q, U: Q, U:  $w$ =[$-1.025$, $-1$, $-0.975$],  $r$ =[ $0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01$]



Rot full-sky maps:
[rot_data_map](https://drive.google.com/drive/folders/18XHVer4XZwZM2ptm7uxZFUC-Oi5kswBr?usp=sharing)
Download the lensed and unlensed versions of T, Q, and U separately. These datasets are already segmented and can be directly used for processing with UNet++.
