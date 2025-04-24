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


Rot full-sky maps:
[rot_data_map](https://drive.google.com/drive/folders/18XHVer4XZwZM2ptm7uxZFUC-Oi5kswBr?usp=sharing)
Download the lensed and unlensed versions of T, Q, and U separately. These datasets are already segmented and can be directly used for processing with UNet++.
