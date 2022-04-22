import numpy as np
import os
from desidlas.datasets.get_sightlines import get_sightlines
from desidlas.prediction.pred_sightline import save_pred
'''
To run DLA finder on NERSC
1. You can use code from this branch: "git clone -b realdata https://github.com/cosmodesi/desi-dlas.git"
2. Then make sure u have installed Tensorflow. "conda install tensorflow"
3. Download 3 CNN model files as described in desidlas/prediction/model/train_highsnr/highsnrmodel; desidlas/prediction/model/train_midsnr/midsnrmodel; desidlas/prediction/model/train_lowsnr/lowsnrmodel
4. Modify line 18 and line 118 in desidlas/prediction/get_partprediction_sightline.py to relocate to your folder 
5. Put the py file which you want to run in the same directory as desidlas (Just like this file), then run it to get the DLA catalog.
'''

#preprocess spectra
outpath='/global/cfs/cdirs/desi/users/jqzou/test_sightlines.npy'
zbest='/global/cfs/cdirs/desi/spectro/redux/fuji/healpix/sv3/dark/99/9994/redrock-sv3-dark-9994.fits'
spectra='/global/cfs/cdirs/desi/spectro/redux/fuji/healpix/sv3/dark/99/9994/coadd-sv3-dark-9994.fits'
sightlines=get_sightlines(spectra,[],zbest,outpath)

#predict spectra
sightline_loc='/global/cfs/cdirs/desi/users/jqzou/test_sightlines.npy'
partpre_loc='/global/cfs/cdirs/desi/users/jqzou/test_partpre.npy'
os.system("python3 /global/cfs/cdirs/desi/users/jqzou/desidlas/prediction/get_partprediction.py -p %s -o %s"%(sightline_loc,partpre_loc))

#get DLA catalog
pred=np.load(partpre_loc,allow_pickle=True).item()
sightlines=np.load(sightline_loc,allow_pickle=True)
dlacat_loc='/global/cfs/cdirs/desi/users/jqzou/test_dlacatalog.fits'
save_pred(sightlines,pred,filename=dlacat_loc)
