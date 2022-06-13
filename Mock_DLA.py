from fitsio import FITS
import fitsio
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import numpy as np
import os
from matplotlib.pyplot import rcParams
import matplotlib._color_data as mcd
import math
rcParams['figure.figsize'] = 10, 5
rcParams['lines.linewidth'] = 2
rcParams['axes.labelsize'] = 15
rcParams['legend.fontsize'] = 12

import h5py
from scipy import interpolate
from picca import wedgize

from desidlas.datasets.preprocess import estimate_s2n,normalize,rebin
from desidlas.datasets.DesiMock import DesiMock
from desidlas.dla_cnn.defs import best_v
import numpy as np
import os
from os.path import join
from pkg_resources import resource_filename
from pathlib import Path
from desidlas.datasets.get_sightlines import get_sightlines


sightline_all = []
### Set directory to desi spectra:
spectrapath='/global/cfs/cdirs/desi/users/hiramk/desi/everest/main/mock/london/v9.0.0/everest_main-0.134/spectra-16'
### Set saving path for sightlines:
savepath='/global/cfs/cdirs/desi/users/hiramk/desi/everest/main/mock/london/v9.0.0/everest_main-0.134/sightlines'
item1 = os.listdir(savepath)
for k in item1:
    itemlist=os.listdir(savepath+'/'+str(k))
    for j in itemlist:
        sightline_all=sightline_all+np.load(savepath+'/'+str(k)+'/'+str(j),allow_pickle = True,encoding='latin1').tolist()


import scipy.signal as signal
from desidlas.datasets.datasetting import split_sightline_into_samples,select_samples_50p_pos_neg,pad_sightline
from desidlas.datasets.preprocess import label_sightline
from desidlas.dla_cnn.spectra_utils import get_lam_data
from desidlas.datasets.get_dataset import make_datasets,make_smoothdatasets

dataset=make_datasets(sightline_all,validate=True,output='/global/cfs/cdirs/desi/users/hiramk/desi/everest/main/mock/london/v9.0.0/everest_main-0.134/dlafinder/dataset.npy')
