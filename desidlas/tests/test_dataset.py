import numpy as np
import scipy.signal as signal
from desidlas.datasets.datasetting import split_sightline_into_samples,select_samples_50p_pos_neg,pad_sightline
from desidlas.datasets.preprocess import label_sightline
from desidlas.dla_cnn.spectra_utils import get_lam_data

sightline_path='desidlas/tests/datafile/sightlines-16-1375.npy'

sightlines=np.load(sightline_path,allow_pickle = True,encoding='latin1')

from desidlas.datasets.get_dataset import make_datasets

def test_datasets:
  
  dataset=make_datasets(sightlines,validate=True)
