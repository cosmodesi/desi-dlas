import numpy as np
from desidlas.dla_cnn.spectra_utils import get_lam_data
#from dla_cnn.data_model.DataMarker import Marker
#no Marker in the mock spectra
from scipy.interpolate import interp1d
from os.path import join, exists
from os import remove
import csv

from desidlas.dla_cnn import defs
#load basic parameter value 
REST_RANGE = defs.REST_RANGE
kernel = defs.kernel

#load modules in preprocess.py
from desidlas.datasets.preprocess import label_sightline
from desidlas.datasets.preprocess import rebin
from desidlas.datasets.preprocess import normalize
from desidlas.datasets.preprocess import estimate_s2n
from desidlas.datasets.preprocess import generate_summary_table
from desidlas.datasets.preprocess import write_summary_table
