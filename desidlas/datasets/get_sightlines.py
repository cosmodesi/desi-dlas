from desidlas.datasets.preprocess import estimate_s2n,normalize,rebin
from desidlas.datasets.DesiMock import DesiMock
from desidlas.dla_cnn.defs import best_v
import numpy as np
import os
from os.path import join
from pkg_resources import resource_filename


datafile_path = os.path.join(resource_filename('desidlas', 'tests'), 'datafile')
#load spectra file for testing
spectra= os.path.join(datafile_path, 'spectra', 'spectra-16-1375.fits')
truth=os.path.join(datafile_path, 'spectra', 'truth-16-1375.fits')
zbest=os.path.join(datafile_path, 'spectra', 'zbest-16-1375.fits')


def get_sightlines(spectra,truth,zbest):
    """
    Insert DLAs manually into sightlines without DLAs or only choose sightlines with DLAs
    
    spectra:path of spectra file in DESI(fits format), this is used for the information of spectra like flux,wavelength
    truth:path of truth file in DESI(fits format), this is used for the information of DLAs
    zbest:path of zbest file in DESI(fits format),this is used for the redshift of QSOs
    
    Return
    ---------
    sightlines:list of `dla_cnn.data_model.sightline.Sightline` object
    
    """
    sightlines=[]
    #get folder list

    specs = DesiMock()
    specs.read_fits_file(spectra,truth,zbest)
    keys = list(specs.data.keys())
    for jj in keys:
        sightline = specs.get_sightline(jj,camera = 'all', rebin=False, normalize=True)
        #calculate S/N
        sightline.s2n=estimate_s2n(sightline)
        if (sightline.z_qso >= 2.33)&(sightline.s2n >= 1):#apply filtering
                #only use blue band data
                sightline.flux = sightline.flux[0:sightline.split_point_br]
                sightline.error = sightline.error[0:sightline.split_point_br]
                sightline.loglam = sightline.loglam[0:sightline.split_point_br]
                rebin(sightline, best_v['b'])
                sightlines.append(sightline)
                
    np.save('desidlas/tests/datafile/sightlines-16-1375.npy',sightlines)
    return sightlines
