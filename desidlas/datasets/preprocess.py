""" Code for pre-processing DESI data"""


from desidlas.dla_cnn.spectra_utils import get_lam_data
from desidlas.dla_cnn import defs
REST_RANGE = defs.REST_RANGE
kernel = defs.kernel


def label_sightline(sightline, kernel=kernel, REST_RANGE=REST_RANGE, pos_sample_kernel_percent=0.3):
    """
    Add labels to input sightline based on the DLAs along that sightline
    """
    lam, lam_rest, ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)
    samplerangepx = int(kernel*pos_sample_kernel_percent/2)
    ix_dlas=[]
    coldensity_dlas=[]
    for dla in sightline.dlas:
        if (912<(dla.central_wavelength/(1+sightline.z_qso))<1220) & (dla.central_wavelength>=3700):
            ix_dlas.append((abs(lam[ix_dla_range]-dla.central_wavelength)).argmin())
            coldensity_dlas.append(dla.col_density)

    classification = np.zeros((sum(ix_dla_range)), dtype=np.float32)
    for ix_dla in ix_dlas:
        classification[ix_dla-samplerangepx*2:ix_dla+samplerangepx*2+1] = -1
        lyb_ix = sightline.get_lyb_index(ix_dla)
        classification[lyb_ix-samplerangepx:lyb_ix+samplerangepx+1] = -1
    for ix_dla in ix_dlas:
        classification[ix_dla-samplerangepx:ix_dla+samplerangepx+1] = 1

    offsets_array = np.full([sum(ix_dla_range)], np.nan, dtype=np.float32)
    column_density = np.full([sum(ix_dla_range)], np.nan, dtype=np.float32)
    for i in range(int(samplerangepx+1)):
        for ix_dla,j in zip(ix_dlas,range(len(ix_dlas))):
            offsets_array[ix_dla+i] = -i if np.isnan(offsets_array[ix_dla+i]) else offsets_array[ix_dla+i]
            offsets_array[ix_dla-i] =  i if np.isnan(offsets_array[ix_dla-i]) else offsets_array[ix_dla-i]
            column_density[ix_dla+i] = coldensity_dlas[j] if np.isnan(column_density[ix_dla+i]) else column_density[ix_dla+i]
            column_density[ix_dla-i] = coldensity_dlas[j] if np.isnan(column_density[ix_dla-i]) else column_density[ix_dla-i]
    offsets_array = np.nan_to_num(offsets_array)
    column_density = np.nan_to_num(column_density)

    sightline.classification = classification
    sightline.offsets = offsets_array
    sightline.column_density = column_density

    return classification, offsets_array, column_density

''' Basic Recipe
0. Load the DESI mock spectrum
1. Resample to a constant dlambda/lambda dispersion
2. Renomalize the flux?
3. Generate a Sightline object with DLAs
4. Add labels 
5. Write to disk (numpy or TF)
'''

import numpy as np
from scipy.interpolate import interp1d
from desidlas.parameters import norm_range
import itertools

def rebin(sightline, v):
    """
    Resample and rebin the input Sightline object's data to a constant dlambda/lambda dispersion.
    Parameters
    ----------
    sightline: :class:`dla_cnn.data_model.Sightline.Sightline`
    v: float, and np.log(1+v/c) is dlambda/lambda, its unit is m/s, c is the velocity of light
    Returns
    -------
    :class:`dla_cnn.data_model.Sightline.Sightline`:
    """
    # TODO -- Add inline comments
    c = 2.9979246e8

    dlnlambda = np.log(1+v/c)
    wavelength = 10**sightline.loglam
    max_wavelength = wavelength[-1]
    min_wavelength = wavelength[0]
    pixels_number = int(np.round(np.log(max_wavelength/min_wavelength)/dlnlambda))+1
    new_wavelength = wavelength[0]*np.exp(dlnlambda*np.arange(pixels_number))
    
    npix = len(wavelength)
    wvh = (wavelength + np.roll(wavelength, -1)) / 2.
    wvh[npix - 1] = wavelength[npix - 1] + \
                    (wavelength[npix - 1] - wavelength[npix - 2]) / 2.
    dwv = wvh - np.roll(wvh, 1)
    dwv[0] = 2 * (wvh[0] - wavelength[0])
    med_dwv = np.median(dwv)
    
    cumsum = np.cumsum(sightline.flux * dwv)
    cumvar = np.cumsum(sightline.error * dwv, dtype=np.float64)
    
    fcum = interp1d(wvh, cumsum,bounds_error=False)
    fvar = interp1d(wvh, cumvar,bounds_error=False)
    
    nnew = len(new_wavelength)
    nwvh = (new_wavelength + np.roll(new_wavelength, -1)) / 2.
    nwvh[nnew - 1] = new_wavelength[nnew - 1] + \
                     (new_wavelength[nnew - 1] - new_wavelength[nnew - 2]) / 2.
    
    bwv = np.zeros(nnew + 1)
    bwv[0] = new_wavelength[0] - (new_wavelength[1] - new_wavelength[0]) / 2.
    bwv[1:] = nwvh
    
    newcum = fcum(bwv)
    newvar = fvar(bwv)
    
    new_fx = (np.roll(newcum, -1) - newcum)[:-1]
    new_var = (np.roll(newvar, -1) - newvar)[:-1]
    
    # Normalize (preserve counts and flambda)
    new_dwv = bwv - np.roll(bwv, 1)
    new_fx = new_fx / new_dwv[1:]
    # Preserve S/N (crudely)
    med_newdwv = np.median(new_dwv)
    new_var = new_var / (med_newdwv/med_dwv) / new_dwv[1:]
    
    left = 0
    while np.isnan(new_fx[left])|np.isnan(new_var[left]):
        left = left+1
    right = len(new_fx)
    while np.isnan(new_fx[right-1])|np.isnan(new_var[right-1]):
        right = right-1
    
    test = np.sum((np.isnan(new_fx[left:right]))|(np.isnan(new_var[left:right])))
    assert test==0, 'Missing value in this spectra!'
    
    sightline.loglam = np.log10(new_wavelength[left:right])
    sightline.flux = new_fx[left:right]
    sightline.error = new_var[left:right]
    
    return sightline


def normalize(sightline, full_wavelength, full_flux,norm_range=norm_range):
    '''
    Normalize spectrum by dividing the mean value of continnum at lambda[left,right]
    ------------------------------------------
    parameters:
    
    sightline: dla_cnn.data_model.Sightline.Sightline object;
    camera : str, 'b' : the blue channel of the specctra, 'r': the r channel of the spectra,
                  'z' : the z channel of the spectra, 'all': all spectra
    
    --------------------------------------------
    return
    
    sightline: the sightline after normalized
    
    '''
    blue_limit = norm_range[0]
    red_limit = norm_range[-1]
    rest_wavelength = full_wavelength/(sightline.z_qso+1)
    assert blue_limit <= red_limit,"No Lymann-alpha forest, Please check this spectra: %i"%sightline.id#when no lymann alpha forest exists, assert error.
    #use the slice we chose above to normalize this spectra, normalize both flux and error array using the same factor to maintain the s/n.
    good_pix = (rest_wavelength>=blue_limit)&(rest_wavelength<=red_limit)
    normalizer=np.abs(np.nanmedian(full_flux[good_pix]))#normalizer must be e positive
    sightline.flux = sightline.flux/normalizer
    sightline.error = sightline.error/normalizer
    
def estimate_s2n(sightline,norm_range=norm_range):
    """
    Estimate the s/n of a given sightline, using the lymann forest part and excluding dlas.
    -------------------------------------------------------------------------------------
    parameters；
    sightline: class:`dla_cnn.data_model.sightline.Sightline` object, we use it to estimate the s/n,
               and since we use the lymann forest part, the sightline's wavelength range should contain 1070~1170
    --------------------------------------------------------------------------------------
    return:
    s/n : float, the s/n of the given sightline.
    """
    #determine the lymann forest part of this sightline
    blue_limit = norm_range[0]
    red_limit = norm_range[-1]
    wavelength = 10**sightline.loglam
    rest_wavelength = wavelength/(sightline.z_qso+1)
    #lymann forest part of this sightline, contain dlas 
    test = (rest_wavelength>blue_limit)&(rest_wavelength<red_limit)
    s2n = sightline.flux/sightline.error
    return np.abs(np.nanmedian(s2n[test]))

def cut_zmin(sightline,s2nlevel=3,lybx=3):
    '''
    Estimate the median S/N for each pixel (smoothed by 10 pixels), to determine the minimum wavelength for statistical analysis.
     -------------------------------------------------------------------------------------
    parameters；
    sightline: class:`dla_cnn.data_model.sightline.Sightline` object, we use it to estimate the s/n
    s2nlevel: minimum S/N level for spectra
    lybx: the S/N would be higher because of the lyb emision, so we use lybx*s2nlevel as the cut.
    --------------------------------------------------------------------------------------
    return:
    lammin: the start wavelength of statistical analysis
    '''
    lamrange=[]
    s2nlist=[]
    lyb_emi=1025*(1+sightline.z_qso)
    #lyb region: 10000km/s around lyb emission
    dlam=1e4*lyb_emi/3e5
    lyb_range=np.abs(10**sightline.loglam[10:len(sightline.loglam)-10]-lyb_emi)<dlam
    flux_matrix=np.array(list(map(lambda x:x[0][x[1]-10:x[1]+10],zip(itertools.repeat(sightline.flux), range(10,len(sightline.loglam)-10)))))
    error_matrix=np.array(list(map(lambda x:x[0][x[1]-10:x[1]+10],zip(itertools.repeat(sightline.error), range(10,len(sightline.loglam)-10)))))
    s2n=np.nanmedian(flux_matrix/error_matrix,axis=1)
    s2n[lyb_range]=s2n[lyb_range]/lybx
    try:
        lammin=10**sightline.loglam[10:len(sightline.loglam)-10][s2n>s2nlevel][0]
        assert lammin>3600, 'wrong lam min'
    except:
        lammin=0
    return lammin

def cut_z_region(sightline,red_kms=3000, lam_start=1025,blue_kms=5000,lam_end=1215.67, lam_start_instru=3700,s2nlevel=3,lybx=3):
    '''
    Estimate redshift region for statistical analysis.
     -------------------------------------------------------------------------------------
    parameters；
    sightline: class:`dla_cnn.data_model.sightline.Sightline` object.
    red_kms: the distance further than the start wavelength.
    lam_start: start wavelength at rest-frame.
    blue_kms: the distance closer than the end wavelength.
    lam_end: end wavelength at rest-frame.
    lam_start_instru: Considering the data quality, we set a start fixed wavelength.
    N12: set a fixed range for start lam, if we use this method, N12=True.
    --------------------------------------------------------------------------------------
    return:
    z_start: the start redshift of statistical analysis
    z_end: the end redshift of statistical analysis
    z_use: If we use this sightline to do statistical analysis, z_use=True, otherwise z_use=False.
    '''
    speed_of_light = 3e5
    z_qso=sightline.z_qso
    z_instru=lam_start_instru/1215.67 -1 #2.04
    #estimate the s2n>3 pixel in spectrum to be the minimum redshift
    lammin=cut_zmin(sightline,s2nlevel=s2nlevel,lybx=lybx)
    if lammin>0:
        z_min=lammin/1215.67 -1
        assert z_min>0
        z_start = lam_start * (1 + z_qso) / 1215.67 - 1 + red_kms / speed_of_light
        z_end = lam_end * (1 + z_qso) / 1215.67 - 1- blue_kms / speed_of_light
        z_start = max(z_start,z_min,z_instru)
        if z_start >= z_end:
            print('%s:wrong rest-frame region'%sightline.id)
            z_use = 0
        else:
            z_use = 1
    else:
        print('Can not get S/N>%s wave, zmin=0 :%s'%(s2nlevel,sightline.id))
        z_use = 0
        z_start = 0
        z_end = 0
    return z_start, z_end, z_use



    


