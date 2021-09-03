#this module is used to combine the model's predictions for each pixel into the one sightline with several DLAs' predictions.
import numpy as np 
#from desidlas.data_model.Sightline import Sightline
from desidlas.data_model.Prediction import Prediction
from desidlas.dla_cnn.spectra_utils import get_lam_data
from desidlas.datasets.datasetting import split_sightline_into_samples
from astropy.table import Table
import scipy.signal as signal
def compute_peaks(sightline,PEAK_THRESH):
    """
    Calculate the peak for offsets value
    
    Parameters:
    -----------------------------------------------
    sightline: 'dla_cnn.data_model.Sightline' object
    PEAK_THRESH: float, the threshold to accept a peak=0.2
    
    Returns
    -----------------------------------------------
    sightline
    
    """
    PEAK_SEPARATION_THRESH = 0.1        # Peaks must be separated by a valley at least this low

    # Translate relative offsets to histogram
    offset_to_ix = np.arange(len(sightline.prediction.offsets)) + sightline.prediction.offsets
    offset_to_ix[offset_to_ix < 0] = 0
    offset_to_ix[offset_to_ix >= len(sightline.prediction.offsets)] = len(sightline.prediction.offsets)
    offset_hist, ignore_offset_range = np.histogram(offset_to_ix, bins=np.arange(0,len(sightline.prediction.offsets)+1))

    # Somewhat arbitrary normalization
    offset_hist = offset_hist / 80.0

    po = np.pad(offset_hist, 2, 'constant', constant_values=np.mean(offset_hist))
    offset_conv_sum = (po[:-4] + po[1:-3] + po[2:-2] + po[3:-1] + po[4:])
    smooth_conv_sum = signal.medfilt(offset_conv_sum, 9)
    # ensures a 0 value at the beginning and end exists to avoid an unnecessarily pathalogical case below
    smooth_conv_sum[0] = 0
    smooth_conv_sum[-1] = 0

    peaks_ixs = []
    while True:
        peak = np.argmax(smooth_conv_sum)   # Returns the first occurace of the max
        # exit if we're no longer finding valid peaks
        if smooth_conv_sum[peak] < PEAK_THRESH:
            break
        # skip this peak if it's off the end or beginning of the sightline
        if peak <= 10 :#or peak >= REST_RANGE[2]-10:
            smooth_conv_sum[max(0,peak-15):peak+15] = 0
            continue
        # move to the middle of the peak if there are multiple equal values
        ridge = 1
        while smooth_conv_sum[peak] == smooth_conv_sum[peak+ridge]:
            ridge += 1
        peak = peak + ridge//2
        peaks_ixs.append(peak)

        # clear points around the peak, that is, anything above PEAK_THRESH in order for a new DLA to be identified the peak has to dip below PEAK_THRESH
        clear_left = smooth_conv_sum[0:peak+1] < PEAK_SEPARATION_THRESH # something like: 1 0 0 1 1 1 0 0 0 0
        clear_left = np.nonzero(clear_left)[0][-1]+1                    # Take the last value and increment 1
        clear_right = smooth_conv_sum[peak:] < PEAK_SEPARATION_THRESH   # something like 0 0 0 0 1 1 1 0 0 1
        clear_right = np.nonzero(clear_right)[0][0]+peak                # Take the first value & add the peak offset
        smooth_conv_sum[clear_left:clear_right] = 0

    sightline.prediction.peaks_ixs = peaks_ixs
    sightline.prediction.offset_hist = offset_hist
    sightline.prediction.offset_conv_sum = offset_conv_sum
    #if peaks_ixs==[]:
        #print(sightline.id,np.amax(smooth_conv_sum))
    return sightline

def analyze_pred(sightline,pred,conf, offset, coldensity,PEAK_THRESH):
    """
    Gnerate DLA catalog for each sightline
    
    Parameters:
    -----------------------------------------------
    sightline: 'dla_cnn.data_model.Sightline' object
    pred:list, label for every window, 0 means no DLA in this window and 1 means this window has a DLA
    conf:list, [0,1]confidence level, label for every window, pred is 0 when conf is below the critical value (0.5 default), pred is 1 when conf is above the critical value
    offset:list, [-60,+60] , label for every window, pixel numbers between DLA center and the window center
    coldensity:list, label for every window, the estimated NHI column density
    PEAK_THRESH: float, the threshold to accept a peak=0.2
    
    Returns
    -----------------------------------------------
    dla_tbl: 'astropy.table.Table' object
    
    """
    #delete the offset value when pred=0
    for i in range(0,len(pred)):
        if (pred[i]==0):#or(real_classifier[i]==-1):
            offset[i]=0
    sightline.prediction = Prediction(loc_pred=pred, loc_conf=conf, offsets=offset, density_data=coldensity)
    compute_peaks(sightline,PEAK_THRESH)
    sightline.prediction.smoothed_loc_conf()
    data_split=split_sightline_into_samples(sightline)
    lam_analyse=data_split[5]
    
    #generate absorbers catalog for every sightline
    dla_tbl = Table(names=('TARGET_RA','TARGET_DEC', 'ZQSO','Z','TARGETID','S/N','DLAID','NHI','DLA_CONFIDENCE','NHI_STD','ABSORBER_TYPE'),dtype=('float','float','float','float','int','float','str','float','float','float','str'),meta={'EXTNAME': 'DLACAT'})
    for jj in range(0,len(sightline.prediction.peaks_ixs)):
        peak=sightline.prediction.peaks_ixs[jj]
        peak_lam_spectrum = lam_analyse[peak]
        z_dla = float(peak_lam_spectrum) / 1215.67 - 1
        peak_lam_rest=float(peak_lam_spectrum)/(1+sightline.z_qso)
        _, mean_col_density_prediction, std_col_density_prediction, bias_correction =             sightline.prediction.get_coldensity_for_peak(peak)

        absorber_type =  "DLA" if mean_col_density_prediction >= 20.3 else "LYB" if sightline.is_lyb(peak) else "SUBDLA"
        dla_tbl.add_row((sightline.ra,sightline.dec,sightline.z_qso,float(z_dla),sightline.id,sightline.s2n,
                         str(sightline.id)+'00'+str(jj), float(mean_col_density_prediction),
                         min(1.0,float(sightline.prediction.offset_conv_sum[peak])),float(std_col_density_prediction),absorber_type))
        
    return dla_tbl




