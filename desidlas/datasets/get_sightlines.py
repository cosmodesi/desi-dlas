
def get_sightlines(spectra,truth,zbest,outpath):

    """
    Read Fits file and preprocess the sightlines
    Use the DesiMock class to read fits file and perform pre-processing work such as rebinning, normalizing, and calculating the signal-to-noise ratio(S/N),
    then store the pre-processed spectra data into the Sighline class. 
    Based on the dataset we want to generate, these filters can be changed. Here lists the filters to spectra:
    spectra with z<2.33 are filtered
    spectra with S/N <1 are filtered
    only use the blue band in this sightline
    
    spectra:path of spectra file in DESI(fits format), this is used for the information of spectra like flux,wavelength
    truth:path of truth file in DESI(fits format), this is used for the information of DLAs
    zbest:path of zbest file in DESI(fits format),this is used for the redshift of QSOs
    outpath: path to save the output file
    
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
                rebin(sightline, best_v['all'])
                sightlines.append(sightline)
                
    np.save(outpath,sightlines)
    return sightlines
