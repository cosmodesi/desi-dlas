
def get_sightlines(spectra,truth,zbest,outpath):

    """
    Insert DLAs manually into sightlines without DLAs or only choose sightlines with DLAs
    
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
                rebin(sightline, best_v['b'])
                sightlines.append(sightline)
                
    np.save(outpath,sightlines)
    return sightlines
