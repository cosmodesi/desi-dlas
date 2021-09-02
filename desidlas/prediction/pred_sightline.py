#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
from dla_cnn.data_model.Sightline import Sightline
from dla_cnn.data_model.Prediction import Prediction
from dla_cnn.spectra_utils import get_lam_data
import matplotlib.pyplot as plt
from linetools.spectra.xspectrum1d import XSpectrum1D
from matplotlib.pyplot import MultipleLocator
from pyigm.abssys.dla import DLASystem
from pyigm.abssys.lls import LLSSystem
from pyigm.abssys.utils import hi_model
import scipy.signal as signal
from dla_cnn.desi.analyze_prediction import analyze_pred
from dla_cnn.desi.training_sets import split_sightline_into_samples
from astropy.table import Table, vstack,hstack
#draw sightline dla&pred dla
def get_dla(zabs,NHI,matrix_lam,matrix_flux,wvoff=60.):
    spec = XSpectrum1D.from_tuple((matrix_lam,matrix_flux))
    if NHI<20.3:
        dla = LLSSystem((0,0), zabs, None, NHI=NHI)      
    else:
        dla = DLASystem((0,0), zabs, None, NHI)
    wvcen = (1+zabs)*1215.67
    gd_wv = (spec.wavelength.value > wvcen-wvoff-30) & (spec.wavelength.value < wvcen+wvoff+30)
    co = 1.5#np.mean(spec.flux[gd_wv])#amax
    lya, lines = hi_model(dla, spec, lya_only=True)
    return lya.wavelength[gd_wv],co*lya.flux[gd_wv]
def draw_sightline(sightline,pred,pred_abs): 
    data_split=split_sightline_into_samples(sightline)
    lam_analyse=data_split[5]
    flux_analyse=data_split[6]
    ab=np.max(flux_analyse)
    matrix_lam=np.array(lam_analyse)
    matrix_flux=np.array(flux_analyse)
    #classifier=pred[sightline.id]['pred']
    #conf=pred[sightline.id]['conf']
    lya=[]
    lya_preds=[]
    wvcen=[]
    central_wave=[]
    col_density=[]
    col_d=[]
    for dla in sightline.dlas:
        zabs=(dla.central_wavelength)/1215.67-1
        NHI=dla.col_density   
        lya.append(get_dla(zabs,NHI,matrix_lam,matrix_flux,wvoff=60.))
            #(lyawavelength_1,lyaflux_1)=get_dla(zabs_1,NHI_1,matrix_lam,matrix_flux,wvoff=60.)
        wvcen.append(dla.central_wavelength)
        col_density.append(NHI)
    for pred_ab in pred_abs:
        z=pred_ab['spectrum']/1215.67-1
        nhi=pred_ab['column_density']
        lya_preds.append(get_dla(z,nhi,matrix_lam,matrix_flux,wvoff=60.))
        central_wave.append(pred_ab['spectrum'])
        col_d.append(nhi)

    plt.rcParams['figure.figsize'] = (12.0, 6.0)
    
    plt.plot(lam_analyse,flux_analyse,color='black')
    #plt.legend(bbox_to_anchor=(0.88,1.02,10,20), loc=3,ncol=1, mode=None, borderaxespad=0,fontsize=18)
    
    for lyaabs in lya:
        plt.plot(lyaabs[0],lyaabs[1],color='blue',label='real dla')
    for lya_pred in lya_preds:
        plt.plot(lya_pred[0],lya_pred[1],color='red',label='pred dla')  
    plt.legend(bbox_to_anchor=(0.88,1.02,10,20), loc=3,ncol=1, mode=None, borderaxespad=0,fontsize=18)
    plt.axvline(x=(sightline.z_qso+1)*1215.67,ls="-",c='yellow',linewidth=3)
    #plt.text((sightline.z_qso+1)*1215.67+10,ab,'lya_emission',fontsize=12,color='blue')
    plt.xlim([900*(1+sightline.z_qso),1250*(1+sightline.z_qso)])
    for ii in range(0,len(wvcen)):   
        plt.axvline(x=wvcen[ii],ls="-",c="blue",linewidth=2)
        plt.text(wvcen[ii]+5,ab-1,'GT:'+'%.2f'%(wvcen[ii]),fontsize=18,color='blue')
        plt.text(wvcen[ii]+5,ab,'log${N_{\mathregular{HI}}}$'+'=%.2f'%(col_density[ii]),fontsize=18,color='blue')
    for jj in range(0,len(central_wave)):   
        plt.axvline(x=central_wave[jj],ls="-",c="red",linewidth=2)
        plt.text(central_wave[jj]+10,ab-3,'GT:'+'%.2f'%(central_wave[jj]),fontsize=18,color='red')
        plt.text(central_wave[jj]+10,ab-2,'log${N_{\mathregular{HI}}}$'+'=%.2f'%(col_d[jj]),fontsize=18,color='red')
    plt.ylabel('Relative Flux',fontsize=20)
    plt.xlabel('Wavelength'+'['+'$\AA$'+']',fontsize=20)
    plt.title('spec-%s snr-%s'%(sightline.id,sightline.s2n),fontdict=None,loc='center',pad='20',fontsize=30,color='blue')

   
    #plt.savefig('/Users/zjq/sightlines/717/low/%s.png'%(sightline.id))
    plt.show()


def save_pred(sightlines,pred,PEAK_THRESH,level,filename=None):
    pred_abs = Table(names=('TARGET_RA','TARGET_DEC', 'ZQSO','Z','TARGETID','S/N','DLAID','NHI','DLA_CONFIDENCE','NHI_STD','ABSORBER_TYPE'),dtype=('float','float','float','float','int','float','str','float','float','float','str'),meta={'EXTNAME': 'DLACAT'})
    for ii in range(0,len(sightlines)):
        sightline=sightlines[ii]
        conf=pred[sightline.id]['conf']
        #classifier=[]
        #for ii in range(0,len(conf)):
            #if conf[ii]>level:
                #classifier.append(1)
            #else:
                #classifier.append(0)
        #classifier=np.array(classifier)
        #real_classifier=real_claasifiers[ii]
        classifier=pred[sightline.id]['pred']
        offset=pred[sightline.id]['offset']
        coldensity=pred[sightline.id]['coldensity']
        pred_abs=vstack((pred_abs,analyze_pred(sightline,classifier,conf,offset,coldensity,PEAK_THRESH)))
    pred_abs.write(filename,overwrite=True)
    return pred_abs

def label_catalog(real_catalog,pred_catalog,realname=None,predname=None):
    tp_pred=[]
    fn_num=0
    fp_num=0
    #if pred_catalog['label'][0]=='str':
       #pred_catalog.remove_column('label')
    pred_catalog.add_column('str',name='label')
    pred_catalog.add_index('DLAID')
    #if real_catalog['label'][0]=='str':
       #real_catalog.remove_column('label')
    real_catalog.add_column('str',name='label')
    for real_dla in real_catalog:
        pred_dlas=pred_catalog[pred_catalog['TARGETID']==real_dla['TARGETID']]
        central_wave=1215.67*(1+real_dla['Z'])
        pred_wave=1215.67*(1+pred_dlas['Z'])
        col_density=real_dla['NHI']
        pred_coldensity=pred_dlas['NHI']
        targetid= real_dla['TARGETID']
        s2n=real_dla['S/N']
        lam_difference=np.abs(pred_wave-central_wave)
        if len(lam_difference) != 0:
            nearest_ix = np.argmin(lam_difference) 
            if (lam_difference[nearest_ix]<=10)&(pred_dlas[nearest_ix]['ABSORBER_TYPE']!='LYB')&(pred_dlas[nearest_ix]['label']=='str'):#距离小于10且不是lyb
                real_dla['label']='tp'
                dlaid=pred_dlas[nearest_ix]['DLAID']
                pred_catalog.loc[dlaid]['label']='tp'
                tp_pred.append([central_wave,col_density,pred_wave[nearest_ix],pred_coldensity[nearest_ix],targetid, s2n])
            else:
                real_dla['label']='fn'
                fn_num=fn_num+1
        else:
            real_dla['label']='fn'
            fn_num=fn_num+1  
    for pred_dla in pred_catalog:
        if pred_dla['ABSORBER_TYPE']=='LYB':
            pred_dla['label']='LYB'
        else:
            if pred_dla['label']=='str':
                pred_dla['label']='fp'
                fp_num=fp_num+1
    
    real_catalog.write(realname,overwrite=True)
    pred_catalog.write(predname,overwrite=True)
    return tp_pred, fn_num, fp_num
    
      
def get_results(real_catalog,pred_catalog,realname=None,predname=None,tpname=None):
    tp_pred, fn_num, fp_num=label_catalog(real_catalog,pred_catalog,realname=realname,predname=predname)
    print('true_positive=%s,false_negative=%s,false_positive=%s'%(len(tp_pred),fn_num,fp_num))
    np.save(tpname,tp_pred)
    '''
    #draw hist
    delta_z=[]
    delta_NHI=[]
    real_nhi=[]
    for pred in tp_pred:
        pred_z=pred[2]/1215.67-1
        real_z=pred[0]/1215.67-1
        delta_z.append(pred_z-real_z)
        delta_NHI.append(pred[3]-pred[1])
        real_nhi.append(pred[1])
    arr_mean = np.mean(delta_z)
    arr_var = np.var(delta_z)
    arr_std = np.std(delta_z,ddof=1)
    plt.figure(figsize=(10,10))
    plt.scatter(real_nhi,delta_NHI)
    plt.savefig('/home/bwang/data/jfarr-0.2-4/delta_NHI.pdf')
    
    arr_mean_2 = np.mean(delta_NHI)
    arr_var_2 = np.var(delta_NHI)
    arr_std_2 = np.std(delta_NHI,ddof=1)
    plt.figure(figsize=(10,10))
    plt.title('stddev=%.4f mean=%.5f'%(arr_std,arr_mean),fontdict=None,loc='center',pad='20',fontsize=20,color='red')
    plt.hist(delta_z,bins=50,density=False)#,edgecolor='black')
    plt.ylabel('N',fontsize=20)
    plt.xlabel('$\Delta$'+'z',fontsize=20)
    plt.tick_params(labelsize=18)
    plt.savefig('/home/bwang/data/jfarr-0.2-4/delta_z_validation_new.pdf')

    plt.figure(figsize=(10,10))
    plt.title('stddev=%.4f mean=%.5f'%(arr_std_2,arr_mean_2),fontdict=None,loc='center',pad='20',fontsize=20,color='red')
    plt.hist(delta_NHI,bins=100,density=False)#,edgecolor='black')
    plt.ylabel('N',fontsize=20)
    plt.xlabel('$\Delta$'+'log${N_{\mathregular{HI}}}$',fontsize=20)
    plt.tick_params(labelsize=18)
    plt.xlim(-1.0,1.0)
    plt.savefig('/home/bwang/data/jfarr-0.2-4/delta_NHI_validation_new.pdf')
    '''
def draw_analyze(sightline,pred_abs,pred):   
    lam, lam_rest, ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso)
    kernelrangepx = 200
    cut=((np.nonzero(ix_dla_range)[0])>=kernelrangepx)&((np.nonzero(ix_dla_range)[0])<=(len(lam)-kernelrangepx-1))   
    lam_analyse=lam[ix_dla_range][cut]
    flux_analyse=sightline.flux[ix_dla_range][cut]
    ab=np.max(flux_analyse)
    matrix_lam=np.array(lam_analyse)
    matrix_flux=np.array(flux_analyse)
    classifier=pred[sightline.id]['pred']
    conf=pred[sightline.id]['conf']
    lya=[]
    lya_preds=[]
    wvcen=[]
    central_wave=[]
    col_density=[]
    col_d=[]
    for dla in sightline.dlas:
        zabs=(dla.central_wavelength)/1215.67-1
        NHI=dla.col_density   
        lya.append(get_dla(zabs,NHI,matrix_lam,matrix_flux,wvoff=60.))
            #(lyawavelength_1,lyaflux_1)=get_dla(zabs_1,NHI_1,matrix_lam,matrix_flux,wvoff=60.)
        wvcen.append(dla.central_wavelength)
        col_density.append(NHI)
    for pred_ab in pred_abs:
        z=pred_ab['spectrum']/1215.67-1
        nhi=pred_ab['column_density']
        lya_preds.append(get_dla(z,nhi,matrix_lam,matrix_flux,wvoff=60.))
        central_wave.append(pred_ab['spectrum'])
        col_d.append(nhi)

    plt.rcParams['figure.figsize'] = (12.0, 12.0)
    plt.subplot(211)
    plt.plot(lam_analyse,flux_analyse,color='black')
    #plt.legend(bbox_to_anchor=(0.88,1.02,10,20), loc=3,ncol=1, mode=None, borderaxespad=0,fontsize=18)
    
    for lyaabs in lya:
        plt.plot(lyaabs[0],lyaabs[1],color='blue',label='real dla')
    for lya_pred in lya_preds:
        plt.plot(lya_pred[0],lya_pred[1],color='red',label='pred dla')  
    plt.legend(bbox_to_anchor=(0.88,1.02,10,20), loc=3,ncol=1, mode=None, borderaxespad=0,fontsize=18)
    plt.xlim([3700,1250*(1+sightline.z_qso)])
    plt.axvline(x=(sightline.z_qso+1)*1215.67,ls="-",c='yellow',linewidth=3)
    plt.text((sightline.z_qso+1)*1215.67+10,ab,'lya_emission',fontsize=12,color='blue')
    for ii in range(0,len(wvcen)):   
        plt.axvline(x=wvcen[ii],ls="-",c="blue",linewidth=2)
        #plt.text(wvcen[ii]+10,ab-1,'GT:'+'%.2f'%(wvcen[ii]),fontsize=12,color='blue')
        plt.text(wvcen[ii]+10,ab,'log${N_{\mathregular{HI}}}$'+'=%.2f'%(col_density[ii]),fontsize=18,color='blue')
    for jj in range(0,len(central_wave)):   
        plt.axvline(x=central_wave[jj],ls="-",c="red",linewidth=2)
        #plt.text(central_wave[jj]-60,ab-1.5,'GT:'+'%.2f'%(central_wave[jj]),fontsize=12,color='red')
        plt.text(central_wave[jj]+10,ab-1,'log${N_{\mathregular{HI}}}$'+'=%.2f'%(col_d[jj]),fontsize=18,color='red')
    plt.ylabel('Relative Flux',fontsize=20)
    plt.xticks([])
    plt.title('spec-%s'%(sightline.id),fontdict=None,loc='center',pad='20',fontsize=30,color='blue')

    plt.subplot(212)
    plt.plot(lam_analyse,conf,color='black',label='conf')
    plt.plot(lam_analyse,classifier,color='green',label='classifier')
    plt.legend(bbox_to_anchor=(0.88,1.02,8,20), loc=3,ncol=1, mode=None, borderaxespad=0,fontsize=18)
    plt.xlim([3700,1250*(1+sightline.z_qso)])
    plt.axhline(y=0.5,ls="--",c="green",linewidth=1)
    for ii in range(0,len(wvcen)):  
        plt.axvline(x=wvcen[ii],ls="-",c="blue",linewidth=2)
        plt.text(wvcen[ii]+10,1,'GT:'+'%.2f'%(wvcen[ii]),fontsize=18,color='blue')
    for jj in range(0,len(central_wave)):   
        plt.axvline(x=central_wave[jj],ls="-",c="red",linewidth=2)
        plt.text(central_wave[jj]-100,1.1,'GT:'+'%.2f'%(central_wave[jj]),fontsize=18,color='red')
        
    plt.ylabel('Confidence&label',fontsize=20)
    plt.xlabel('Wavelength'+'['+'$\AA$'+']',fontsize=20)
    
    #plt.savefig('/Users/zjq/sightlines/717/tp/%s.png'%(sightline.id))
    plt.show()

