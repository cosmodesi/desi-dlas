#these modules are used to generated the whole DLA catalog for sightlines, and compared the predicted DLA catalog with real DLA catalog if it exits.
import numpy as np 
import matplotlib.pyplot as plt
from desidlas.prediction.analyze_prediction import analyze_pred
from astropy.table import Table, vstack,hstack

def save_pred(sightlines,pred,PEAK_THRESH,level,filename=None):
    """
    Using prediction for windows to get prediction for sightlines, get a pred DLA catalog.
    
    Parameters
    ---------------
    sightlines: data_model.Sightline object list
    pred: dict
    PEAK_THRESH: float
    level: float, confidence value, classifier is 0 when conf is below the critical value (0.5 default), else be 1.
    filename: str, use it to save DLA catalog
    
    Returns
    ---------------
    pred_abs: astropy.table.Table 
    """
    pred_abs = Table()
    for ii in range(0,len(sightlines)):
        sightline=sightlines[ii]
        conf=pred[sightline.id]['conf']
        lam_analyse = pred[sightline.id]['lam'][0]
        #if we want to change confidence level from CNN output
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
        pred_abs=vstack((pred_abs,analyze_pred(sightline,classifier,conf,offset,coldensity,PEAK_THRESH, lam_analyse)))
    pred_abs.write(filename,overwrite=True)
    return pred_abs

def label_catalog(real_catalog,pred_catalog,realname=None,predname=None):
    """
    Compare real absorbers and predicted absorbers to add TP, FN, FP informations to DLA catalogs, calculate numbers of TP,FN,FP.
    
    Parameters
    ---------------
    real_catalog:astropy.table.Table class 
    pred_catalog:astropy.table.Table class
    realname: str, use it to save the labeled real DLA catalog
    predname: str, use it to save the labeled pred DLA catalog
    
    Reutrns
    ----------------
    tp_preds: list
    fn_num: int
    fp_num: int
   
    """
    tp_pred=[]
    fn_num=0
    fp_num=0
    pred_catalog.add_column('str',name='label')
    pred_catalog.add_index('DLAID')
    real_catalog.add_column('str',name='label')
    
    for real_dla in real_catalog:
        pred_dlas=pred_catalog[pred_catalog['TARGETID']==real_dla['TARGETID']]
        central_wave=1215.67*(1+real_dla['Z'])
        pred_wave=1215.67*(1+pred_dlas['Z'])
        col_density=real_dla['NHI']
        pred_coldensity=pred_dlas['NHI']
        targetid= real_dla['TARGETID']
        s2n=real_dla['S/N']
        #Compare the predicted DLA location with the real DLA location
        lam_difference=np.abs(pred_wave-central_wave)
        if len(lam_difference) != 0:
            nearest_ix = np.argmin(lam_difference) 
            #if the distance between the two DLA is less than 10 Å and not lyb absorption, this real DLA is taken as a true positive
            if (lam_difference[nearest_ix]<=10)&(pred_dlas[nearest_ix]['ABSORBER_TYPE']!='LYB')&(pred_dlas[nearest_ix]['label']=='str'):
                real_dla['label']='tp'
                dlaid=pred_dlas[nearest_ix]['DLAID']
                pred_catalog.loc[dlaid]['label']='tp'
                tp_pred.append([central_wave,col_density,pred_wave[nearest_ix],pred_coldensity[nearest_ix],targetid, s2n])
            #if there is no prediction corresponding to this true DLA, then this is taken to be a false negative
            else:
                real_dla['label']='fn'
                fn_num=fn_num+1
        #same as the previous lines
        else:
            real_dla['label']='fn'
            fn_num=fn_num+1  
    for pred_dla in pred_catalog:
        if pred_dla['ABSORBER_TYPE']=='LYB':
            pred_dla['label']='LYB'
        else:
            #the prediction is neither lyb nor tp are false positive
            if pred_dla['label']=='str':
                pred_dla['label']='fp'
                fp_num=fp_num+1
    
    real_catalog.write(realname,overwrite=True)
    pred_catalog.write(predname,overwrite=True)
    return tp_pred, fn_num, fp_num
    
      
def get_results(real_catalog,pred_catalog,realname=None,predname=None,tpname=None):
    """
    Compare real absorbers and predicted absorbers to add TP, FN, FP informations to DLA catalogs, calculate numbers of TP,FN,FP and draw histogram.
    
    Parameters
    ---------------
    real_catalog:astropy.table.Table class 
    pred_catalog:astropy.table.Table class
    realname: str, use it to save the new real DLA catalog
    predname: str, use it to save the new pred DLA catalog
    tpname: str, use it to save tp predictions.
    
    """
    tp_pred, fn_num, fp_num=label_catalog(real_catalog,pred_catalog,realname=realname,predname=predname)
    print('true_positive=%s,false_negative=%s,false_positive=%s'%(len(tp_pred),fn_num,fp_num))
    np.save(tpname,tp_pred)
    
    #draw histogram
    delta_z=[]
    delta_NHI=[]
    real_nhi=[]
    for pred in tp_pred:
        pred_z=pred[2]/1215.67-1
        real_z=pred[0]/1215.67-1
        delta_z.append(pred_z-real_z)
        delta_NHI.append(pred[3]-pred[1])
        real_nhi.append(pred[1])
    #calculate the mean value and standard deviation of d_z and d_NHI    
    arr_mean = np.mean(delta_z)
    arr_var = np.var(delta_z)
    arr_std = np.std(delta_z,ddof=1)
    arr_mean_2 = np.mean(delta_NHI)
    arr_var_2 = np.var(delta_NHI)
    arr_std_2 = np.std(delta_NHI,ddof=1)
    
    plt.figure(figsize=(10,10))
    plt.title('stddev=%.4f mean=%.5f'%(arr_std,arr_mean),fontdict=None,loc='center',pad='20',fontsize=20,color='red')
    plt.hist(delta_z,bins=50,density=False)#,edgecolor='black')
    plt.ylabel('N',fontsize=20)
    plt.xlabel('$\Delta$'+'z',fontsize=20)
    plt.tick_params(labelsize=18)
    #plt.savefig('/home/bwang/data/jfarr-0.2-4/delta_z_validation_new.pdf')

    plt.figure(figsize=(10,10))
    plt.title('stddev=%.4f mean=%.5f'%(arr_std_2,arr_mean_2),fontdict=None,loc='center',pad='20',fontsize=20,color='red')
    plt.hist(delta_NHI,bins=100,density=False)#,edgecolor='black')
    plt.ylabel('N',fontsize=20)
    plt.xlabel('$\Delta$'+'log${N_{\mathregular{HI}}}$',fontsize=20)
    plt.tick_params(labelsize=18)
    plt.xlim(-1.0,1.0)
    #plt.savefig('/home/bwang/data/jfarr-0.2-4/delta_NHI_validation_new.pdf')
    
