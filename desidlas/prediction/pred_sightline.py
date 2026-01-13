#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import numpy as np
import matplotlib.pyplot as plt
from desidlas.prediction.analyze_prediction import analyze_pred
from astropy.table import Table, vstack
from desidlas.parameters import PEAK_THRESH,level
from tqdm import tqdm
import multiprocessing


from astropy.io import fits
from astropy.table import Table
from pathlib import Path

def save_pred(sightlines, pred, PEAK_THRESH=PEAK_THRESH, level=level, filename=None):
    pred = np.load(pred, allow_pickle=True)
    sightlines = np.load(sightlines, allow_pickle=True)

    rows = []
    for ii in range(len(sightlines)):
        sightline = sightlines[ii]
        
        # 跳过空sightline
        if sightline == [] or sightline is None:
            continue
        
        # 跳过预测失败的
        if pred[ii] is None:
            continue
        
        try:
            # 修复：安全获取lam
            lam_data = pred[ii]['lam']
            if isinstance(lam_data, np.ndarray):
                if lam_data.ndim == 2:
                    lam_analyse = lam_data[0]
                elif lam_data.ndim == 1:
                    lam_analyse = lam_data
                else:
                    # 如果是更高维度，取第一个
                    lam_analyse = lam_data.flatten()
            else:
                # 如果是标量或其他，直接使用
                lam_analyse = lam_data
            
            conf        = pred[ii]['conf']
            classifier  = pred[ii]['pred']
            offset      = pred[ii]['offset']
            coldensity  = pred[ii]['coldensity']

            tab = analyze_pred(sightline, classifier, conf, offset, coldensity, PEAK_THRESH, lam_analyse)
            
            for row in tab:
                rows.append({name: row[name] for name in tab.colnames})
                
        except Exception as e:
            print(f"  Warning: Error analyzing sightline {ii}: {e}")
            # 调试信息
            if pred[ii] is not None:
                print(f"    lam shape: {pred[ii]['lam'].shape if hasattr(pred[ii]['lam'], 'shape') else type(pred[ii]['lam'])}")
                print(f"    lam type: {type(pred[ii]['lam'])}")
            continue

    # 创建输出表
    if len(rows) == 0:
        out = Table(names=('TARGET_RA','TARGET_DEC','Z_QSO','Z_DLA','TARGETID',
                          'S2N','DLAID','NHI','DLA_CONFIDENCE','NHI_STD','ABSORBER_TYPE'),
                   dtype=('float','float','float','float','int','float','U50','float','float','float','U10'))
    else:
        out = Table(rows=rows, names=('TARGET_RA','TARGET_DEC','Z_QSO','Z_DLA','TARGETID',
                                      'S2N','DLAID','NHI','DLA_CONFIDENCE','NHI_STD','ABSORBER_TYPE'))

    if filename is None:
        raise ValueError("filename must be provided")
    
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    out.write(filename, overwrite=True)
    
    return out




def save_pred_all(sightline_loc, partpre_loc, dlacat_loc):
    assert len(sightline_loc) == len(partpre_loc) == len(dlacat_loc)
    for i in tqdm(range(len(sightline_loc)), total=len(sightline_loc)):
        try:
            Path(dlacat_loc[i]).parent.mkdir(parents=True, exist_ok=True)
            save_pred(sightline_loc[i], partpre_loc[i], 
                     PEAK_THRESH=PEAK_THRESH, level=level, filename=dlacat_loc[i])
        except Exception as e:
            print(f"\nError processing file {i}: {e}")
            import traceback
            traceback.print_exc()
            try:
                out = Table(names=('TARGET_RA','TARGET_DEC','Z_QSO','Z_DLA','TARGETID',
                                   'S2N','DLAID','NHI','DLA_CONFIDENCE','NHI_STD','ABSORBER_TYPE'),
                            dtype=('float','float','float','float','int','float','U50','float','float','float','U10'))
                Path(dlacat_loc[i]).parent.mkdir(parents=True, exist_ok=True)
                out.write(dlacat_loc[i], overwrite=True)
            except Exception as write_exc:
                print(f"  Warning: failed to write empty catalog: {write_exc}")
            continue

'''
def save_pred(sightlines,pred,PEAK_THRESH=PEAK_THRESH,level=level,filename=None):
    pred=np.load(pred,allow_pickle=True)
    sightlines=np.load(sightlines,allow_pickle=True)
    pred_abs = Table(names=('TARGET_RA','TARGET_DEC', 'Z_QSO','Z_DLA','TARGETID','S2N','DLAID','NHI','DLA_CONFIDENCE','NHI_STD','ABSORBER_TYPE'),dtype=('float','float','float','float','int','float','str','float','float','float','str'),meta={'EXTNAME': 'DLACAT'})
    for ii in tqdm(range(0,len(sightlines))):
        sightline=sightlines[ii]
        if sightline!=[]:
            lam_analyse = pred[ii]['lam'][0]
            conf=pred[ii]['conf']
            classifier=pred[ii]['pred']
            offset=pred[ii]['offset']
            coldensity=pred[ii]['coldensity']
            #lam_analyse = pred[sightline.id]['lam'][0]
            #conf=pred[sightline.id]['conf']
        #do not cut level
        # classifier=[]
        # for ii in range(0,len(conf)):
        # if conf[ii]>level:
        #     classifier.append(1)
        # else:
        #     classifier.append(0)
        # classifier=np.array(classifier)
            #classifier=pred[sightline.id]['pred']
            #offset=pred[sightline.id]['offset']
            #coldensity=pred[sightline.id]['coldensity']
            pred_abs=vstack((pred_abs,analyze_pred(sightline,classifier,conf,offset,coldensity,PEAK_THRESH,lam_analyse)))
    pred_abs.write(filename,overwrite=True)
    return pred_abs
'''
'''
def save_pred_all(sightline_loc,partpre_loc,dlacat_loc):
    cpu_count=256
    task_data = [(sightline_loc[i], partpre_loc[i], PEAK_THRESH, level, dlacat_loc[i]) for i in range(len(sightline_loc))]
    with multiprocessing.Pool(cpu_count) as pool:
        pool.starmap(save_pred, task_data)
'''
'''
def save_pred_all(sightline_loc, partpre_loc, dlacat_loc):
    # 顺序版（最稳）
    for i in tqdm(range(len(sightlines)), total=len(sightlines)):
        save_pred(sightline_loc[i], partpre_loc[i],
                  PEAK_THRESH=PEAK_THRESH, level=level, filename=dlacat_loc[i])
'''
'''
def save_pred_all(sightline_loc, partpre_loc, dlacat_loc):
    assert len(sightline_loc) == len(partpre_loc) == len(dlacat_loc), \
        "输入列表长度不一致"

    for s_path, p_path, out_path in tqdm(
        list(zip(sightline_loc, partpre_loc, dlacat_loc)),
        total=len(sightline_loc)
    ):
        # 确保输出目录存在
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # 将预测 npy → 解析为 DLACAT（保持你原有的 save_pred 接口）
        save_pred(s_path, p_path,
                  PEAK_THRESH=PEAK_THRESH,
                  level=level,
                  filename=out_path)
'''
def label_catalog(real_catalog,pred_catalog,realname=None,predname=None):
    
    #uesd for drawing histogram
    tp_pred=[]
    #calculate fn and fp number
    fn_num=0
    fp_num=0
    c=3e5
    v=800
    pred_catalog.add_column('str',name='label')
    pred_catalog.add_index('DLAID')
    real_catalog.add_column('str',name='label')
    for real_dla in real_catalog:
        pred_dlas=pred_catalog[pred_catalog['TARGETID']==real_dla['TARGETID']]
        central_wave=1215.67*(1+real_dla['Z_DLA'])
        pred_wave=1215.67*(1+pred_dlas['Z_DLA'])
        col_density=real_dla['NHI']
        s2n=real_dla['S2N']
        targetid=real_dla['TARGETID']
        pred_coldensity=pred_dlas['NHI']
        lam_difference=np.abs(pred_wave-central_wave)
        dlam=v*central_wave/c
        if len(lam_difference) != 0:
            nearest_ix = np.argmin(lam_difference) 
            if (lam_difference[nearest_ix]<=dlam)&(pred_dlas[nearest_ix]['ABSORBER_TYPE']!='LYB')&(pred_dlas[nearest_ix]['label']=='str'):#has not be identified as a true positive
                real_dla['label']='tp'
                dlaid=pred_dlas[nearest_ix]['DLAID']
                #label the prediction
                pred_catalog.loc[dlaid]['label']='tp'
                tp_pred.append([central_wave,col_density,pred_wave[nearest_ix],pred_coldensity[nearest_ix],targetid,s2n])
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
    
      
def get_results(real_catalog,pred_catalog,realname=None,predname=None,tpname=None,plot_path=None):
    tp_pred, fn_num, fp_num=label_catalog(real_catalog,pred_catalog,realname=realname,predname=predname)
    print('true_positive=%s,false_negative=%s,false_positive=%s'%(len(tp_pred),fn_num,fp_num))
    np.save(tpname, tp_pred)
    #draw hist
    delta_z=[]
    delta_NHI=[]
    for pred in tp_pred:
        pred_z=pred[2]/1215.67-1
        real_z=pred[0]/1215.67-1
        delta_z.append(pred_z-real_z)
        delta_NHI.append(pred[3]-pred[1])
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
    plt.savefig('%s/delta_z.pdf'%(plot_path))

    plt.figure(figsize=(10,10))
    plt.title('stddev=%.4f mean=%.5f'%(arr_std_2,arr_mean_2),fontdict=None,loc='center',pad='20',fontsize=20,color='red')
    plt.hist(delta_NHI,bins=100,density=False)#,edgecolor='black')
    plt.ylabel('N',fontsize=20)
    plt.xlabel('$\Delta$'+'log${N_{\mathregular{HI}}}$',fontsize=20)
    plt.tick_params(labelsize=18)
    plt.savefig('%s/delta_NHI.pdf'%(plot_path))
