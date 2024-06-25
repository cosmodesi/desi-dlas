from desidlas.prediction.get_prediction import save_pred
import numpy as np

release='fuji'
pix={'special':['dark'],'sv1':['backup','bright','dark','other'],
'sv2':['backup','bright','dark'],
'sv3':['backup','bright','dark'],'cmx':['other']}
for survey in list(pix.keys()):
    programs=pix[survey]
    for program in programs:
        print(survey,program)
        partpre_loc='/home/zjqi/data/desi/partpre/%s/%s/%s/partpre.npy'%(release,survey,program)
        sightlinepath=os.listdir('/home/zjqi/data/desi/sightlines/%s/%s/%s'%(release,survey,program))
        if np.isin('cutpre-sightlines.npy',sightlinepath):
            sightline_loc='/home/zjqi/data/desi/sightlines/%s/%s/%s/cutpre-sightlines.npy'%(release,survey,program)
        else:
            sightline_loc='/home/zjqi/data/desi/sightlines/%s/%s/%s/pre-sightlines.npy'%(release,survey,program)
        partpre=np.load(partpre_loc,allow_pickle=True).item()
        sightlines=np.load(sightline_loc,allow_pickle=True).item()
        save_pred(sightlines,pred,filename='/home/zjqi/data/desi/dlacatalog/%s/%s/%s/CNN-dlacatalog.fits'%(release,survey,program))
        
        