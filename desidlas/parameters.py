""" Definitions used throughout the package"""

# Need to fill in TBD with the correct value
#REST_RANGE = [920, 1214, "TBD"]
REST_RANGE = [900, 1346, "TBD"]
kernel = {'highsnr':400,'lowsnr':600} # SDSS value -- UPDATE!!
best_v = {'b': 62996, 'r': 44859, 'z': 34720,'all': 44735}#the best value of rebining for each channel, its unit is m*s^(-1).
norm_range=[1420,1480]#for normalization and estimate s2n
#for NHI measurements bias adjustment
bias_adjust=0#(0.0028149011281380278276520456870457564946264028549194,-0.0646188010849933769375041947569116018712520599365234,-0.004256561717710568779060587019102968042716383934021,23.555317918478582583929892280139029026031494140625)
#how many pixel used to get the NHI median 2*normal_range
normal_range = 40#30
pos_sample_kernel_percent=0.3#60*2/400
smooth_model='highsnr'#'lowsnr'
camera='all'#'b','r','z'
continuum_model=False
PEAK_THRESH=0.2
level=0.5