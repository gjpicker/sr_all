import os 
import numpy as np 
import PIL.Image 

from utils import util as util

fn_np = lambda x :np.array(PIL.Image.open(x))

v1,v2 = [], [] 
for i in  range(5):
    hr =fn_np ("hr_%d.jpg"%(i) )
    sr =fn_np ("sr_%d.jpg"%(i) )
    visuals={"SR":sr ,"HR":hr}
    scale=4
    asnr, ssim = util.calc_metrics(visuals['SR'], visuals['HR'], crop_border=scale)

    v1.append(asnr)
    v2.append(ssim)

    print (asnr,ssim,"--")

print (np.mean(v1), np.mean(v2))
