from scipy.misc import imread,imresize,imsave
import numpy as np
import os 
img_dir = "eggnog798tocaton5502/"
for j in range(5):
    final_img = np.zeros((224*2+10,224*6+5*10,3))+255
    #final_img[:256,:512]=imread("../datasets/GTA/Label256Full/%08d.png"%105148)
    #imgl = os.listdir(img_dir)
    for i in range(12):
        #print i
        idx = i*2000
        if i==11: idx=30000
        im=imread(img_dir+"samp_%d_%d.png"%(idx,j))
        final_img[i/6*234:i/6*234+224,i%6*234:i%6*234+224]=im
    imsave(img_dir+"eggnog_%d.png"%j,final_img)
