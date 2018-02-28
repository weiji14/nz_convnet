import os
import sys
import time
import pyscreenshot as ImageGrab  #cross-platform
#from PIL import ImageGrab        #for windows
import numpy as np
import cv2

from model import keras_model
#%%

try:
    size = int(sys.argv[1])
    px, py = (size,size)
    
except IndexError:
    px, py = (512,512)

try:
    threshold = (int(sys.argv[2])/100)
except IndexError:
    threshold = 0.5

x, y = ((1680-px)/2, (1050-py)/2)    #set to your screen resolution
model_loaded = keras_model(img_width=px, img_height=py)
model_loaded.load_weights(sys.path[0]+"/data/model/model-weights.hdf5")

#%%

if __name__ == "__main__":

    while True:
        #time.sleep(0.01)
        ary = np.array(ImageGrab.grab(bbox=(x,y,x+px,y+py)))
        try:
            assert(ary.shape == (px,py,3))
            #cv2.imshow('Input screenshot', ary[:,:,::-1])  #convert RGB to BGR
            
            ary_hat = model_loaded.predict(np.stack([ary]))[0][:,:,0]
            ary_hat = np.array(ary_hat*255, dtype=np.uint8)  #rescale from float32 to uint8
            ret,thresh = cv2.threshold(ary_hat, int(255*threshold), 255, 0) #set threshold as over 0.5 (255/2=128)
            img, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cv2.imshow('Live building detector', cv2.drawContours(ary, contours, -1, (0,255,0), 2)[:,:,::-1])
    
            if cv2.waitKey(25) & 0xFF == ord('q'):  #press q to quit
                cv2.destroyAllWindows()
                break
        except AssertionError:
            pass
    print('end')