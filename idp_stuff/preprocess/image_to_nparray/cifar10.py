import cv2
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm

#Define pictures directory
folders = glob.glob('/home/amio/idp_sr/')
imagenames_list = []

#Get all images to a list
for folder in folders:
    for f in glob.glob(folder + '/*.png'):
        imagenames_list.append(f)

out = []
imgCounter = 0
labelCounter = 0

#for imgCounter in range(len(imagenames_list)):        
	#im = Image.open(imagenames_list[imgCounter])
im = Image.open('/home/amio/idp_sr/test-image-cover-1.png')
im = (np.array(im))

r = im[:,:,0].flatten()
g = im[:,:,1].flatten()
b = im[:,:,2].flatten()
#label = [labelCounter]
label = [1]
#out1 = []
out = np.append(out, (np.array(list(label) + list(r) + list(g) + list(b),np.uint8)), axis = 0)
#out = np.array(out.tolist())
##np.append (out, out1)

#print(out)
out.tofile("out.bin")