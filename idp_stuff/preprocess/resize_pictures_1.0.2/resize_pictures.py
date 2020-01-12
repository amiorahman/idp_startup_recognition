import cv2
import numpy
import glob
from PIL import Image
from resizeimage import resizeimage

#Define pictures directory
folders = glob.glob('/home/amio/Downloads/startup/startup/')
imagenames_list = []

#Get all images to a list
for folder in folders:
    for f in glob.glob(folder + '/*.png'):
        imagenames_list.append(f)

#Resize images and save as new images with new dimensions   
imgCounter = 0
 
for imgCounter in range(len(imagenames_list)):
	with open(imagenames_list[imgCounter], 'r+b') as f:
	    with Image.open(f) as image:
	        cover = resizeimage.resize_cover(image, [384, 384])
	        cover.save('/home/amio/idp_sr/yesStartUp/' + str(imgCounter) + '.png', image.format)
	        imgCounter += 1
	        #picExt += 1