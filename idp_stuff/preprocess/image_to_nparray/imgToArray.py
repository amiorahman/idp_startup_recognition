import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
import keras
from keras.preprocessing import image

#Define pictures directory
folders = glob.glob('/home/amio/idp_sr/images/')
imagenames_list = []

#Get all images to a list
for folder in folders:
    for f in glob.glob(folder + '/*.png'):
        imagenames_list.append(f)

#Define output array list and image counter for every image
out = []
imgCounter = 0

#Get array list for all resized images
for i in tqdm(range(len(imagenames_list))):
    img = image.load_img(imagenames_list[imgCounter], target_size=(32,32,1), grayscale=False)
    img = image.img_to_array(img)
    img = img/255
    out.append(img)

X = np.array(out)
print(X)
#X.tofile("/home/amio/idp_sr/imageArrayOutput/out.bin")