import os
import cv2
import numpy as np
import sys
import pandas as pd
from skimage import io, img_as_float
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
import multiprocessing as mp
import ctypes
from functools import partial
import multiprocessing

#globals values
labelPath = "/nfs/project/CVPR_WAD2018/ch4_instance_segmentation/train/images"
imgPath = "F:\\course\\deeplearning\\kaggledata\\train_img\\"

def ReadDir(labelPath,imgPath):
    images = []
    for root, dirs, files in os.walk(imgPath):
        for img_name in files:
            if not img_name.lower ().endswith (('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            images.append(img_name)
    return images

def deal(img_name):
    label_name = img_name[0:img_name.find (".")] + "_instanceIds" + ".png"
    labelname = labelPath + label_name
    img = Image.open (imgPath + img_name)
    tlabel = np.asarray (Image.open (labelname)) // 1000
    tlabel[tlabel != 0] = 255
    res = Image.blend (img, Image.fromarray (tlabel).convert ('RGB'), alpha=0.4)
    result = cv2.cvtColor (np.asarray (res), cv2.COLOR_RGB2BGR)
    while (result.shape[0] > 600):
        result = cv2.pyrDown (result)
    cv2.imshow("result",result)
    cv2.waitKey(3)
    cv2.imwrite ("G:\\LabedKaggleImage\\" + img_name, result)

if __name__ == '__main__':
    images = ReadDir (labelPath, imgPath)
    #cores = multiprocessing.cpu_count ()
    cores = 2
    pool = multiprocessing.Pool (processes=cores)
    pool.map (deal, images)
