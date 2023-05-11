from __future__ import print_function
import os
import cv2 as cv
import numpy as np
import argparse
import random as rng
import math
from matplotlib import pyplot as plt
import statistics
from scipy import optimize
import circle_fit


catalog_name = [
    "01_S2_7",
    "02_S1_3",
    "03_S1_5",
    "04_S1_6",
    "05_S4_2",
    "06_S4_5",
    "07_S4_6",
    "08_S5_2",
    "09_S5_4",
    "10_S5_5",
    "11_S3_7",
]


for i in range(1, len(catalog_name)):
    # Load source image
    img_path = (
        "E:\\Google_drive\\Studia\\!_Doktorat\\NAWA\\2021_11\\SFT\\2021_11_26_SFT\\Photos\\Canon\\Samples\\Markers\\"
        + catalog_name[i][:]
        + "\\"
    )
    img_list = os.listdir(img_path)
    result_path = img_path
    # cropp_path='E:\\Google_drive\\Studia\\!_Doktorat\\NAWA\\2021_11\\SFT\\2021_11_26_SFT\\Photos\\Canon\\cropped\\'

    # Result_file=open(result_path+'results.txt','w+')

    for img_file in img_list:
        if img_file.endswith(".jpg" or ".JPG"):
            src = cv.imread(img_path + img_file)
            if src is None:
                print("Could not open or find the image:", img_file)
                exit(0)
            print("Processing of: " + img_file)
            circlesInPhoto = circle_fit.find_circles(src)
            Result_file = open(
                result_path + os.path.splitext(img_file)[0] + ".txt", "w+"
            )
            Result_file.write(img_file + "\n")
            np.savetxt(Result_file, circlesInPhoto, fmt="%.4f", delimiter=",")
            Result_file.close()
    # Result_file.close()
