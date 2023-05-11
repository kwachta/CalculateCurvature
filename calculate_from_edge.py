import os
import cv2 as cv
import numpy as np
import math
import circle_fit
import matplotlib.pyplot as plt

slider_max = 255
title_window = "Threshold_window"
val = slider_max / 2


def on_trackbar(val, srctemp):
    R_src = srctemp.copy()
    # BGR coding
    (B, G, R) = cv.split(R_src)
    copy_src = R.copy()
    ret, threshold_img = cv.threshold(copy_src, val, 255, cv.THRESH_BINARY)
    # threshold_img=cv.adaptiveThreshold(copy_src,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #                        cv.THRESH_BINARY_INV,val,3)
    kernel = np.ones((3, 3), np.uint8)

    # threshold_img=cv.erode(threshold_img,kernel,iterations=1)
    # threshold_img=cv.dilate(threshold_img,kernel,iterations=1)
    grad_x = cv.Sobel(threshold_img, -1, 1, 0)

    cv.imshow(title_window, threshold_img)


def sample_edges(src):
    """'function is based on the image with cut out markers and their surroundings (only "edges" of the sample are present)"""
    R_src = src.copy()
    # BGR coding
    (B, G, R) = cv.split(R_src)
    copy_src = R.copy()
    copy_src = cv.blur(copy_src, (3, 3))
    # copy_src=cv.equalizeHist(copy_src)

    ret, threshold_img = cv.threshold(copy_src, 96, 255, cv.THRESH_BINARY)
    # ret,threshold_img=cv.threshold(copy_src,103,255,cv.THRESH_BINARY)
    # block_size = 9
    # threshold_img=cv.adaptiveThreshold(copy_src,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #                        cv.THRESH_BINARY_INV,block_size,3)

    canny_output = cv.Canny(threshold_img, 20, 40)

    grad_x = cv.Sobel(threshold_img, -1, 1, 0)
    # grad_x=cv.Scharr(threshold_img,-1,1,0)
    # abs_grad_x = cv.convertScaleAbs(grad_x)
    # cv.imshow('t',grad_x)
    # cv.waitKey()
    # plt.imshow(grad_x)
    # plt.show()
    # Find contours
    contours, hierarchy = cv.findContours(grad_x, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

    # get edges
    contour_list = []
    circle_list = []
    for contour in contours:
        moments = cv.moments(contour)

        if contour[:, :, 1].max() - contour[:, :, 1].min() > 100:
            if moments["mu20"] != 0:
                param = moments["mu02"] / moments["mu20"]
                # print(param)
                area = cv.contourArea(contour)
                # if ((param>0.98) & (area > 100)):
                if 1:
                    contour_list.append(contour)
                    cv.drawContours(R_src, contour_list, -1, (255, 0, 0), 2)
                    x = []
                    y = []
                    for t in contour:
                        x.append(t[0][0])
                        y.append(t[0][1])
                    x = np.r_[x]
                    y = np.r_[y]
                    temp = circle_fit.fit_circle(x, y)
                    print(temp[0][2])
                Result_file = open(
                    result_path + os.path.splitext(img_file)[0] + ".txt", "w+"
                )
                Result_file.write(img_file + "\n")
                for c in contour_list:
                    np.savetxt(Result_file, np.squeeze(c), fmt="%.4f", delimiter=",")
                Result_file.close()

    # cv.imshow('view',R_src)
    # cv.imwrite('test_out.jpg',R_src)
    # cv.waitKey()


catalog_names = [
    "02_S1_3",
    "03_S1_5",
    "04_S1_6",
]


for i in range(0, len(catalog_names)):
    img_path = (
        "C:\\Users\\Karol\\source\\repos\\Calculate curvature\\Example_images"
        + catalog_names[i][:]
        + "\\"
    )
    img_list = os.listdir(img_path)
    result_path = img_path

    # Result_file=open(result_path+'results.txt','w+')

    for img_file in img_list:
        if img_file.endswith(".jpg" or ".JPG"):
            src = cv.imread(img_path + img_file)
            # src = cv.imread('IMG_7107_cropp.JPG')
            # src = cv.imread('lines.png')
            if src is None:
                print("Could not open or find the image:", img_file)
                exit(0)
            print("Processing of: " + img_file)
            # remove surroundings of circles from the photo
            srcWithoutCircles = src.copy()
            circlesInPhoto = circle_fit.find_circles(src)
            for circle_param in circlesInPhoto:
                x_c_0 = int(circle_param[0]) - int(circle_param[2]) - 25
                x_c_1 = int(circle_param[0]) + int(circle_param[2]) + 25
                y_c_0 = int(circle_param[1]) - int(circle_param[2]) - 25
                y_c_1 = int(circle_param[1]) + int(circle_param[2]) + 25
                if x_c_0 < 0:
                    x_c_0 = 0
                if y_c_0 < 0:
                    y_c_0 = 0
                srcWithoutCircles[y_c_0:y_c_1, x_c_0:x_c_1] = 0
            # cv.imshow('temp',copy_src)
            # cv.waitKey()

            # # trackbar to tune the threshold
            # cv.namedWindow(title_window)
            # trackbar_name = "Threshold %d" % slider_max
            # cv.createTrackbar(
            #     trackbar_name,
            #     title_window,
            #     0,
            #     slider_max,
            #     on_trackbar(val, srcWithoutCircles),
            # )
            # cv.waitKey()
            sample_edges(srcWithoutCircles)
