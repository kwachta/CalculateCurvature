import os
import cv2 as cv
import numpy as np
import circle_fit

slider_max = 255
title_window = "Threshold_window"
write_to_file = True  # should the edge pixels be written to a file?

catalog_names = [
    "02_S1_3",
    "03_S1_5",
    "04_S1_6",
]


def on_trackbar(val):
    R_src = srcWithoutCircles.copy()
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


# window with trackbar to tune the threshold
cv.namedWindow(title_window)
trackbar_name = "Threshold %d" % slider_max

cv.createTrackbar(
    trackbar_name,
    title_window,
    127,
    slider_max,
    on_trackbar,
)

print("Filename\tCurvature [1/pix]")

for i in range(0, len(catalog_names)):
    img_path = "Example_images\\" + catalog_names[i][:] + "\\"
    img_list = os.listdir(img_path)

    for img_file in img_list:
        if img_file.endswith(".jpg" or ".JPG"):
            src = cv.imread(img_path + img_file)
            if src is None:
                print("Could not open or find the image:", img_file)
                exit(0)

            # remove markers and their surrounding from the photo
            srcWithoutCircles = src.copy()
            circlesInPhoto = circle_fit.find_markers(src)
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

            current_slider_value = cv.getTrackbarPos(trackbar_name, title_window)
            on_trackbar(current_slider_value)
            cv.waitKey()
            sample_edge_array = circle_fit.find_sample_edges(
                srcWithoutCircles, current_slider_value
            )

            # write edge pixels to a file
            if write_to_file:
                result_file = open(
                    img_path + os.path.splitext(img_file)[0] + ".txt", "w+"
                )
                result_file.write(img_file + "\n")
                result_file.write("x[pix],y[pix]\n")
                np.savetxt(result_file, sample_edge_array, fmt="%d", delimiter=",")
                result_file.close()

            # calculate curvature for joined contour
            fitted_circle = circle_fit.fit_circle(
                sample_edge_array[:, 0], sample_edge_array[:, 1]
            )
            print(img_file, "\t", str(1 / fitted_circle[2]))
