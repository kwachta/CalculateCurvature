import math
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import cv2 as cv


def plot_circle(x_c, y_c, r_c, x, y):
    circle1 = plt.Circle((x_c, y_c), r_c)
    fig, ax = plt.subplots()
    plt.xlim(x_c - 2 * r_c, x_c + 2 * r_c)
    plt.ylim(y_c - 2 * r_c, y_c + 2 * r_c)
    ax.set_aspect(1)
    ax.add_artist(circle1)
    plt.plot(x, y, color="red")

    plt.grid(True)
    plt.show()


def fit_circle(x, y):
    def circ_func(params, x, y):
        xc, yc, R = params[0], params[1], params[2]
        residual = R - np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
        return residual

    # starting parameters
    circ_param = np.array([np.mean(x), np.mean(y), (np.max(x) - np.min(x)) / 2])
    # optimization
    circ_param = optimize.leastsq(circ_func, x0=circ_param, args=(x, y))

    # print paramters
    # print(circ_param)

    # extract middle and radius
    x_m = circ_param[0][0]
    y_m = circ_param[0][1]
    r_m = circ_param[0][2]

    # plot the circle for debugging purpose
    # plot_circle(x_m,y_m,r_m,x,y)
    # return (x_m,y_m,r_m)
    return circ_param


def find_circles(src):
    # Convert image to gray and blur it
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(src_gray, (3, 3))

    ## Create Window and show begning picture
    # source_window = 'Source'
    # cv.namedWindow(source_window)
    # cv.imshow(source_window, src)

    # thresholding
    kernel = np.ones((3, 3), np.uint8)
    block_size = 9
    threshold_img = cv.adaptiveThreshold(
        src_gray,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        block_size,
        3,
    )

    threshold_img = cv.dilate(threshold_img, kernel, iterations=1)
    threshold_img = cv.erode(threshold_img, kernel, iterations=2)
    # cv.imshow('t',threshold_img)
    # cv.waitKey()
    # Detect edges using Canny
    canny_output = cv.Canny(threshold_img, 100, 200)

    copy_gray = src_gray.copy()
    copy_src = src.copy()

    # Find contours
    contours, hierarchy = cv.findContours(
        threshold_img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE
    )

    # Get circle contours
    contour_list = []
    circle_list = []
    for contour in contours:
        moments = cv.moments(contour)
        if moments["mu20"] + moments["mu02"] != 0:
            param = (
                1
                / (2 * math.pi)
                * (moments["m00"] * moments["m00"])
                / (moments["mu20"] + moments["mu02"])
            )
            area = cv.contourArea(contour)
            if (param > 0.98) & (area > 80):
                contour_list.append(contour)
                cv.drawContours(copy_src, contour_list, -1, (0, 255, 0), 2)
                x = []
                y = []
                for t in contour:
                    x.append(t[0][0])
                    y.append(t[0][1])
                x = np.r_[x]
                y = np.r_[y]
                circle_param = fit_circle(x, y)
                circle_list.append(circle_param[0][:])
                # show fitted circle - for debug purposes
                # circle_fit.plot_circle(circle_param[0][0],circle_param[0][1],circle_param[0][2],x,y)
    # print contours with circles
    circ_stack = np.vstack(circle_list)
    circ_stack_2 = np.squeeze(circ_stack)
    circ_stack_2 = circ_stack_2[np.argsort(circ_stack_2[:, 1])]
    # write to file
    # if not cv.imwrite('output_test.jpg', copy_src):
    #        raise Exception("Could not write image")

    # Show gray pictures with detected circles
    # cv.imshow('Contours', copy_src)
    # cv.waitKey()
    return circ_stack_2


def write_cropped_img(src_img, circles, cropp_file):
    x_min = int(np.min(circles[:, 0]))
    x_max = int(np.max(circles[:, 0]))
    y_min = int(np.min(circles[:, 1]))
    y_max = int(np.max(circles[:, 1]))
    # print(x_min, x_max, y_min, y_max)
    # cv.imshow('im',src_img[y_min-48:y_max+48,x_min-48:x_max+48])
    if not cv.imwrite(
        cropp_file, src_img[y_min - 48 : y_max + 48, x_min - 48 : x_max + 48]
    ):
        raise Exception("Could not write image")


# test
# x=np.array([155,154,151,150,149,148,147,143,143,142,142,141,141,142,142,143,143,148,149,150,153,154,159,160,162,163,164,165,166,170,170,171,171,172,172,171,171,170,170,165,164,163,161,160])
# y=np.array([713,714,714,715,715,716,716,720,721,722,724,725,733,734,736,737,738,743,743,744,744,745,745,744,744,743,743,742,742,738,737,736,734,733,724,723,722,721,720,715,715,714,714,713])

# fit_circle(x,y)
