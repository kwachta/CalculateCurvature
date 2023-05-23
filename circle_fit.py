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

    # extract middle and radius
    x_m = circ_param[0][0]
    y_m = circ_param[0][1]
    r_m = circ_param[0][2]

    # plot the circle for debugging purpose
    # plot_circle(x_m,y_m,r_m,x,y)
    return [x_m, y_m, r_m]


def find_circles(src) -> np.ndarray:
    """Function finds circles on the image\n
    Function returns 2D array with circles in the form of [x,y,R] parameters"""

    # Convert image to gray and blur it
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(src_gray, (3, 3))

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
                circle_list.append(circle_param)
                # show fitted circle - for debug purposes
                # circle_fit.plot_circle(circle_param[0][0],circle_param[0][1],circle_param[0][2],x,y)
    # print contours with circles
    circ_stack = np.vstack(circle_list)
    circ_stack_2 = np.squeeze(circ_stack)
    circ_stack_2 = circ_stack_2[np.argsort(circ_stack_2[:, 1])]
    # write image to file
    # if not cv.imwrite('output_test.jpg', copy_src):
    #        raise Exception("Could not write image")

    # Show gray pictures with detected circles
    # cv.imshow('Contours', copy_src)
    # cv.waitKey()
    return circ_stack_2


def find_sample_edges(src, threshold_value):
    """'function is working on the image without markers and their surroundings (only "edges" of the sample are present)"""
    R_src = src.copy()
    # BGR coding
    (B, G, R) = cv.split(R_src)
    # taking only red component of the image
    copy_src = R.copy()
    copy_src = cv.blur(copy_src, (3, 3))

    ret, threshold_img = cv.threshold(copy_src, threshold_value, 255, cv.THRESH_BINARY)

    grad_x = cv.Sobel(threshold_img, -1, 1, 0)
    # Find contours
    contours, hierarchy = cv.findContours(grad_x, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

    # get edges
    contour_list = []
    for contour in contours:
        moments = cv.moments(contour)

        # filter out short contours
        if (contour[:, :, 1].max() - contour[:, :, 1].min() < 100) or (
            moments["mu20"] == 0
        ):
            continue
        param = moments["mu02"] / moments["mu20"]
        area = cv.contourArea(contour)
        # check if is a circle
        if (param > 0.98) & (area > 100):
            contour_list.append(contour)
            cv.drawContours(R_src, contour_list, -1, (255, 0, 0), 2)
            x = []
            y = []
            for t in contour:
                x.append(t[0][0])
                y.append(t[0][1])
            x = np.r_[x]
            y = np.r_[y]

    # take all contours and join them into single one
    edge_array = np.squeeze(contour_list[0])
    for i in range(1, len(contour_list)):
        temp_array = np.squeeze(contour_list[i])
        edge_array = np.vstack((edge_array, temp_array))
    return edge_array
