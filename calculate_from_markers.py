from __future__ import print_function
import os
import cv2 as cv
import numpy as np
import circle_fit

write_to_file = True  # should locations of deteced circles be written to a file?

catalog_names = [
    "02_S1_3",
    "03_S1_5",
    "04_S1_6",
]

print("Filename\tCurvature [1/pix]")

for i in range(1, len(catalog_names)):
    # Load source image
    img_path = "Example_images\\" + catalog_names[i][:] + "\\"
    img_list = os.listdir(img_path)

    for img_file in img_list:
        if img_file.endswith(".jpg" or ".JPG"):
            src = cv.imread(img_path + img_file)
            if src is None:
                print("Could not open or find the image:", img_file)
                exit(0)
            circles_in_photo = circle_fit.find_circles(src)
            if write_to_file:
                result_file = open(
                    img_path + os.path.splitext(img_file)[0] + ".txt", "w+"
                )
                # save markers locations into files
                result_file.write(img_file + "\n")
                result_file.write("x[pix],y[pix],R[pix]\n")
                np.savetxt(result_file, circles_in_photo, fmt="%.4f", delimiter=",")
                result_file.close()
            x = circles_in_photo[:, 0]
            y = circles_in_photo[:, 1]
            sample_fitted_circle = circle_fit.fit_circle(x, y)
            print(img_file + "\t" + str(1 / sample_fitted_circle[2]))
