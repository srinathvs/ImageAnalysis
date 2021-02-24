# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import os
import numpy as np
import cv2
import math
import os
import matplotlib.pyplot as plt
from skimage.transform import pyramid_gaussian


def question1(path, Sigmaval):
    # Gaussian blurring :
    image = cv2.imread(path)
    blur = cv2.GaussianBlur(image, ((2 * Sigmaval) + 1, (2 * Sigmaval) + 1), 0)
    plt.subplot(121), plt.imshow(image), plt.title('Original Image')
    plt.subplot(122), plt.imshow(blur), plt.title('Gaussian blurred image')
    plt.show()

    # Gradient of an image :
    sobelX = cv2.Sobel(blur, cv2.CV_64F, ksize=2 * Sigmaval + 1, dx=1, dy=0)
    sobelY = cv2.Sobel(blur, cv2.CV_64F, ksize=2 * Sigmaval + 1, dx=0, dy=1)

    plt.subplot(221), plt.imshow(blur), plt.title('Blurred image')
    plt.subplot(222), plt.imshow(sobelX), plt.title('XGradient')
    plt.subplot(223), plt.imshow(sobelY), plt.title('YGradient')
    plt.show()

    xabs = cv2.convertScaleAbs(sobelX)
    yabs = cv2.convertScaleAbs(sobelY)

    combinedimage = cv2.addWeighted(xabs, .5, yabs, .5, 0)
    magnitude = cv2.magnitude(sobelX, sobelY)

    # orientation is here, not sure on how to show it
    orientation = cv2.phase(sobelX, sobelY, angleInDegrees=True)

    theta = np.arctan2(sobelX, sobelY)
    oreintation_image = (theta + np.pi) * 255 / (2 * np.pi)

    plt.subplot(221), plt.imshow(combinedimage), plt.title("combined X and Y")
    plt.subplot(222), plt.imshow(magnitude), plt.title("Magnitude")
    plt.subplot(223), plt.imshow(oreintation_image), plt.title('Orientation')
    plt.show()

    # thresholding for finding non maximal edges
    blur_dup = blur.copy()
    threshold = eval(input("Enter the threshold(Preferably lesser than 1 : "))
    print(magnitude)
    for x in range(len(blur)):
        for y in range(len(blur[x])):
            if magnitude[x][y].all() < threshold:
                blur[x][y] = [0, 0, 0]
            else:
                blur[x][y] = [255, 255, 255]
    plt.subplot(121), plt.imshow(blur_dup), plt.title('Blur original')
    plt.subplot(122), plt.imshow(blur), plt.title('Post local Maxima')
    plt.show()

    weakthresh = eval(input("Enter weak threshold"))
    strongthresh = eval(input("Enter strong threshold"))


def question2(path, Pyramidsize):
    image = cv2.imread(path)
    layer = image.copy()
    gaussianpyramid = [layer]
    for i in range(Pyramidsize):
        plt.subplot(Pyramidsize, 2, i + 1)

        # using pyrDown() function
        layer = cv2.pyrDown(layer)
        gaussianpyramid.append(layer)
        plt.imshow(layer)
        cv2.imshow("str(i)", layer)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    plt.show()

    laplaciantop = gaussianpyramid[-1]
    laplacianpyramid = [laplaciantop]
    for i in range(Pyramidsize, 0, -1):
        size = (gaussianpyramid[i - 1].shape[1], gaussianpyramid[i - 1].shape[0])
        gaussExpanded = cv2.pyrUp(gaussianpyramid[i], dstsize=size)
        laplacian = cv2.subtract(gaussianpyramid[i - 1], gaussExpanded)
        laplacianpyramid.append(laplacian)
        plt.subplot(Pyramidsize, 2, i), plt.imshow(laplacian), plt.title('laplacian' + str(i))

    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Writepath = "C:\Users\srina\Desktop\Winter2021\ImageProcessing2\Base.png"
    path_to_image = input("Enter the path to the image : \t")
    sigma = eval(input("Enter the value of sigma :"))
    #question1(path_to_image, sigma)
    scale = eval(input("Enter number of images needed"))
    question2(path_to_image, scale)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
