import cv2
import os
import numpy as np
import cv2
import math
from scipy import signal
import os
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from skimage.transform import pyramid_gaussian


def question1(path):
    # Gaussian blurring :
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    blur = cv2.GaussianBlur(image, ((2 * 10) + 1, (2 * 10) + 1), 0)
    plt.subplot(121), plt.imshow(image), plt.title('Original Image')
    plt.subplot(122), plt.imshow(blur), plt.title('Gaussian blurred image')
    plt.show()

    # Gradient of an image :
    sobelX = cv2.Sobel(blur, cv2.CV_64F, ksize=2 * 3 + 1, dx=1, dy=0)
    sobelY = cv2.Sobel(blur, cv2.CV_64F, ksize=2 * 3 + 1, dx=0, dy=1)

    plt.subplot(321), plt.imshow(image, cmap='gray'), plt.title(' image')
    plt.subplot(322), plt.imshow(sobelX, cmap='gray'), plt.title('XGradient')
    plt.subplot(323), plt.imshow(sobelY, cmap='gray'), plt.title('YGradient')

    combinedimage = cv2.addWeighted(sobelX, .5, sobelY, .5, 0)
    plt.subplot(324), plt.imshow(combinedimage, cmap='gray'), plt.title('combined edges')

    magnitude = cv2.magnitude(sobelX, sobelY)
    plt.subplot(325), plt.imshow(magnitude, cmap='gray'), plt.title('magnitude')
    plt.show()
    print("Magnitude is : \n")
    print(magnitude)
    edges = np.argwhere(magnitude[:, :])

    length, width = image.shape
    # size of radius starts at 10 pixels and goes upto quarter the image size in width or length, whichever is greater
    Rmax = max((length / 4), (width / 4))
    Rmin = 60
    RangeR = int(Rmax - Rmin)
    print(RangeR)
    # create accumulator
    accumulator = np.zeros((length, width, int(Rmax)), dtype=np.uint64)
    accumdict = {}
    for rowval in range(len(magnitude)):
        for colval in range(len(magnitude[rowval])):
            if magnitude[rowval][colval] > 0:
                radius = int(Rmin)
                while radius < int(Rmax):
                    a, b = find_ab(radius, rowval, colval)
                    print("a and b  returned are : ", a, b)
                    if a and b > 0:
                        accumulator[int(a), int(b), int(radius)] += magnitude[rowval][colval]
                        if (int(a), int(b), int(radius)) in accumdict.keys():
                            accumdict[(int(a), int(b), int(radius))] += magnitude[rowval][colval]
                        else:
                            accumdict[(int(a), int(b), int(radius))] = 1
                    radius += 10
    img2 = ndimage.maximum_filter(accumulator, size=(5, 5, 5))
    img_thresh = img2.mean()+img2.std()*6
    labels, num_labels = ndimage.label(accumulator > img_thresh)
    coords = ndimage.measurements.center_of_mass(accumulator, labels=labels, index=np.arange(1, num_labels + 1))[:5]

    sortedpeaks = sorted(accumdict.items(), key=lambda x: x[1], reverse=True)[:50]
    print("Printing sorted peaks")
    print(coords)

    for tupleinstance in coords:
        print(tupleinstance)
        x, y, radius = tupleinstance
        print("the x, y, radius values are : ", x, y, radius)
        image = cv2.circle(image, (int(x), int(y)), int(radius), (0, 0, 255), 5)

    cv2.imshow('Final_circles', image)
    cv2.waitKey(0)
    cv2.destroyWindow('Final_circles')

    plt.subplot(111), plt.imshow(image, cmap = 'gray'), plt.title('circles')
    plt.show()


# using pythogras theorm, for pythogorean triplets the sides are  ( m^2-1, 2m and m^2 + 1). Here, we know the square of the radius, so m^2 +1 is known
def find_ab(radius, x, y):
    msquare = radius - 1
    m = math.sqrt(msquare)
    print("Radius and m is : ", radius, m, msquare)
    if is_integer(m) and m > 0 and is_integer(x) and is_integer(y):
        side1 = msquare - 1
        side2 = 2 * m
        print("side 1 and side 2 are : ", side1, side2)
        print("x and y are :", x, y)

        a = side2 - x
        b = side1 - y

        a = abs(a)
        b = abs(b)
        print("a and b are : ", a, b)
        print(is_integer(a), is_integer(b))
        return int(a), int(b)
    else:
        return 0, 0


def is_integer(n):
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False


if __name__ == '__main__':
    # Writepath = "C:\Users\srina\Desktop\Winter2021\ImageProcessing2\Assignment2\british-coins.jpg"
    path_to_image = input("Enter the path to the image : \t")
    question1(path_to_image)
