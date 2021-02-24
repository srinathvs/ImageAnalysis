import cv2
import os
import numpy as np
import cv2
import math
from scipy import signal
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from skimage.transform import pyramid_gaussian

# value of epsilon used
epsilon = 1


def compress(path):
    global epsilon
    image = cv2.imread(path, 0)
    r, c = image.shape
    plt.subplot(421), plt.imshow(image, cmap='gray'), plt.title('main image')
    # Doing the compression to the rows
    image_avg1a, diff, sec = recursive_eval_rows(image, True, 1)
    comp1 = image_avg1a.copy()
    plt.subplot(423), plt.imshow(image_avg1a, cmap='gray'), plt.title('First compression')

    decompressed1 = decompress(comp1, True, diff, sec)
    plt.subplot(424), plt.imshow(decompressed1, cmap='gray'), plt.title('First decompression')

    # doing the same thing to half the columns
    image_avg1, diff1, sec1 = recursive_eval_rows(image_avg1a, False, 1)
    comp2 = image_avg1.copy()
    plt.subplot(425), plt.imshow(image_avg1, cmap='gray'), plt.title('Second compression')

    # multistage decompression for both previous stages
    decompress2 = decompress(comp2, False, diff1, sec1)
    decompress2a = decompress(decompress2, True, diff, sec)
    plt.subplot(426), plt.imshow(decompress2a, cmap='gray'), plt.title('Second decompression')

    # getting half the rows for third compression
    r1, c1 = image_avg1.shape
    img2copy = image_avg1.copy()
    img2copy = image_avg1[0:int(r1 / 2)][0:c1]
    # third compression rowwise in half the previously averaged rows
    image_avg2a, diff2a, sec2a = recursive_eval_rows(img2copy, False, 1)
    x, y = image_avg2a.shape
    print("Shape of compression is : ", x, y)
    # multistage compression
    comp3 = image_avg2a.copy()

    image_changed = image_avg1.copy()
    image_changed[0:x][0:int(y / 2)] = image_avg2a
    cv2.imshow('third compression', image_changed)
    cv2.waitKey(0)
    cv2.destroyWindow('third compression')
    plt.subplot(427), plt.imshow(image_avg2a, cmap='gray'), plt.title('third ccompression')

    # third decompression - but we have to de-attach compressed part to the uncompressed parts of the image now :
    decompress3a = decompress(comp3, False, diff2a, sec2a)
    image_changed1 = image_avg1.copy()
    # taking the partial image found in second iteration :
    image_changed1[0:x][0:int(y / 2)] = decompress3a
    cv2.imshow('third compression', image_changed1)
    cv2.waitKey(0)
    cv2.destroyWindow('third compression')

    decompress3b = decompress(image_changed1, False, diff1, sec1)
    cv2.imshow('third decompression', decompress3b)
    cv2.waitKey(0)
    cv2.destroyWindow('third decompression')
    decompress3c = decompress(decompress3b, True, diff, sec)
    cv2.imshow('third decompression', decompress3c)
    cv2.waitKey(0)
    cv2.destroyWindow('third decompression')
    plt.subplot(428), plt.imshow(decompress3c, cmap='gray'), plt.title("Third decompression")

    plt.show()


def recursive_eval_rows(image, row_state, iteration):
    global epsilon
    if not row_state:
        image = image.T

    r, c = image.shape
    print("Image shape is  : ", image.shape)

    halfc = int(c / 2 * iteration)
    secondary_image = np.zeros((r, halfc), dtype=np.double)
    differences_matrix = np.zeros((r, halfc), dtype=np.double)
    if halfc > 1:
        for row in range(len(image)):
            for col in range(0, int(len(image[row])), 2):
                if col + 1 <= int(c / 2 * iteration):
                    if col == 0:
                        secondary_image[row][col] = (image[row][col] + image[row][col + 1]) / 2
                        differences_matrix[row][col] = (image[row][col] - image[row][col + 1]) / 2
                        if differences_matrix[row][col] < epsilon:
                            differences_matrix[row][col] = 0
                        if secondary_image[row][col] < epsilon:
                            secondary_image[row][col] = 0

                    else:
                        secondary_image[row][col - 1] = (image[row][col] + image[row][col + 1]) / 2
                        differences_matrix[row][col - 1] = (image[row][col] - image[row][col + 1]) / 2
                        if differences_matrix[row][col - 1] < epsilon:
                            differences_matrix[row][col - 1] = 0
                        if secondary_image[row][col - 1] < epsilon:
                            secondary_image[row][col] = 0

        print(differences_matrix)
        print(secondary_image)
        for row in range(len(image)):
            image[row][0:halfc] = secondary_image[row][0:halfc]
            image[row][halfc:c] = differences_matrix[row][0::-1]

        print(image)
        if not row_state:
            image = image.T
            secondary_image = secondary_image.T
            differences_matrix = differences_matrix.T
        cv2.imshow('compressed', image)
        cv2.imshow('secondary', secondary_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return image, differences_matrix, secondary_image
    return image, differences_matrix, secondary_image


def decompress(image, row_state, difference, secondary):
    if not row_state:
        image = image.T
        secondary = secondary.T
        difference = difference.T
    r, c = image.shape
    rhalf = int(r / 2)
    chalf = secondary.shape[1]
    print("Shape of image is : ", image.shape, "\n shape of secondary is : ", secondary.shape, "\n",
          "shape of difference is : ", difference.shape, "value of c is : ", c)
    for i in range(0, r):
        for j in range(0, chalf):
            if j == 0:
                image[i][j] = secondary[i][j] + difference[i][j]
                image[i][j + 1] = secondary[i][j] - difference[i][j]
            elif 0 < j < c:
                image[i][j * 2] = secondary[i][j] + difference[i][j]
                if j < c:
                    image[i][j * 2 + 1] = secondary[i][j] - difference[i][j]

    if not row_state:
        image = image.T
        secondary = secondary.T
        difference = difference.T
    cv2.imshow('decompressed', image)
    cv2.waitKey(0)
    cv2.destroyWindow('decompressed')
    return image


def Q1a(path1, path2):
    print("Fidning FFT for Lena and Iris")
    lena = cv2.imread(path2, 0)
    iris = cv2.imread(path1, 0)
    plt.subplot(221), plt.imshow(lena, cmap='gray'), plt.title('lena')
    plt.subplot(222), plt.imshow(iris, cmap='gray'), plt.title('iris')

    fl = np.fft.fft2(lena)
    fshift1 = np.fft.fftshift(fl)
    magnitude_spectrum1 = 20 * np.log(np.abs(fshift1))

    plt.subplot(223), plt.imshow(magnitude_spectrum1, cmap='gray'), plt.title('Spectrum lena')

    fi = np.fft.fft2(iris)
    fshift2 = np.fft.fftshift(fi)
    magnitude_spectrum2 = 20 * np.log(np.abs(fshift2))

    plt.subplot(224), plt.imshow(magnitude_spectrum2, cmap='gray'), plt.title('Spectrum iris')
    plt.show()

if __name__ == '__main__':
    print("Performing Haar wavelet compresion")

    Writepath1 = "C:\\Users\\srina\\Desktop\\Winter2021\\ImageProcessing2\\Assignment-4\\house.jpg"
    Writepath2 = "C:\\Users\\srina\\Desktop\\Winter2021\\ImageProcessing2\\Assignment-5\\iris-illustration.png"
    Writepath3 = "C:\\Users\\srina\\Desktop\\Winter2021\\ImageProcessing2\\Assignment-5\\lena.jpg"

    compress(Writepath1)
    # Q1a(Writepath2, Writepath3)
