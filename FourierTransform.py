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


def getCannyEdges(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    plt.subplot(221), plt.imshow(image, cmap='gray'), plt.title('Base image')
    line_img = image.copy()
    line_img1 = image.copy()
    line_instance = image.copy()
    # Setting different sigma sizes to check results
    canny3 = cv2.Canny(image, 150, 200, apertureSize=3)
    canny5 = cv2.Canny(image, 150, 200, apertureSize=5)
    canny7 = cv2.Canny(image, 150, 200, apertureSize=7)

    plt.subplot(222), plt.imshow(canny3, cmap='gray'), plt.title('Canny 3x3')
    plt.subplot(223), plt.imshow(canny5, cmap='gray'), plt.title('Canny 5x5')
    plt.subplot(224), plt.imshow(canny7, cmap='gray'), plt.title('Canny 7x7')
    plt.show()

    hough_accum = cv2.HoughLines(canny3, 1, np.pi / 180, 120, min_theta=0, max_theta=90)
    print(hough_accum)
    for line in hough_accum:
        rho, teeta = line[0]
        a = np.cos(teeta)
        b = np.sin(teeta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + line_instance.shape[0] * (-b))
        y1 = int(y0 + line_instance.shape[1] * a)
        x2 = int(x0 - line_instance.shape[0] * (-b))
        y2 = int(y0 - line_instance.shape[1] * a)
        cv2.line(line_instance, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('hough_eval', line_instance)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hough = cv2.HoughLinesP(canny3, 1, np.pi / 360, 100, minLineLength=20, maxLineGap=10)

    for index in range(len(hough)):
        for ptx1, pty1, ptx2, pty2 in hough[index]:
            cv2.line(line_img, (ptx1, pty1), (ptx2, pty2), (0, 255, 0), 1)
    plt.subplot(311), plt.imshow(image, cmap='gray'), plt.title('Base image')
    plt.subplot(312), plt.imshow(line_img), plt.title('Lines')
    hough1 = cv2.HoughLinesP(canny7, 1, np.pi / 360, 100, minLineLength=20, maxLineGap=10)

    for index in range(len(hough1)):
        for ptx1, pty1, ptx2, pty2 in hough1[index]:
            cv2.line(line_img1, (ptx1, pty1), (ptx2, pty2), (0, 255, 0), 1)

    plt.subplot(313), plt.imshow(line_img1, cmap='gray'), plt.title('Lines with sigma = 7')
    plt.show()

    cv2.imshow('hough', line_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def q2():
    # create the three images in python
    one_img = np.zeros((1000, 1000, 1), dtype="uint8")
    rows, cols, ch = one_img.shape
    one_img[:, int(cols / 2):-1] = 255

    cv2.imshow('one', one_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    two_img = np.zeros((1000, 1000, 1), dtype="uint8")
    r2, c2, ch2 = two_img.shape

    two_img[int(r2 / 4):int(3 * r2 / 4), int(c2 / 4):int(3 * c2 / 4)] = 255

    cv2.imshow('two', two_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    three_img_inst = two_img.copy()
    M = cv2.getRotationMatrix2D((int(r2 / 2), int(c2 / 2)), 45, 1)
    three_img = cv2.warpAffine(three_img_inst, M, (two_img.shape[1], two_img.shape[0]))

    cv2.imshow('three', three_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # put paths to the stored images in path1,2 and 3
    pathone = "C:\\Users\\srina\\Desktop\\Winter2021\\ImageProcessing2\\Assignment-4\\one.png"
    pathtwo = 'C:\\Users\\srina\\Desktop\\Winter2021\\ImageProcessing2\\Assignment-4\\two.png'
    paththree = "C:\\Users\\srina\\Desktop\\Winter2021\\ImageProcessing2\\Assignment-4\\three.png"

    # reimporting the three images from file
    imp_1 = cv2.imread(pathone, 0)
    imp_2 = cv2.imread(pathtwo, 0)
    imp_3 = cv2.imread(paththree, 0)

    f = np.fft.fft2(imp_1)
    fshift = np.fft.fftshift(f)
    # this step is to bring values closer together to represent easily, else, range of values would be too vast
    magnitude_spectrum1 = 20 * np.log(np.abs(fshift))

    f1 = np.fft.fft2(imp_2)
    fshift1 = np.fft.fftshift(f1)
    # this step is to bring values closer together to represent easily, else, range of values would be too vast
    magnitude_spectrum2 = 20 * np.log(np.abs(fshift1))

    f2 = np.fft.fft2(imp_3)
    fshift2 = np.fft.fftshift(f2)
    # this step is to bring values closer together to represent easily, else, range of values would be too vast
    magnitude_spectrum3 = 20 * np.log(np.abs(fshift2))

    magnitude_spectrum1[np.isneginf(magnitude_spectrum1)] = 0
    magnitude_spectrum2[np.isneginf(magnitude_spectrum2)] = 0
    magnitude_spectrum3[np.isneginf(magnitude_spectrum3)] = 0

    plt.subplot(311), plt.imshow(magnitude_spectrum1, cmap='gray'), plt.title('spec1')
    plt.subplot(312), plt.imshow(magnitude_spectrum2, cmap='gray'), plt.title('spec2')
    plt.subplot(313), plt.imshow(magnitude_spectrum3, cmap='gray'), plt.title('spec3')

    plt.show()


def q3(path):
    cam = cv2.imread(path, 0)

    cv2.imshow('Cameraman', cam)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    sigma1 = 10
    sigma2 = 30

    plt.subplot(321), plt.imshow(cam, cmap='gray'), plt.title('Cameraman')

    # Create gaussian kernels for convolution
    Gauss_kernel1 = cv2.getGaussianKernel(sigma=sigma1, ksize=(sigma1))
    Gauss_kernel2 = cv2.getGaussianKernel(sigma=sigma2, ksize=(sigma2))

    f = np.fft.fft2(cam)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    plt.subplot(322), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('Spectrum')

    # Gaussian blurring FFT
    filter1 = cv2.filter2D(magnitude_spectrum, -1, kernel=Gauss_kernel1)
    filter2 = cv2.filter2D(magnitude_spectrum, -1, kernel=Gauss_kernel2)

    filtert1 = cv2.filter2D(cam, -1, Gauss_kernel1)
    filtert2 = cv2.filter2D(cam, -1, Gauss_kernel2)

    cv2.imshow('Testign1', filtert1)
    cv2.imshow('Testing2', filtert2)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.subplot(323), plt.imshow(filter1, cmap='gray'), plt.title('Filtered(sigma=10)')
    plt.subplot(324), plt.imshow(filter2, cmap='gray'), plt.title('Filtered(sigma=30)')

    # Getting IFT

    iffshift1 = np.fft.ifftshift(filter1)
    ret1 = np.fft.ifft2(iffshift1)
    iffshift2 = np.fft.ifftshift(filter2)
    ret2 = np.fft.ifft2(iffshift2)

    ret1 = np.abs(np.log(np.abs(ret1)))
    ret2 = np.abs(np.log(np.abs(ret2)))

    print("first vals are : ", ret1)
    print("Second vals are : ", ret2)

    plt.subplot(325), plt.imshow(ret1, cmap='gray'), plt.title('IFFT(sigma=10)')
    plt.subplot(326), plt.imshow(ret2, cmap='gray'), plt.title('IFFT(sigma=30)')

    plt.show()


if __name__ == '__main__':
    # Writepath1 = "C:\Users\srina\Desktop\Winter2021\ImageProcessing2\Assignment-4\house.jpg"
    # path_to_image = input("Enter the path to the image : \t")
    # getCannyEdges(path_to_image)
    # q2()
    q3("C:\\Users\\srina\\Desktop\\Winter2021\\ImageProcessing2\\Assignment-4\\cameraman.tif")
