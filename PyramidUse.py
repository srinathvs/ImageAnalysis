import cv2
import os
import numpy as np
import cv2
import math
from scipy import signal
import os
import matplotlib.pyplot as plt


def question1(path1, path2):
    imageA = cv2.imread(path1)
    imageB = cv2.imread(path2)

    Pyramidsize = 4

    layer = imageA.copy()

    gaussianpyramid = [layer]
    for number in range(Pyramidsize):
        # using pyrDown() function
        layer = cv2.pyrDown(layer)
        gaussianpyramid.append(layer)
        cv2.imshow(str(number), layer)

    layerB = imageB.copy()

    gaussianpyramidB = [layerB]
    for numberx in range(Pyramidsize):
        # using pyrDown() function
        layerB = cv2.pyrDown(layerB)
        gaussianpyramidB.append(layerB)
        cv2.imshow(str(numberx), layerB)

    # generate Laplacian Pyramid for Images

    laplacianA = [gaussianpyramid[-1]]
    for numberl in range(4, 0, -1):
        size = (gaussianpyramid[numberl - 1].shape[1], gaussianpyramid[numberl - 1].shape[0])
        gaussExpanded = cv2.pyrUp(gaussianpyramid[numberl], dstsize=size)
        Laplacian_currA = cv2.subtract(gaussianpyramid[numberl - 1], gaussExpanded)
        laplacianA.append(Laplacian_currA)
        cv2.imshow(str(numberl), Laplacian_currA)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    laplacianB = [gaussianpyramidB[-1]]
    for numberb in range(4, 0, -1):
        size = (gaussianpyramidB[numberb - 1].shape[1], gaussianpyramidB[numberb - 1].shape[0])
        gaussExpanded = cv2.pyrUp(gaussianpyramidB[numberb], dstsize=size)
        Laplacian_currB = cv2.subtract(gaussianpyramidB[numberb - 1], gaussExpanded)
        laplacianB.append(Laplacian_currB)
        cv2.imshow(str(numberb), Laplacian_currB)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    FinalImage = []
    num = 1
    for lapA, lapB in zip(laplacianA, laplacianB):
        w, h, dpt = lapB.shape
        result = cv2.matchTemplate(lapA, lapB, cv2.TM_CCORR)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(lapA, top_left, bottom_right, 255, 2)
        plt.subplot(3, 2, num)
        plt.imshow(lapA)
        plt.title('Showing  : ' + str(num))
        num += 1
        FinalImage.append(lapA)
    plt.show()

    # now reconstruct
    final_image = FinalImage[0]
    for i in range(1, 4):
        size = (FinalImage[i].shape[1], FinalImage[i].shape[0])
        final_image = cv2.pyrUp(final_image, dstsize=size)
        final_image = cv2.add(final_image, FinalImage[i])
        plt.subplot(3, 2, i)
        plt.imshow(final_image)
        plt.title('Final Image' + str(i))
        cv2.imshow('Final RESULT', final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    plt.show()




def question2(pathA, pathB):
    imageA = cv2.imread(pathA)

    imageB = cv2.imread(pathB)

    # finding Gaussian pyramids for both images
    Pyramidsize = 4

    layer = imageA.copy()

    gaussianpyramid = [layer]
    for number in range(Pyramidsize):
        # using pyrDown() function
        layer = cv2.pyrDown(layer)
        gaussianpyramid.append(layer)
        cv2.imshow(str(number), layer)

    layerB = imageB.copy()

    gaussianpyramidB = [layerB]
    for numberx in range(Pyramidsize):
        # using pyrDown() function
        layerB = cv2.pyrDown(layerB)
        gaussianpyramidB.append(layerB)
        cv2.imshow(str(numberx), layerB)

    # generate Laplacian Pyramid for Images

    laplacianA = [gaussianpyramid[-1]]
    for numberl in range(4, 0, -1):
        size = (gaussianpyramid[numberl - 1].shape[1], gaussianpyramid[numberl - 1].shape[0])
        gaussExpanded = cv2.pyrUp(gaussianpyramid[numberl], dstsize=size)
        Laplacian_currA = cv2.subtract(gaussianpyramid[numberl - 1], gaussExpanded)
        laplacianA.append(Laplacian_currA)
        cv2.imshow(str(numberl), Laplacian_currA)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    laplacianB = [gaussianpyramidB[-1]]
    for numberb in range(4, 0, -1):
        size = (gaussianpyramidB[numberb - 1].shape[1], gaussianpyramidB[numberb - 1].shape[0])
        gaussExpanded = cv2.pyrUp(gaussianpyramidB[numberb], dstsize=size)
        Laplacian_currB = cv2.subtract(gaussianpyramidB[numberb - 1], gaussExpanded)
        laplacianB.append(Laplacian_currB)
        cv2.imshow(str(numberb), Laplacian_currB)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Adding halves of the image at each level :
    FinalImage = []
    num = 1
    for lapA, lapB in zip(laplacianA, laplacianB):
        rows, cols, dpt = lapA.shape
        # taking left columns from laplcian A and right columns from laplacianB
        blend = np.hstack((lapA[:, 0:int(cols / 2)], lapB[:, int(cols / 2):]))
        plt.subplot(3, 2, num)
        cv2.imshow('blend', blend)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        FinalImage.append(blend)
        plt.imshow(blend, cmap='gray')
        plt.title('Blend' + str(num))
        num += 1

    plt.show()

    # now reconstruct
    final_image = FinalImage[0]
    for i in range(1, 4):
        size = (FinalImage[i].shape[1], FinalImage[i].shape[0])
        final_image = cv2.pyrUp(final_image, dstsize=size)
        final_image = cv2.add(final_image, FinalImage[i])
        plt.subplot(3, 2, i)
        plt.imshow(final_image)
        plt.title('Final Image' + str(i))
        cv2.imshow('Final RESULT', final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    plt.show()


def getimages():
    path_to_image = input("Enter the path to the first image : \t")
    path_to_image2 = input("Enter the path to the second image : \t")
    return path_to_image, path_to_image2


if __name__ == '__main__':
    # Q1ref : "C:\Users\srina\Desktop\Winter2021\ImageProcessing2\Assignment-3\ref.png"
    # Q1tem : "C:\Users\srina\Desktop\Winter2021\ImageProcessing2\Assignment-3\temp.png"
    # enter path as reference then template
    pq1, pq2 = getimages()
    question1(pq1, pq2)

    # Q21 : "C:\Users\srina\Desktop\Winter2021\ImageProcessing2\Assignment-3\ImAbl.jpg"
    # Q22 : "C:\Users\srina\Desktop\Winter2021\ImageProcessing2\Assignment-3\ImBbl.jpg"
    # p1, p2 = getimages()
    # question2(p1, p2)
