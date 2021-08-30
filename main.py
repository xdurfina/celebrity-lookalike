import os
import pickle
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage import io, color
from skimage.feature import local_binary_pattern
from scipy.spatial.distance import euclidean
from itertools import combinations
from sklearn.metrics import auc

############################
# MADE BY JAROSLAV DURFINA #
############################

def getDicFromTxt():
    imageAnnotations = {}
    celebImages = {}

    facesList = os.listdir('faces')

    with open("identity_CelebA.txt", "r") as file:
        for line in file:
            tempLine = line.split(sep=' ')
            tempLine[1] = tempLine[1].rstrip('\n')
            if tempLine[0] in facesList:
                imageAnnotations.update({tempLine[0]: tempLine[1]})
                if tempLine[1] not in celebImages:
                    celebImages.update({tempLine[1]:[tempLine[0]]})
                else:
                    tempList = celebImages.get(tempLine[1])
                    tempList.append(tempLine[0])
                    celebImages.update({tempLine[1]: tempList})

    return imageAnnotations, celebImages

def saveIAandCI(imageanot, celebanot):
    with open('imageAnnotations.txt', 'wb') as fo:
        pickle.dump(imageanot, fo)
    with open('celebImages.txt', 'wb') as fi:
        pickle.dump(celebanot, fi)
    print('SAVING OF IA AND CA WAS SUCCESSFUL')

def loadIAandCI():
    with open('imageAnnotations.txt', 'rb') as fo:
        imageAnnotations = pickle.load(fo)
    with open('celebImages.txt', 'rb') as fi:
        celebImages = pickle.load(fi)
    print('LOADING OF IA AND CA WAS SUCCESSFUL')
    return imageAnnotations, celebImages

def getLBP(color_image):
    img = color.rgb2gray(color_image)
    patterns = local_binary_pattern(img, 8, 1)
    hist, _ = np.histogram(patterns, bins=np.arange(2 ** 8 + 1), density=True)
    return hist


def getSIFT(color_image):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(color_image, None)
    return kp, des


def getLBPandSIFTannotations():
    LBPannotations = {}
    SIFTannotations = {}
    for filename in os.listdir('faces'):
        img = cv2.imread('faces/' + filename)
        LBPanon = getLBP(img)
        img = cv2.imread('faces/' + filename, cv2.IMREAD_GRAYSCALE)
        SIFTkp, SIFTdes = getSIFT(img)

        img_SIFT_list = []
        tempSIFTkp = []
        for element in SIFTkp:
            tempList = [element.angle, element.class_id, element.octave, element.pt, element.response, element.size]
            tempSIFTkp.append(tempList)

        img_SIFT_list.append(tempSIFTkp)
        img_SIFT_list.append(SIFTdes)

        # Adding to dictionaries
        LBPannotations.update({str(filename): LBPanon})
        SIFTannotations.update({str(filename): img_SIFT_list})

    return LBPannotations, SIFTannotations


def saveAnnotations(LBPanot, SIFTanot):
    with open('LBPannotations.txt', 'wb') as fo:
        pickle.dump(LBPanot, fo)
    with open('SIFTannotations.txt', 'wb') as fi:
        pickle.dump(SIFTanot, fi)
    print('SAVING OF LBP AND SIFT WAS SUCCESSFUL')


def loadAnnotations():
    with open('LBPannotations.txt', 'rb') as fo:
        LBPanot = pickle.load(fo)
    with open('SIFTannotations.txt', 'rb') as fi:
        SIFTanot = pickle.load(fi)
    print('LOADING OF LBP AND SIFT ANNOTATIONS WAS SUCCESSFUL')


    newSIFTanot = {}
    for i, imageID in enumerate(SIFTanot):
        allKeypoints = []
        imageValue = SIFTanot.get(imageID)
        imageDes = imageValue[1]
        for j, keypoint in enumerate(imageValue[0]):
            tempKP = cv2.KeyPoint(x=keypoint[3][0], y=keypoint[3][1], _size=keypoint[5], _angle=keypoint[0],
                                  _response=keypoint[4], _octave=keypoint[2], _class_id=keypoint[1])
            allKeypoints.append(tempKP)
        newSIFTanot.update({imageID: [allKeypoints, imageDes]})

    return LBPanot, newSIFTanot


def getTP(celebimages):
    allCombinations = []
    for listt in celebimages.values():
        comb = combinations(listt, 2)
        for combination in list(comb):
            allCombinations.append(list(combination))
    thisReturnAllCombinations = random.sample(allCombinations, 500)
    print('TP COMBINATIONS GENERATED SUCCESSFULLY')
    return thisReturnAllCombinations

def getIP(celebimages):
    IPcombinations = []
    for i in range(500):
        celebrity1 = random.choice(list(celebimages.values()))
        photo1 = random.choice(celebrity1)
        celebrity2 = random.choice(list(celebimages.values()))
        photo2 = random.choice(celebrity2)
        IPcombinations.append([photo1, photo2])
    print('IP COMBINATIONS GENERATED SUCCESSFULLY')
    return IPcombinations


def compareTwoImagesWithLBP(x, y):
    dst = euclidean(x, y)
    return dst



def compareAllImagesWithLBP(lbpannotations, combinations):
    allDistances = []
    for index, combination in enumerate(combinations):
        image1 = lbpannotations.get(combination[0])
        image2 = lbpannotations.get(combination[1])
        dst = compareTwoImagesWithLBP(image1, image2)
        allDistances.append(dst)

    thisReturn = fromDistancesToDivided(allDistances)
    return thisReturn

def compareAllImagesWithSIFT(siftannotations, combinations):
    flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict())
    allDistances = []
    for index, combination in enumerate(combinations):
        image1 = siftannotations.get(combination[0])
        kp1 = image1[0]
        desc1 = image1[1]
        image2 = siftannotations.get(combination[1])
        kp2 = image2[0]
        desc2 = image2[1]

        matches = flann.knnMatch(desc1, desc2, k=2)

        good_points = []
        for m, n in matches:
            if m.distance < 0.9*n.distance:
                good_points.append(m)

        #Visualize
        img = cv2.imread('faces/' + combination[0] )
        imgToCompare = cv2.imread('faces/' + combination[1])
        result = cv2.drawMatches(img, kp1, imgToCompare, kp2, good_points, None)
        cv2.imshow('Matches found', cv2.resize(result, None, fx=3, fy=3))
        cv2.waitKey(0)

        number_keypoints = 0
        if len(kp1) <= len(kp2):
            number_keypoints = len(kp1)
        else:
            number_keypoints = len(kp2)

        howGood = len(good_points) / number_keypoints * 100

        allDistances.append(howGood)

    thisReturn = fromDistancesToDivided(allDistances)

    return thisReturn

def fromDistancesToDivided(distances):
    normalized = [float(i) / max(distances) for i in distances]
    rounded = [round(num, 2) for num in normalized]
    rounded.sort()
    boxes = np.linspace(0, 1, num=100)
    roundedBoxes = [round(num, 2) for num in boxes]

    numberOfNumbers = [0] * 100
    for index, number in enumerate(roundedBoxes):
        for percent in rounded:
            if percent <= number:
                numberOfNumbers[index] += 1

    divided = [x / len(rounded) for x in numberOfNumbers]
    return divided


def plotROC(x,y,name):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, label=name)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    plt.show()


def celebritySearch(image,siftannot):
    best = 0
    bestImage = ''
    flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict())
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    jaroKp, jaroDes = getSIFT(img)

    for imagex in siftannot:
        key = imagex
        celebKp = siftannot[key][0]
        celebDes = siftannot[key][1]

        matches = flann.knnMatch(jaroDes, celebDes, k=2)

        good_points = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good_points.append(m)

        number_keypoints = 0
        if len(jaroKp) <= len(celebKp):
            number_keypoints = len(jaroKp)
        else:
            number_keypoints = len(celebKp)

        howGood = len(good_points) / number_keypoints * 100

        if (howGood > best) and (howGood < 50):
            bestImage = key
            best = howGood

    return bestImage


if __name__ == "__main__":
    # imageAnnotations, celebImages = getDicFromTxt()
    # saveIAandCI(imageAnnotations, celebImages)
    imageAnnotations, celebImages = loadIAandCI()

    # LBPannotations, SIFTannotations = getLBPandSIFTannotations()
    # saveAnnotations(LBPannotations, SIFTannotations)
    LBPannotations, SIFTannotations = loadAnnotations()

    TPcombinations = getTP(celebImages)
    IPcombinations = getIP(celebImages)

    all_TP_Distances_LBP = compareAllImagesWithLBP(LBPannotations, TPcombinations)
    all_IP_Distances_LBP = compareAllImagesWithLBP(LBPannotations, IPcombinations)

    all_TP_Distances_SIFT = compareAllImagesWithSIFT(SIFTannotations, TPcombinations)
    all_IP_Distances_SIFT = compareAllImagesWithSIFT(SIFTannotations, IPcombinations)

    plotROC(all_TP_Distances_SIFT, all_IP_Distances_SIFT, 'SIFT - ROC CURVE')
    plotROC(all_TP_Distances_LBP, all_IP_Distances_LBP, 'LBP - ROC CURVE')

    print('AUC OF LBP')
    AUClbp = f'ROC Curve (AUC={auc(all_TP_Distances_LBP, all_IP_Distances_LBP):.4f})'
    print(AUClbp)

    print('AUC OF SIFT')
    AUCsift = f'ROC Curve (AUC={auc(all_TP_Distances_SIFT, all_IP_Distances_SIFT):.4f})'
    print(AUCsift)

    bestImage = celebritySearch('yourphoto.jpg', SIFTannotations)
    print('You look like:', bestImage)

    imgJaro = cv2.imread('yourphoto.jpg')
    imgBest = cv2.imread('faces/' + bestImage)
    cv2.imshow('You look like:', imgBest)
    cv2.imshow('This is you:', imgJaro)
    cv2.waitKey(0)


