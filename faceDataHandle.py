# coding:utf-8
import os
import random

import numpy as np
import cv2
from scipy.misc import imresize
import pandas as pd

def readFaceData(path,long=300,wide=300,deep=3):
    imageList = os.listdir(path)
    n = len(imageList)
    X = np.empty((n,long,wide,deep))
    Y = []
    for i,imageName in enumerate(imageList):
        X[i,:,:,:] = imresize(cv2.imread(path+'/'+imageName),(long,wide))
        Y.append(imageName[0:4])
    label = [Y[0],Y[-1]]
    return X,pd.get_dummies(np.array(Y)).values,label


def shuffle(X, Y):
    Y = np.array(Y)
    n = len(Y)
    index = random.sample(range(n), n)
    x_, y_ = X[index, :, :, :], Y[index,:]
    return x_, y_


def getTrainData(path="./data/trainingData",longs=300,wide=300):
    X, Y,label = readFaceData(path, longs, wide)
    x_train,y_train = shuffle(X,Y)
    return x_train,y_train,label

# x,y,label = getTrainData()
# print(label)