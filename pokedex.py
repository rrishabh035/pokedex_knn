import pandas as pd

import numpy as np

import cv2

​

def handleData():

    dataFrame = pd.read_csv("Train/train.csv")

    data = np.array(dataFrame)

    rootFolder = "Train/Images/"

    fileNameTemp = rootFolder + data[0,0]

    img = cv2.imread(fileNameTemp)

    img = cv2.resize(img,(150,150))

    imgTempPixel = np.array(np.ravel(img),ndmin=2)    

    cv2.imwrite("image.jpg",img)

    for x in range(1,len(data)):

        fileNameTemp = rootFolder + data[x,0]

        img = cv2.imread(fileNameTemp)

        img = cv2.resize(img,(150,150))

        imgTempPixel = np.append(imgTempPixel,np.array(np.ravel(img),ndmin=2),axis=0)

        

    return imgTempPixel,data[0:,1]

    

    

def ecDistance(instance1, instance2):

    return np.sum((instance1-instance2)**2)**.5

    

def testCaseGet():

    dataFrame = pd.read_csv("Test/test.csv")

    data = np.array(dataFrame)

    rootFolder = "Test/Images/"

    fileNameTemp = rootFolder + data[0,0]

    img = cv2.imread(fileNameTemp)

    img = cv2.resize(img,(150,150)) 

    imgTempPixel = np.array(np.ravel(img),ndmin=2)    

      

    for x in range(1,len(data)):

        fileNameTemp = rootFolder + data[x,0]

        img = cv2.imread(fileNameTemp)

        img = cv2.resize(img,(150,150))

        imgTempPixel = np.append(imgTempPixel,np.array(np.ravel(img),ndmin=2),axis=0)

        

    return imgTempPixel

data,dataStamp = handleData()

def knn(test_case):

    

    result = []

    for k in range(len(data)):

        result.append((ecDistance(data[k,],test_case),dataStamp[k]))

    result = sorted(result)

    result = np.array(result[:1])

    b = np.unique(result[:,1],return_counts=True)

    idx = np.argmax(b[1])

    pred = b[0][idx]

    return pred

    

    

