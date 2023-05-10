import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0) #0 is the id number for our webcam
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300

folder = "Data/C"
counter = 0
labels = ["A", "B", "C"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    #crop the image once we get it
    if hands:
        hand = hands[0] #0 because we are only using one hand
        #get bounding box information out of that
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        #crop the image based on the bbox values
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset] #img is a matrix, parameters = start height to end height & start width to end width
        imgCropShape = imgCrop.shape


        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            calculated_width = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (calculated_width, imgSize))
            imgResizeShape = imgResize.shape
            width_gap =  math.ceil((imgSize - calculated_width)/2)
            imgWhite[:, width_gap:calculated_width+width_gap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw = False)
            print(prediction, index)

        else:
            k = imgSize / w
            calculated_height = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, calculated_height))
            imgResizeShape = imgResize.shape
            height_gap = math.ceil((imgSize - calculated_height) / 2)
            imgWhite[height_gap:calculated_height + height_gap,:] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)
        cv2.putText(imgOutput, labels[index], (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)

        cv2.imshow("Image Crop", imgCrop)
        cv2.imshow("Image White", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)


