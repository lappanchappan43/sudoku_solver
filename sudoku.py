import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Remove tf warnings from console

import cv2
import numpy as np
from keras.models import load_model

from solver import solver, print_sudoku

MODEL = load_model('model/model.h5')

# img = cv2.imread('sudoku_images/2.jpeg')
img = cv2.imread('sudoku_images/1.jpg')
img = cv2.resize(img, (450, 450))

def show_image(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preprocess(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    process = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # process = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)

    # show_image(process)

    # Dilate
    # kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], dtype=np.uint8)
    # process = cv2.dilate(process, kernel)
    # process = cv2.morphologyEx(process, cv2.MORPH_OPEN, kernel)

    # show_image(process)

    return process

def contours(threshImg, img):
    cnts, _ = cv2.findContours(threshImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015*perimeter, True)

        if len(approx) == 4:
            return approx
    #         cv2.drawContours(img, [cnts[0]], -1, (0, 255, 0), 2)

    # show_image(img)


def reorder(myPoints):
    # Ordering the points as: top_left, top_right, bottom_left, bottom_right
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def warp_image(img, cnts):
    pts1 = np.float32(cnts) # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0],[450, 0], [0, 450],[450, 450]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (450, 450))
    imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

    return imgWarpColored

def split_box(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for row in rows:
        cols = np.hsplit(row, 9)
        for col in cols:
            boxes.append(col)
    return boxes

def predict(boxes):
    global MODEL
    result = []
    for box in boxes:
        img = np.asarray(box)
        img = img[5:img.shape[0] - 5, 5:img.shape[1] -5]
        img = cv2.resize(img, (32, 32))

        img = img/255
        img = img.reshape(1, 32, 32, 1)

        classIndex = MODEL.predict_classes(img)
        pred = MODEL.predict(img)
        prob = np.amax(pred)

        if prob > 0.8:
            result.append(classIndex[0])
        else:
            result.append(0)
    
    return result

def to_matrix(numbers, n):
    return [numbers[i:i+n] for i in range(0, len(numbers), n)]

def display_number(numbers, img):
    width = int(img.shape[1]/9)
    height = int(img.shape[0]/9)
    for x in range(0, 9):
        for y in range(0, 9):
            if numbers[(y*9)+x] != 0:
                cv2.putText(img, str(numbers[(y*9)+x]),
                (x*width+int(width/2)-10, int((y+0.8)*height)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                2, (0, 255, 0), 2, cv2.LINE_AA)
    return img

threshImg = preprocess(img)
cnts = contours(threshImg, img)
cnts = reorder(cnts)

warp = warp_image(img, cnts)
boxes = split_box(warp)

predictions = predict(boxes)

blank_img = np.zeros((warp.shape[0], warp.shape[1], 3), np.uint8)

detect_img = display_number(predictions, blank_img)
show_image(detect_img)

predictions = to_matrix(predictions, 9)
solver(predictions)
print(predictions)
# print_sudoku(predictions)
solved = sum(predictions, [])
detect_img = display_number(solved, blank_img)
show_image(detect_img)