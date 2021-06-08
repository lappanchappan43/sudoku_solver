import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Remove tf warnings from console

import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

from solver import solver, print_sudoku

json_file = open('model/new_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
MODEL = model_from_json(loaded_model_json)
# load weights into new model
MODEL.load_weights("model/new_model.h5")
# MODEL = load_model('model/new_model.h5')

# img = cv2.imread('sudoku_images/2.jpeg')

def show_image(img):
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preprocess(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    process = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    return process

def contours(threshImg, img):
    cnts, _ = cv2.findContours(threshImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, cnts, -1, (255, 0, 0), 2)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015*perimeter, True)

        if len(approx) == 4:
            return approx
    #         cv2.drawContours(img, [cnts[0]], -1, (0, 255, 0), 2)

    # show_image(img)

def swap_order(point):
    # Get the (x1, y1, x2, y2) co-ordinates
    new_point = np.zeros((4, 1, 2), dtype=np.int32)

    sum = point.sum(1)
    diff = np.diff(point, axis=1)

    new_point[0], new_point[1], new_point[2], new_point[3] = point[np.argmin(sum)], point[np.argmin(diff)], \
                                                                point[np.argmax(diff)], point[np.argmax(sum)]
    
    return new_point

def reorder_grid(point):
    # Ordering the points as: top_left, top_right, bottom_left, bottom_right
    point = point.reshape((4, 2))
    new_point = swap_order(point=point)

    return new_point

def warp_image(img, cnts):
    # Warp Perspective
    point1 = np.float32(cnts)
    point2 = np.float32([[0, 0],[450, 0], [0, 450],[450, 450]])
    # Calculate perspective transform from 4 given points: (x1, y1), (x2, y1), (x1, y2), (x2, y2)
    new_warp_matrix = cv2.getPerspectiveTransform(point1, point2) 
    # Convert perspective transformed matrix with respect to the given image 
    warped_image = cv2.warpPerspective(img, new_warp_matrix, (450, 450))
    warped_image = cv2.cvtColor(warped_image,cv2.COLOR_BGR2GRAY)

    # show_image(warped_image)

    return warped_image

def split_box(img, SUDOKU_MATRIX):
    # Split image into row 
    rows = np.vsplit(img, SUDOKU_MATRIX)
    individual_grid = list()

    for row in rows:
        # Get all column for each row (for 9x9: total len(individual_grid) = 81)
        cols = np.hsplit(row, SUDOKU_MATRIX)
        [individual_grid.append(col) for col in cols]
        
    return individual_grid

def predict(individual_grid, THRESHOLD):
    # Predict each grid
    global MODEL
    # print(dir(MODEL))
    result = []
    for grid in individual_grid:
        grid_img = np.asarray(grid)
        # remove unwanted spaces from each grid
        grid_img = grid_img[5:grid_img.shape[0] - 5, 5:grid_img.shape[1] -5]
        # Resize image to (32, 32)
        grid_img = cv2.resize(grid_img, (32, 32))

        grid_img = grid_img/255 # Normalize image
        grid_img = grid_img.reshape(1, 32, 32, 1)

        # Predict the grid image
        prediction = MODEL.predict_classes(grid_img)
        pred = MODEL.predict(grid_img)
        prob = np.amax(pred)
        
        result.append(prediction[0]) if prob > THRESHOLD else result.append(0)

    return result

def to_matrix(numbers, n):
    # Convert the image 1D matrix to 2D matrix (Ex: [1, 2, 3, 4, 5, 6] -> [[1, 2, 3], [4, 5, 6]])
    return [numbers[i:i+n] for i in range(0, len(numbers), n)]

def display_number(numbers, img, SUDOKU_MATRIX):
    row, col = SUDOKU_MATRIX, SUDOKU_MATRIX
    width, height = int(img.shape[1]/row), int(img.shape[0]/col)
    
    # Put the numbers for each boxes into the image
    for r in range(0, row):
        for c in range(0, col):
            if numbers[r+(c*col)] != 0:
                cv2.putText(img, str(numbers[r+(c*col)]), (r*width+int(width/2)-9, int((c+0.75)*height)), 
                cv2.FONT_HERSHEY_SIMPLEX,  1, (250, 250, 250), 3, cv2.LINE_AA)
    return img

if __name__ == '__main__':
    img_name = input('Enter image name (Ex: image location (sudoku_images/1.jpg), enter 1.jpg): ') or '1.jpg'

    img = cv2.imread(f'sudoku_images/{img_name}')
    img = cv2.resize(img, (450, 450))

    THRESHOLD = float(input('Enter threshold value for digit recognition (Ex: if (80%) enter: 0.8): ') or '0.8')
    SUDOKU_MATRIX = int(input('Enter SUDOKU matrix (Ex: if 9x9 enter 9): ') or '9')

    show_image(img)
    threshImg = preprocess(img)
    cnts = contours(threshImg, img)

    # reshape_cnts = cnts.reshape((4, 2))
    cnts = reorder_grid(cnts)

    warp = warp_image(img, cnts)
    boxes = split_box(warp, SUDOKU_MATRIX)

    predictions = predict(boxes, THRESHOLD)

    blank_img = np.zeros((warp.shape[0], warp.shape[1], 3), np.uint8)

    # detect_img = display_number(predictions, blank_img)
    # show_image(detect_img)

    a = np.asarray(predictions)
    a = np.where(a>0, 0, 1)

    predictions = to_matrix(predictions, SUDOKU_MATRIX)
    solver(predictions, [SUDOKU_MATRIX, SUDOKU_MATRIX])
    print(predictions)
    # print_sudoku(predictions)
    solved = sum(predictions, [])

    b = solved*a

    blank_img = np.zeros((warp.shape[0], warp.shape[1], 3), np.uint8)
    # detect_img = display_number(b, blank_img)
    detect_img = display_number(solved, blank_img, SUDOKU_MATRIX)
    show_image(detect_img)