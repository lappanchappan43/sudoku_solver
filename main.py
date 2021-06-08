from flask import Flask, render_template, request, jsonify
import io
import numpy as np
import cv2
from PIL import Image
import base64

from sudoku import *

app = Flask(__name__)

def convert_image_base64(image):
    detect_img = Image.fromarray(image.astype("uint8"))
    rawBytes = io.BytesIO()
    detect_img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())

    return img_base64

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    img_file = request.files['file'].read()
    npimg = np.fromstring(img_file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    THRESHOLD = 0.8
    SUDOKU_MATRIX = 9

    threshImg = preprocess(img)
    cnts = contours(threshImg, img)
    cnts = reorder_grid(cnts)

    warp = warp_image(img, cnts)
    boxes = split_box(warp, SUDOKU_MATRIX)

    predictions = predict(boxes, THRESHOLD)

    blank_img = np.zeros((warp.shape[0], warp.shape[1], 3), np.uint8)

    a = np.asarray(predictions)
    a = np.where(a>0, 0, 1)

    predictions = to_matrix(predictions, SUDOKU_MATRIX)
    solver(predictions, [SUDOKU_MATRIX, SUDOKU_MATRIX])
    # print_sudoku(predictions)
    solved = sum(predictions, [])

    blank_img = np.zeros((warp.shape[0], warp.shape[1], 3), np.uint8)
    # detect_img = display_number(b, blank_img)
    detect_img = display_number(solved, blank_img, SUDOKU_MATRIX)
    
    solved_image = convert_image_base64(detect_img)
    wrap_image = convert_image_base64(warp)
    
    return jsonify({'status':'succces', 'processed': str(wrap_image), 'solved':str(solved_image)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6006, debug=True)