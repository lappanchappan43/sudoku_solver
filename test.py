import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Remove tf warnings from console

import cv2
import numpy as np

from keras.models import load_model
from train import normalize

model = load_model('model/model.h5')
for im in os.listdir('test'):
    img = cv2.imread(os.path.join('test', im))

    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))

    img = normalize(img)

    img = img.reshape(1, 32, 32, 1)

    classIndex = model.predict_classes(img)
    pred = model.predict(img)
    prob = np.amax(pred)
    
    print(classIndex, prob, im)
