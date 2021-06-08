import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Remove tf warnings from console

import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


DATASET_PATH = os.path.join(os.getcwd(), 'dataset')
CLASSES = os.listdir(DATASET_PATH)

def get_image_class() -> list:
    images = []
    class_map = []
    for x in range(0, len(CLASSES)):
        img_class_path = os.path.join(DATASET_PATH, str(x))
        for img_name in os.listdir(img_class_path):
            img_path = os.path.join(img_class_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (32, 32))
            images.append(img)
            class_map.append(x)
        
    return images, class_map

def normalize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    img = gray/255
    return img

def split_dataset(images, class_map):
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(images, class_map, shuffle=True, test_size=0.2)
    X_TRAIN, X_VALIDATION, Y_TRAIN, Y_VALIDATION = train_test_split(X_TRAIN, Y_TRAIN, test_size=0.2)

    X_TRAIN = np.array(list(map(normalize, X_TRAIN)))
    X_VALIDATION = np.array(list(map(normalize, X_VALIDATION)))
    X_TEST = np.array(list(map(normalize, X_TEST)))

    X_TRAIN = X_TRAIN.reshape(X_TRAIN.shape[0], X_TRAIN.shape[1], X_TRAIN.shape[2], 1)
    X_VALIDATION = X_VALIDATION.reshape(X_VALIDATION.shape[0], X_VALIDATION.shape[1], X_VALIDATION.shape[2], 1)
    X_TEST = X_TEST.reshape(X_TEST.shape[0], X_TEST.shape[1], X_TEST.shape[2], 1)    

    return X_TRAIN, X_VALIDATION, X_TEST, Y_TRAIN, Y_VALIDATION, Y_TEST

def model():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), activation='relu', input_shape=(32, 32, 1)))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

def show_graph(history, graph_type, val_type, title):
    plt.figure(1)
    plt.plot(history.history[graph_type])
    plt.plot(history.history[val_type])
    plt.legend(['training','validation'])
    plt.title(title)
    plt.xlabel('epoch')
    plt.show()

def plot_graph(history):
    show_graph(history=history, graph_type='loss', val_type='val_loss', title='Loss')
    show_graph(history=history, graph_type='accuracy', val_type='val_accuracy', title='Accuracy')

if __name__ == '__main__':
    images, class_map = get_image_class()
    images, class_map = np.array(images), np.array(class_map)

    X_TRAIN, X_VALIDATION, X_TEST, Y_TRAIN, Y_VALIDATION, Y_TEST = split_dataset(images, class_map)

    datagen = ImageDataGenerator(width_shift_range=0.1, zoom_range=0.2, height_shift_range=0.1, 
                                shear_range=0.1, rotation_range=10)
    datagen.fit(X_TRAIN)

    Y_TRAIN = to_categorical(Y_TRAIN, num_classes=10)
    Y_VALIDATION = to_categorical(Y_VALIDATION, num_classes=10)
    Y_TEST = to_categorical(Y_TEST, num_classes=10)
    
    model = model()
    # model = model.fit_generator(datagen.flow(x=X_TRAIN, y=Y_TRAIN, batch_size=50), steps_per_epoch=2000, epochs=10,
    #                     validation_data=(X_VALIDATION, Y_VALIDATION), shuffle=True)

    history = model.fit_generator(datagen.flow(x=X_TRAIN, y=Y_TRAIN), epochs=10,
                        validation_data=(X_VALIDATION, Y_VALIDATION), shuffle=True)

    # plot_graph(history)

    score = model.evaluate(X_TEST,Y_TEST,verbose=0)
    print('Test Score = ',score[0])
    print('Test Accuracy =', score[1])

    model_json = model.to_json()
    with open("model/new_model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights('model/new_model.h5')

    

    
    


