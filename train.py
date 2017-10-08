import os.path
import csv
from scipy import ndimage
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Lambda, Cropping2D


FOLDER = '../data'
lines = []
with open(FOLDER + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
steering_angles = []


def get_img_path(folder, src):
    filename = src.split('/')[-1]
    return folder + filename


def flip_img(img):
    return np.fliplr(img)


IMG_FOLDER = FOLDER + '/IMG/'
for line in lines:
    steering_center = float(line[3])

    # create adjusted steering measurements for the side camera images
    correction = 0.2  # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    # read in images from center, left and right cameras
    img_center = ndimage.imread(get_img_path(IMG_FOLDER, line[0]))
    img_left = ndimage.imread(get_img_path(IMG_FOLDER, line[1]))
    img_right = ndimage.imread(get_img_path(IMG_FOLDER, line[2]))

    # add images and angles to data set
    images.extend([img_center, img_left, img_right])
    steering_angles.extend([steering_center, steering_left, steering_right])

    # add flipped version of the data point
    images.extend([flip_img(img_center), flip_img(img_left), flip_img(img_right)])
    steering_angles.extend([-steering_center, -steering_left, -steering_right])

X_train = np.array(images)
y_train = np.array(steering_angles)

if os.path.exists('model.h5'):
    model = load_model('model.h5')
else:
    model = Sequential()

    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

model.save('model.h5')
exit()
