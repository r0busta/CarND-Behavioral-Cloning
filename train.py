import argparse
import os.path
import csv
from scipy import ndimage
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Conv2D, Cropping2D, Lambda
from sklearn.utils import shuffle
from utils import get_img_path, flip_img, normalize, plot_images


def read_logs(logs):
    samples = []
    for log in logs:
        with open(log) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                samples.append(line)
    return samples


def load_samples(samples):
    images = []
    steering_angles = []

    for sample in samples:
        steering_center = float(sample[3])

        # create adjusted steering measurements for the side camera images
        correction = 0.2
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        img_center = normalize(ndimage.imread(get_img_path(IMG_FOLDER, sample[0])))
        img_left = normalize(ndimage.imread(get_img_path(IMG_FOLDER, sample[1])))
        img_right = normalize(ndimage.imread(get_img_path(IMG_FOLDER, sample[2])))

        # add images and angles to data set
        images.extend([img_center, img_left, img_right])
        steering_angles.extend([steering_center, steering_left, steering_right])

        # add flipped version of the data point
        images.extend([flip_img(img_center), flip_img(img_left), flip_img(img_right)])
        steering_angles.extend([-steering_center, -steering_left, -steering_right])

    return images, steering_angles


def train(X_train, y_train):
    if os.path.exists('model.h5'):
        model = load_model('model.h5')
    else:
        model = Sequential()

        model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
        model.add(Lambda(lambda x: (x / 255.0) - 0.5))
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
    model.fit(X_train, y_train, shuffle=True, epochs=10)

    model.save('model.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train behavioral cloning convolution neural network model')
    parser.add_argument(
        'samples',
        type=str,
        help='Path to the simulator recordings.'
    )
    parser.add_argument(
        '-l', '--logs',
        action='append',
        help='Path(s) to the simulator sample logs.',
        required=True
    )
    parser.add_argument(
        '-n',
        type=int,
        help='Number of training steps.'
    )
    args = parser.parse_args()

    FOLDER = args.samples
    IMG_FOLDER = FOLDER + '/IMG/'

    samples = read_logs(args.logs)
    samples = shuffle(samples)

    num_samples = len(samples)
    STEPS = args.n or 5
    BATCH_SIZE = int(num_samples/STEPS)
    for offset in range(0, num_samples, BATCH_SIZE):
        end = offset+BATCH_SIZE
        print("Running training step {} from {}".format(int(offset/BATCH_SIZE) + 1, STEPS))
        print("Simulator samples {} - {} from {}".format(offset, end, num_samples))

        images, steering_angles = load_samples(samples[offset:end])

        X_train = np.array(images)
        y_train = np.array(steering_angles)
        X_train, y_train = shuffle(X_train, y_train)

        train(X_train, y_train)

    exit()
