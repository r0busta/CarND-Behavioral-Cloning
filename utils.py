import numpy as np
import matplotlib.pyplot as plt
import math


def get_img_path(folder, src):
    filename = src.split('/')[-1]
    return folder + filename


def flip_img(img):
    return np.fliplr(img)


def to_gray(img):
    return np.expand_dims(np.dot(img, [0.299, 0.587, 0.114]), axis=4)


def mean_normalize(img):
    return (img - 128)/128


def normalize(img):
    return mean_normalize(to_gray(img))


def plot_images(images):
    fig = plt.figure(figsize=(15, 10))
    ncols = 5
    nrows = math.ceil(len(images) / ncols)
    for i in range(len(images)):
        fig.add_subplot(nrows, ncols, i+1)
        img = images[i].squeeze()
        plt.imshow(img, cmap='gray')

    plt.show()
