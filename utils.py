import numpy as np
import matplotlib.pyplot as plt
import math


def get_img_path(folder, src):
    filename = src.split('/')[-1]
    return folder + filename


def flip_img(img):
    return np.fliplr(img)


def normalize(img):
    """
    For the implemented model, normalization is not required. Keeping this function empty.
    """
    return img


def plot_images(images):
    fig = plt.figure(figsize=(15, 10))
    ncols = 5
    nrows = math.ceil(len(images) / ncols)
    for i in range(len(images)):
        fig.add_subplot(nrows, ncols, i+1)
        img = images[i].squeeze()
        plt.imshow(img, cmap='gray')

    plt.show()
