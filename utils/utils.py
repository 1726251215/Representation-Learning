import matplotlib.pyplot as plt
import math
import torch
import cv2
import numpy as np


def show_image(image, out, label):
    image *= 255.0
    image = image.cpu().numpy().astype(dtype=int)
    plt.figure()
    num = image.shape[0]
    row = np.floor(math.sqrt(num))
    col = np.ceil(num / row)
    for i, sample in enumerate(image):
        plt.subplot(row, col, i+1)
        sample = sample[0]
        plt.imshow(sample)

    print("output: ", out.argmax(axis=1))
    print("label: ", label)
    plt.show()







