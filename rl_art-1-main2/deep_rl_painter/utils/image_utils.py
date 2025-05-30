# needs to be reviewed - add proper comments 
# Not even being used in the codebase
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch

def load_image(path, size=(256, 256)):
    """
    Loads an image in grayscale, resizes it, and normalizes pixel values.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    # img = img / 255.0  # normalize to [0,1]
    return img

def preprocess_image(image):
    """
    Converts a numpy image to a normalized torch tensor.
    """
    if len(image.shape) == 2:
        image = image[np.newaxis, :, :]  # add channel dim if needed
    image = torch.FloatTensor(image / 255.0)
    return image
