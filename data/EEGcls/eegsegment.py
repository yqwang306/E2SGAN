from utils.eeg_tools import load_from_ndarray, IF_to_eeg
import cv2
import numpy as np


class EEGSegment:

    def __init__(self, filename, isIF=False, normalizer=None):
        self.data = load_from_ndarray(filename)
        if isIF:
            self.data = IF_to_eeg(self.data, normalizer)

    def size(self):
        return self.data.shape

    def crop(self, left, upper, right, lower):
        print("Before cropping: " + str(self.data))
        print("Origin shape: " + str(self.size()))
        self.data = self.data[upper:lower, left:right]
        print("Cropped: " + str(self.data))
        return self

    def resize(self, w, h, method=cv2.INTER_CUBIC): 
        new_size = (h, w)
        self.data = cv2.resize(self.data, dsize=new_size, interpolation=method)
        return self

    def transpose(self, method):
        self.data = np.flip(self.data, 1)
        return self

    def get_data(self):
        return self.data

    def set_data(self, value):
        self.data = value