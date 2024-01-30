import time
from PIL import Image
from matplotlib import pyplot as plt
import numpy as cp
import cupy as cp2
import time
from PIL import Image
from matplotlib import pyplot as plt
import numpy as cp
import cupy as cp2
import os
import pathlib
import multiprocessing as mp

from utils import ImageUtils
from kernal import Kernal

class CNN:
    def __init__(self, photo_path, photo_height, available_core, stride = 1, padding=1):
        self.photo_path = photo_path
        self.photo_height = photo_height
        self.kernal_shape = (3, 3)
        self.filter_number = 1
        self.stride = stride
        self.padding = padding
        self.available_core = available_core
        
        # image pre-processing and resize
        self.photo = Image.open(self.photo_path)
        self.photo = ImageUtils.rgb_to_greyscale(self.photo)
        self.photo = ImageUtils.resize_by_height(self.photo, photo_height)
        self.photo_mat = ImageUtils.photo_to_matrix(self.photo)
        self.feature_maps = cp.zeros((self.photo_mat.shape[0], self.photo_mat.shape[1], self.filter_number))
        self.feature_maps[:, :, 0] = self.photo_mat
        
    
    def conv(self, feature_map, kernal):
            """Convolution of the photo matrix with the kernal
            """
            conv_mat = cp.zeros((feature_map.shape[0] - kernal.shape[0] + 1, feature_map.shape[1] - kernal.shape[1] + 1))

            for m in range(conv_mat.shape[0]):
                for n in range(conv_mat.shape[1]):
                    conv_mat[m, n] = cp.sum(cp.multiply(feature_map[m:m + kernal.shape[0], n:n + kernal.shape[1]], kernal))
                
            return conv_mat


    def convolution(self, filter_number, kernal_shape=(3, 3)):
        filter_multiplier = filter_number // self.filter_number
        self.filter_number = filter_number
        self.kernal_shape = kernal_shape
        temp_feature_maps = cp.zeros((self.feature_maps.shape[0], self.feature_maps.shape[1], filter_number))
        
        # initialize kernals
        self.kernals = cp.zeros((self.filter_number, self.kernal_shape[0], self.kernal_shape[1]))
        for i in range(filter_number):
            self.kernals[i, :, :] = Kernal(self.kernal_shape[0], self.kernal_shape[1]).get_kernal()
        print(self.kernals)
            
        # expand the image matrix
        self.expansion_length = (kernal_shape[0] - 1) // 2
        self.expanded_feature_maps = cp.zeros((self.photo_mat.shape[0] + 2 * self.expansion_length, 
                                               self.photo_mat.shape[1] + 2 * self.expansion_length,
                                               self.filter_number))
        if self.padding != 0:
            pool = mp.Pool(self.available_core)
            expanded_mat_result = []
            for i in range(self.feature_maps.shape[2]):
                expanded_mat_result.append(pool.apply_async(ImageUtils.expand_photo_matrix, args=(self.feature_maps[:, :, i], self.expansion_length)))
            pool.close()
            pool.join()
            for i in range(self.feature_maps.shape[2]):
                self.expanded_feature_maps[:, :, i] = expanded_mat_result[i].get()
        else:
            # copy the original feature map to the expanded feature map in case padding = 0
            # for next step. Still means no padding, just a small hack as mapping memory 
            # like this shortens the logic for conv step.
            self.expanded_feature_maps = self.feature_maps.copy()
        
        # convolution
        pool = mp.Pool(self.available_core)
        feature_map_result = []
        j = 0
        for i in range(filter_number):
            # pacing the change of feature_map according to the filter_number
            # e.g. if input feature map = 32, output feature map = 64, then change the feature map every 2 iterations
            if i % filter_multiplier == 0 and i != 0:
                j += 1
            feature_map_result.append(pool.apply_async(self.conv, args=(self.expanded_feature_maps[:, :, j], self.kernals[i, :, :])))
        pool.close()
        pool.join()
        for i in range(filter_number):
            temp_feature_maps[:, :, i] = feature_map_result[i].get()
        
        self.feature_maps = temp_feature_maps.copy()
        
        
    def get_feature_map(self):
        return self.feature_maps
            
        
