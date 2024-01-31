from PIL import Image
import numpy as cp
import multiprocessing as mp

from utils import ImageUtils
from kernal import Kernal

class CNN:
    def __init__(self, photo_path, photo_height, available_core, stride = 1):
        self.photo_path = photo_path
        self.photo_height = photo_height
        self.kernal_shape = (3, 3)
        self.filter_number = 1
        self.stride = stride
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
        
        # initialize kernals
        self.kernals = cp.zeros((self.filter_number, self.kernal_shape[0], self.kernal_shape[1]))
        for i in range(filter_number):
            self.kernals[i, :, :] = Kernal(self.kernal_shape[0], self.kernal_shape[1]).get_kernal()
            
        # expand the image matrix
        self.expansion_length = (kernal_shape[0] - 1) // 2
        self.expanded_feature_maps = cp.zeros((self.feature_maps.shape[0] + 2 * self.expansion_length, 
                                               self.feature_maps.shape[1] + 2 * self.expansion_length,
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
        temp_feature_maps = cp.zeros((self.expanded_feature_maps.shape[0] - kernal_shape[0] + 1, 
                                      self.expanded_feature_maps.shape[1] - kernal_shape[1] + 1,
                                      self.filter_number))
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
        
        
    def relu_(self, feature_map):
        feature_map = cp.maximum(feature_map, 0)
        return feature_map
    
    
    def relu(self):
        """Rectified linear unit
        """
        pool = mp.Pool(self.available_core)
        feature_map_result = []
        
        for i in range(self.filter_number):
            feature_map_result.append(pool.apply_async(self.relu_, args=(self.feature_maps[:, :, i],)))
        pool.close()
        pool.join()
        for i in range(self.filter_number):
            self.feature_maps[:, :, i] = feature_map_result[i].get()
        
        
    def max_pooling_(self, feature_map, window_size=(2, 2), stride=(2, 2)):
        w_pool = window_size[0]
        h_pool = window_size[1]
        pool_mat = cp.zeros((self.feature_maps.shape[0] // stride[1], self.feature_maps.shape[1] // stride[0]))
        for m in range(pool_mat.shape[0]):
            for n in range(pool_mat.shape[1]):
                pool_mat[m, n] = cp.max(feature_map[m * stride[1]:m * stride[1] + h_pool, n * stride[0]:n * stride[0] + w_pool])
        return pool_mat
    
        
    def max_pooling(self, window_size=(2, 2), stride=(2, 2)):
        """Max pooling of the photo matrix with the window size and stride (width, height)
        """
        temp_feature_maps = cp.zeros((self.feature_maps.shape[0] // stride[1], self.feature_maps.shape[1] // stride[0], self.feature_maps.shape[-1]))
        pool = mp.Pool(self.available_core)
        feature_map_result = []
        for i in range(self.filter_number):
            feature_map_result.append(pool.apply_async(self.max_pooling_, args=(self.feature_maps[:, :, i], window_size, stride)))
        pool.close()
        pool.join()
        for i in range(self.filter_number):
            temp_feature_maps[:, :, i] = feature_map_result[i].get()
        
        self.feature_maps = temp_feature_maps.copy()
        
        
    def batch_normalization_(self, feature_map, mean, sd):
        result = cp.zeros(feature_map.shape)
        for row in range(feature_map.shape[0]):
            for col in range(feature_map.shape[1]):
                result[row, col] = (feature_map[row, col] - mean) / sd
        return result
        
        
    def batch_normalization(self):
        flattented = cp.ndarray.flatten(self.feature_maps.copy())
        mean = cp.mean(flattented)
        sd = cp.std(flattented)  # standard deviation
        
        pool = mp.Pool(self.available_core)
        feature_map_result = []
        for map in range(self.filter_number):
            feature_map_result.append(pool.apply_async(self.batch_normalization_, args=(self.feature_maps[:, :, map], mean, sd)))
        pool.close()
        pool.join()
        for i in range(self.filter_number):
            self.feature_maps[:, :, i] = feature_map_result[i].get()
            
        
    def get_feature_map(self):
        return self.feature_maps
            
        
