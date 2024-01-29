import time
from PIL import Image
from matplotlib import pyplot as plt
import numpy as cp
import cupy as cp2
import os
import pathlib
import multiprocessing as mp
from numba import njit

from utils import ImageUtils

class Kernal():
    # Predefined popular kernals
    ridge_detection = cp.array([[0 , -1,  0],
                                [-1,  4, -1],
                                [0 , -1,  0]])
    
    edge_detection = cp.array([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]])
    
    depth_detection = cp.array([[-1,  0, 1],
                           [-1,  0, 1],
                           [-1,  0, 1]])
    
    
    def __init__(self, size_m = 3, size_n = 3):
        self.shape = (size_m, size_n)
        self.kernal = cp.random.randn(self.size_m, self.size_n)
        
    
class Convolution_filter():    
    def __init__(self, photo_path, kernal, padding=1):
        self.photo_path = photo_path
        self.photo = Image.open(self.photo_path)
        self.kernal = kernal
        self.padding = padding
        
        self.photo = ImageUtils.rgb_to_greyscale(self.photo)
        self.mat = ImageUtils.photo_to_matrix(self.photo)
        if self.padding != 0:
            expanding_length = (self.kernal.shape[0] - 1) // 2
            self.extended_mat = ImageUtils.expand_photo_matrix(self.mat, expanding_length)
        # self.conv_mat = self.convolution_multiprocessing(8)
        self.conv_mat = self.convolution()
        
    
    @njit
    def convolution(self):
        """Convolution of the photo matrix with the kernal
        """
        conv_mat = cp.zeros((self.extended_mat.shape[0] - self.kernal.shape[0] + 1, self.extended_mat.shape[1] - self.kernal.shape[1] + 1))

        for m in range(conv_mat.shape[0]):
            for n in range(conv_mat.shape[1]):
                conv_mat[m, n] = cp.sum(cp.multiply(self.extended_mat[m:m + self.kernal.shape[0], n:n + self.kernal.shape[1]], self.kernal))
                
        return conv_mat
    
    
    def relu(self):
        """Rectified Linear Unit
        """
        return cp.maximum(self.conv_mat, 0)
    
    
    def max_pooling(self, h_pool, w_pool):
        """Max pooling of the photo matrix with the kernal
        """
        pool_mat = cp.zeros((self.conv_mat.shape[0] // h_pool, self.conv_mat.shape[1] // w_pool))
        
        for m in range(pool_mat.shape[0]):
            for n in range(pool_mat.shape[1]):
                pool_mat[m, n] = cp.max(self.conv_mat[m * h_pool:m * h_pool + h_pool, n * w_pool:n * w_pool + w_pool])
                
        return pool_mat
    
    
    def package_convolution(self, m1, m2):
        """Convolution of the photo matrix with the kernal
        """
        package_mat = cp.zeros((m2 - m1, self.extended_mat.shape[1] - self.kernal.shape[1] + 1))
        
        for m in range(m2 - m1):            
            for n in range(package_mat.shape[1]):
                package_mat[m, n] = cp.sum(cp.multiply(self.extended_mat[m + m1:m + m1 + self.kernal.shape[0], n:n + self.kernal.shape[1]], self.kernal))
        return package_mat
    
    
    def convolution_multiprocessing(self, core_available):
        """Convolution of the photo matrix with the kernal using multiprocessing
        """
        package_size = 256  # minimum package size for optimal performance
        package_number = self.extended_mat.shape[0] // package_size
        if package_number > core_available:
            package_size = self.extended_mat.shape[0] // core_available
            package_number = core_available
        
        pool = mp.Pool(core_available)
        result_chunks = []
        for process in range(package_number):
            m_end = (process + 1) * package_size
            if process == package_number - 1:
                m_end = self.mat.shape[0]
            result_chunks.append(pool.apply_async(self.package_convolution, args=(process * package_size, m_end)))

        pool.close()
        pool.join()
        self.conv_mat = result_chunks[0].get()
        if len(result_chunks) > 1:
            for result in result_chunks[1:]:
                self.conv_mat = cp.vstack((self.conv_mat, result.get()))
        return self.conv_mat
    
    
#class CNN:
    


### Drafts ###
image_path = os.path.abspath(os.path.join(pathlib.Path(__file__).parent.resolve(), '..', 'resource', 'stinkbug.png'))
#cnn = CNN(image_path, Kernal.ridge_detection)
#print(cnn.mat.shape)
#img_plot = plt.imshow(cnn.mat.get(), cmap='gray')
#print(cnn.extended_mat.shape)
#img_plot = plt.imshow(cnn.conv_mat.get(), cmap='gray')
s = time.time()
cnn2 = Convolution_filter(image_path, Kernal.depth_detection)
e = time.time()
print("time taken: " + str(e - s))
img_plot2 = plt.imshow(cnn2.conv_mat, cmap='gray')

a1 = cp.array([[1,2],
               [3,4],
               [5,6],
               [7,8]])
a2 = cp.zeros((4,4))
#a2[1:3, 1:3] = a1
#print(a2)
#print(a1.shape)

"""
a1 = np.array([[1,2],[4,5],[7,8]])
a2 = np.array([[3,4],[6,7],[9,10]])
a3 = np.array([[1,2,3],[4,5,6],[7,8,9]])
ai = a1.shape
print(ai)

"""