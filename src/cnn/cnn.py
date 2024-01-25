import time
from PIL import Image
from matplotlib import pyplot as plt
import numpy as cp
import cupy as cp2
import os
import pathlib
import multiprocessing as mp

class Kernal():
    # Popular kernals
    ridge_detection = cp.array([[0 , -1,  0],
                                [-1,  4, -1],
                                [0 , -1,  0]])
    
    edge_detection = cp.array([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]])
    
    depth_detection = cp.array([[-1,  0, 1],
                           [-1,  0, 1],
                           [-1,  0, 1]])
    
    
class ImageUtils():    
    def rgb_to_greyscale(photo):
        return photo.convert('L')
    
    
    def photo_to_matrix(photo):
        m = cp.asarray(photo)
        return m
    
    
    def expand_photo_matrix(photo_matrix, expansion_length):
        """Extend the photo matrix using "expansion" edge handling

        Args:
            photo_matrix (ndarray): matrix of the photo
            expansion_length (int): length of expansion of a corner
        """
        expanded_matrix = cp.zeros((photo_matrix.shape[0] + 2 * expansion_length, photo_matrix.shape[1] + 2 * expansion_length))
        expanded_matrix[expansion_length:expanded_matrix.shape[0] - expansion_length, 
                        expansion_length:expanded_matrix.shape[1] - expansion_length] = photo_matrix.copy()
        expanded_matrix = ImageUtils.__fill_corners(expansion_length, expanded_matrix)
        expanded_matrix = ImageUtils.__fill_edges(expansion_length, expanded_matrix)
        return expanded_matrix
    
    
    def __fill_corners(expansion_length, expanded_matrix):
        """fill in the corner of the expanded matrix
        
        Args:
            expansion_length (int): length of expansion of a corner
        """
        # top left corner
        new_matrix = expanded_matrix.copy()
        new_matrix[0:expansion_length, 0:expansion_length] = new_matrix[expansion_length, expansion_length]
        
        # bottom left corner
        new_matrix[new_matrix.shape[0] - expansion_length:new_matrix.shape[0], 
                   0:expansion_length] = new_matrix[new_matrix.shape[0] - expansion_length - 1, expansion_length]
        
        # top right corner
        new_matrix[0:expansion_length,
                   new_matrix.shape[1] - expansion_length:new_matrix.shape[1]] = new_matrix[expansion_length, 
                                                                                            new_matrix.shape[1] - expansion_length - 1]
                   
        # bottom right corner
        new_matrix[new_matrix.shape[0] - expansion_length:new_matrix.shape[0],
                   new_matrix.shape[1] - expansion_length:new_matrix.shape[1]] = new_matrix[new_matrix.shape[0] - expansion_length - 1, 
                                                                                            new_matrix.shape[1] - expansion_length - 1]
        
        return new_matrix
    
    
    def __fill_edges(expansion_length, expanded_matrix):
        """fill in the edges of the expanded matrix
        
        Args:
            expansion_length (int): length of expansion of a corner
        """
        # top edge
        new_matrix = expanded_matrix.copy()
        new_matrix = ImageUtils.__fill_horizontal_edge(expansion_length, new_matrix, "top")
        
        # bottom edge
        new_matrix = ImageUtils.__fill_horizontal_edge(expansion_length, new_matrix, "bottom")
        
        # left edge
        new_matrix = ImageUtils.__fill_vertical_edge(expansion_length, new_matrix, "left")
        
        # right edge
        new_matrix = ImageUtils.__fill_vertical_edge(expansion_length, new_matrix, "right")
        
        return new_matrix
    
    
    def __fill_horizontal_edge(expansion_length, expanded_matrix, edge):
        """fill in the horizontal edges of the expanded matrix
        
        Args:
            expansion_length (int): length of expansion of a corner
        """
        new_matrix = expanded_matrix.copy()
        
        if edge == "top":
            ref_row = expansion_length
            start_row = 0
        else:  # ref_row == "bottom"
            ref_row = new_matrix.shape[0] - expansion_length - 1
            start_row = new_matrix.shape[0] - expansion_length
            
        for i in range(expansion_length, expanded_matrix.shape[1] - expansion_length):
            new_matrix[start_row:start_row + expansion_length, i] = new_matrix[ref_row, i]
        
        return new_matrix
    
    
    def __fill_vertical_edge(expansion_length, expanded_matrix, edge):
        """fill in the vertical edges of the expanded matrix
        
        Args:
            expansion_length (int): length of expansion of a corner
        """
        new_matrix = expanded_matrix.copy()
        
        if edge == "left":
            ref_col = expansion_length
            start_col = 0
        else:  # ref_col == "right"
            ref_col = new_matrix.shape[1] - expansion_length - 1
            start_col = new_matrix.shape[1] - expansion_length
            
        for i in range(expansion_length, expanded_matrix.shape[0] - expansion_length):
            new_matrix[i, start_col:start_col + expansion_length] = new_matrix[i, ref_col]
        
        return new_matrix
    
    
class CNN:    
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
        
    
    def convolution(self):
        """Convolution of the photo matrix with the kernal
        """
        conv_mat = cp.zeros((self.extended_mat.shape[0] - self.kernal.shape[0] + 1, self.extended_mat.shape[1] - self.kernal.shape[1] + 1))

        for m in range(conv_mat.shape[0]):
            for n in range(conv_mat.shape[1]):
                conv_mat[m, n] = cp.sum(cp.multiply(self.extended_mat[m:m + self.kernal.shape[0], n:n + self.kernal.shape[1]], self.kernal))
                
        return conv_mat
    
    
    def package_convolution(self, extended_mat, m1, m2):
        """Convolution of the photo matrix with the kernal
        """
        package_mat = cp.zeros((m2 - m1, self.extended_mat.shape[1] - self.kernal.shape[1] + 1))
        
        for m in range(m1, m2):            
            for n in range(package_mat.shape[1]):
                package_mat[m, n] = cp.sum(cp.multiply(self.extended_mat[m:m + self.kernal.shape[0], n:n + self.kernal.shape[1]], self.kernal))
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
                m_end = self.extended_mat.shape[0]
            result_chunks.append(pool.apply_async(self.package_convolution, args=(self.extended_mat, process * package_size, m_end)))

        pool.close()
        pool.join()
        print(result_chunks[0].get())
        #for result in result_chunks:
        #    print(result.get())


### Drafts ###
image_path = os.path.abspath(os.path.join(pathlib.Path(__file__).parent.resolve(), '..', 'resource', 'woman8k.jpg'))
#cnn = CNN(image_path, Kernal.ridge_detection)
#print(cnn.mat.shape)
#img_plot = plt.imshow(cnn.mat.get(), cmap='gray')
#print(cnn.extended_mat.shape)
#img_plot = plt.imshow(cnn.conv_mat.get(), cmap='gray')
s = time.time()
cnn2 = CNN(image_path, Kernal.depth_detection)
e = time.time()
print("time taken: " + str(e - s))
#img_plot2 = plt.imshow(cnn2.conv_mat.get(), cmap='gray')

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