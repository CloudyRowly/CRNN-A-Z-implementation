from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os
import pathlib

class Kernal():
    # Popular kernals
    ridge_detection = np.array([[0 , -1,  0],
                                [-1,  4, -1],
                                [0 , -1,  0]])
    
    edge_detection = np.array([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]])
    
    
class CNN:
    def __init__(self, photo_path, kernal, stride=1, padding=0):
        self.photo_path = photo_path
        self.kernal = kernal
        self.stride = stride
        self.padding = padding
    
    
    def photo_to_matrix(self):
        mat = np.asarray(Image.open(self.photo_path))
        return mat
    
    
    def expand_photo_matrix(self, photo_matrix):
        expanding_length = (self.kernal.shape[0] - 1) // 2
        pass
    

image_path = os.path.abspath(os.path.join(pathlib.Path(__file__).parent.resolve(), '..', 'resource', 'stinkbug.png'))
cnn = CNN(image_path, Kernal.ridge_detection)
print(cnn.photo_to_matrix())
img_plot = plt.imshow(cnn.photo_to_matrix())

"""
a1 = np.array([[1,2],[4,5],[7,8]])
a2 = np.array([[3,4],[6,7],[9,10]])
a3 = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a1 * a2)
"""
