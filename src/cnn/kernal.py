import numpy as cp

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
        self.kernal = cp.random.randn(size_m, size_n)
        
    def get_kernal(self):
        return self.kernal