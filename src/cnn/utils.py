import numpy as cp
import cupy as cp2

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