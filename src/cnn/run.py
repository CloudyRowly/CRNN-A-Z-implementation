import numpy as cp 
from cnn import CNN
import time
import matplotlib.pyplot as plt
import os
import pathlib

class run():
    def render_features(feature_map, saveas, subplot_shape=(1, 1)):
        fig, ax = plt.subplots(subplot_shape[0], subplot_shape[1])
        for y in range(subplot_shape[0]):
            for x in range(subplot_shape[1]):
                iteration = y * subplot_shape[1] + x
                ax[y, x].imshow(feature_map[:, :, iteration]).set_cmap("gray")
                ax[y, x].get_xaxis().set_ticks([])
                ax[y, x].get_yaxis().set_ticks([])
                # ax[y, x].set_title("map #" + str(iteration))
        
        plt.savefig(saveas, bbox_inches="tight")
        plt.close(fig)
    

def __main__():
    image_path = os.path.abspath(os.path.join(pathlib.Path(__file__).parent.resolve(), '..', 'resource', 'stinkbug.png'))

    s = time.time()
    cnn = CNN(image_path, 32, 8)
    
    cnn.convolution(64)
    feature_map_1 = cnn.get_feature_map()
    cnn.relu()
    feature_map_1_relu = cnn.get_feature_map()
    
    cnn.max_pooling()
    feature_map_2 = cnn.get_feature_map()
    
    cnn.convolution(128)
    feature_map_3 = cnn.get_feature_map()
    cnn.relu()
    feature_map_3_relu = cnn.get_feature_map()
    
    cnn.max_pooling()
    feature_map_4 = cnn.get_feature_map()
    
    cnn.convolution(256)
    feature_map_5 = cnn.get_feature_map()
    cnn.relu()
    feature_map_5_relu = cnn.get_feature_map()
    
    cnn.convolution(256)
    feature_map_6 = cnn.get_feature_map()
    cnn.relu()
    feature_map_6_relu = cnn.get_feature_map()
    
    cnn.max_pooling(window_size=(1, 2), stride = (1, 2))
    feature_map_7 = cnn.get_feature_map()
    
    cnn.convolution(512)
    cnn.relu()
    feature_map_8 = cnn.get_feature_map()
    
    cnn.batch_normalization()
    
    cnn.convolution(512)
    cnn.relu()
    feature_map_9 = cnn.get_feature_map()
    
    cnn.batch_normalization()
    
    cnn.max_pooling(window_size=(1, 2), stride = (1, 2))
    feature_map_10 = cnn.get_feature_map()
    
    cnn.convolution(512, kernal_shape=(2, 2), padding=0)
    cnn.relu()
    feature_map_11 = cnn.get_feature_map()
    
    cnn.dense()
    feature_sequence = cnn.get_feature_map()
        
    e = time.time()
    print("time taken: " + str(e - s))
    '''
    run.render_features(feature_map_1, "L1.png", (8,8))
    run.render_features(feature_map_1_relu, "L1_relu.png", (8,8))
    run.render_features(feature_map_2, "L2.png", (8,8))
    run.render_features(feature_map_3, "L3.png", (16, 8))
    run.render_features(feature_map_3_relu, "L3_relu.png", (16, 8))
    run.render_features(feature_map_4, "L4.png", (16, 8))
    run.render_features(feature_map_5, "L5.png", (16, 16))
    run.render_features(feature_map_5_relu, "L5_relu.png", (16, 16))
    run.render_features(feature_map_6_relu, "L6_relu.png", (16, 16))
    run.render_features(feature_map_6, "L6.png", (16, 16))
    run.render_features(feature_map_7, "L7.png", (16, 16))
    run.render_features(feature_map_8, "L8.png", (16, 32))
    run.render_features(feature_map_9, "L9.png", (16, 32))
    run.render_features(feature_map_10, "L10.png", (16, 32))
    run.render_features(feature_map_11, "L11.png", (16, 32))
    '''
    
    
if __name__ == '__main__':
    __main__()