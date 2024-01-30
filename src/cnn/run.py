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
                ax[y, x].set_title("map #" + str(iteration))
        
        plt.savefig(saveas, bbox_inches="tight")
        plt.close(fig)
    

def __main__():
    image_path = os.path.abspath(os.path.join(pathlib.Path(__file__).parent.resolve(), '..', 'resource', 'stinkbug.png'))

    s = time.time()
    cnn = CNN(image_path, 32, 8)
    cnn.convolution(32)
    e = time.time()
    print("time taken: " + str(e - s))
    feature_map = cnn.get_feature_map()
    run.render_features(feature_map, "L1.png", (4, 8))
    time.sleep(10)


if __name__ == '__main__':
    __main__()