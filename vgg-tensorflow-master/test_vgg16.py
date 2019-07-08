import numpy as np
import tensorflow as tf
from skimage import io
import matplotlib.pyplot as plt

import vgg16
import utils

import os
from PIL import Image

img1 = utils.load_image("./test_data/cyber_1.jpg")
img2 = utils.load_image("./test_data/n-cyber.jpg")

batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))

batch = np.concatenate((batch1, batch2), 0)

# with tf.Session(\
#    config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device('/cpu:0'):
    with tf.Session() as sess:
        images = tf.placeholder("float", [2, 224, 224, 3])
        feed_dict = {images: batch}

        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(images)
        feature = sess.run(vgg.fc8, feed_dict=feed_dict)# 需要提取哪一层特征，就在这里做修改，比如fc6，只需要把vgg.fc7修改为vgg.fc6
        print(feature)
        print(feature.shape)
        plt.imshow(feature)
        io.imshow(feature)
