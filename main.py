import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

from helpers import load_img, vgg_layers, StyleContentModel, train_step, tensor_to_image, imshow

# variables
epochs = 3
steps_per_epoch = 3

style_weight=1e-2
content_weight=1e4
total_variation_weight=30 # avoid high frequency variations



def main():
    content_path, style_path = sys.argv[1], sys.argv[2]
    
    # load images
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    # define the layers
    content_layers = ['block5_conv2'] 
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1']

    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    style_extractor = vgg_layers(style_layers)
    style_outputs = style_extractor(style_image*255)

    extractor = StyleContentModel(style_layers, content_layers)
    results = extractor(tf.constant(content_image))
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    print("Starting Gradient Descent")
    image = tf.Variable(content_image)
    opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image, style_targets, style_weight, num_style_layers, content_targets, content_weight, num_content_layers, extractor, total_variation_weight, opt)
        print("Train step: {}".format(step))
    
    imshow(style_image, content_image, image)
    tensor_to_image(image).save('output.jpg')
        

if __name__ == '__main__':
    main()