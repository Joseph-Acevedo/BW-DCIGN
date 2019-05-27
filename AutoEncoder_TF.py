from datetime import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected


"""
Variable autoencoder for attempting to recreate images after efficient learning

o               o
o   o       o   o
o   o   o   o   o
o   o   o   o   o
o   o   o   o   o
o   o   o   o   o
o   o       o   o
o               o

"""
class AutoencoderTF:

    """ === Global Constants === """

    # Input image constants
    image_dim = 28

    # Network topology constants
    num_input = image_dim ** 2  # A 28x28 picture
    num_hid1 = num_input // 2   # Creating a bottleneck by reducing layer size by 2
    num_hid2 = num_hid1 // 2 
    num_hid3 = num_hid1
    num_output = num_input     # Same as input so it can attempt to recreate image

    # Function constant
    actf = tf.nn.relu           # ReLU Activation function

    # Tunable settings for training the autoencoder
    lr = 0.01                   # The learning rate ~ 1%
    num_epochs = 5
    batch_size = 150
    num_test_images = 10


    def __init__(self):
        print("[{}] Gathering MNIST dataset...".format(dt.now().strftime('%H:%M.%S')))
        self.mnist = input_data.read_data_sets("/MNIST_data/", one_hot=True)
        print("[{}] Creating network...".format(dt.now().strftime('%H:%M.%S')))
        self.create_and_train_network()
        print("[{}] Network training complete: displaying results...".format(dt.now().strftime('%H:%M.%S')))

    """
    Creates the network with weights and biases based on pre-defined constants
    TODO: Make the creation of layers and weights and biases more general, not hardcoded
    """
    def create_and_train_network(self):
        X = tf.placeholder(tf.float32, shape = [None, self.num_input])
        # Used since the input data is of varying size, allows us to avoid
        # dimensionality errors
        initializer = tf.variance_scaling_initializer()

        # Assigning the weights
        w1 = tf.Variable(initializer([self.num_input, self.num_hid1]), dtype=tf.float32)
        w2 = tf.Variable(initializer([self.num_hid1, self.num_hid2]), dtype=tf.float32)
        w3 = tf.Variable(initializer([self.num_hid2, self.num_hid3]), dtype=tf.float32)
        w4 = tf.Variable(initializer([self.num_hid3, self.num_output]), dtype=tf.float32)

        # Assigning the bias(ies'eses?????)
        b1 = tf.Variable(tf.zeros(self.num_hid1))
        b2 = tf.Variable(tf.zeros(self.num_hid2))
        b3 = tf.Variable(tf.zeros(self.num_hid3))
        b4 = tf.Variable(tf.zeros(self.num_output))

        # Applying activation function to all layers
        hid_layer1   = tf.nn.relu( features=(tf.matmul(X, w1) + b1), name='ReLU')
        hid_layer2   = tf.nn.relu( features=(tf.matmul(hid_layer1, w2) + b2), name='ReLU' )
        hid_layer3   = tf.nn.relu( features=(tf.matmul(hid_layer2, w3) + b3), name='ReLU' )
        output_layer = tf.nn.relu( features=(tf.matmul(hid_layer3, w4) + b4), name='ReLU' )

        # Is this.... loss?
        # Yes it is, MSE loss to be specific
        loss = tf.reduce_mean(tf.square(output_layer - X))

        optimizer = tf.train.AdamOptimizer(self.lr)
        train = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

        print("[{}] Network created; Beginning training...".format(dt.now().strftime('%H:%M.%S')))

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.num_epochs):
                num_batches = self.mnist.train.num_examples // self.batch_size
                for iteration in range(num_batches):
                    X_batch, Y_batch = self.mnist.train.next_batch(self.batch_size)
                    sess.run(train, feed_dict={X:X_batch})

                train_loss = loss.eval(feed_dict={X:X_batch})
                print("[{}] Epoch: {} Loss: {}".format(dt.now().strftime('%H:%M.%S'), epoch, train_loss))
        

            self.results = output_layer.eval(feed_dict={X:self.mnist.test.images[:self.num_test_images]}, session=sess)

            f, a = plt.subplots(2, 10, figsize = (20, 4))
            for i in range(self.num_test_images):
                a[0][i].imshow(np.reshape(self.mnist.test.images[i], (self.image_dim, self.image_dim)))
                a[1][i].imshow(np.reshape(self.results[i], (self.image_dim, self.image_dim)))

            plt.show()
        


print("[{}] Starting up...".format(dt.now().strftime('%H:%M.%S')))
AutoencoderTF()