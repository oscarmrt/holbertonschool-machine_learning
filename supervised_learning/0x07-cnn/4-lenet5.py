#!/usr/bin/env python3
"""Program that builds a modified version of the LeNet-5
architecture using tensorflow"""
import tensorflow as tf


def lenet5(x, y):
    """Function that builds a modified version of the LeNet-5
    architecture using tensorflow"""
    # convolutional layer 1
    # 6 kernels 5x5, padding = same
    init = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.Conv2D(
          filters=6,  # Number of filters.
          kernel_size=5,  # Size of each filter is 5x5.
          padding="same",  # padding is applied to the input.
          activation=tf.nn.relu,  # relu activation function
          kernel_initializer=init)(x)

    # pooling layer #1
    pool1 = tf.layers.MaxPooling2D(pool_size=[2, 2],
                                   strides=2)(conv1)

    # convolutional layer 2
    # 16 kernels 5x5, padding = valid
    conv2 = tf.layers.Conv2D(
          filters=16,  # Number of filters.
          kernel_size=5,  # Size of each filter is 5x5.
          padding="valid",  # padding is applied to the input.
          activation=tf.nn.relu,  # relu activation function
          kernel_initializer=init)(pool1)

    # pooling layer #2
    pool2 = tf.layers.MaxPooling2D(pool_size=[2, 2],
                                   strides=2)(conv2)

    # Reshaping output into a single dimention array for input
    # to fully connected layer
    # pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])
    pool2_flat = tf.layers.Flatten()(pool2)

    # Fully connected layer #1: Has 120 neurons
    dense1 = tf.layers.Dense(units=120, activation=tf.nn.relu,
                             kernel_initializer=init)(pool2_flat)

    # dense1_flat = Flatten()(dense1)
    # Fully connected layer #2: Has 84 neurons
    dense2 = tf.layers.Dense(units=84, activation=tf.nn.relu,
                             kernel_initializer=init)(dense1)

    # Output layer, 10 neurons for each digit
    dense3 = tf.layers.Dense(units=10, kernel_initializer=init)(dense2)

    softmax = tf.nn.softmax(dense3)

    # Compute the cross-entropy loss
    y_pred = dense3
    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    # Use adam optimizer to reduce cost
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    # For testing and prediction
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return softmax, train_op, loss, accuracy
