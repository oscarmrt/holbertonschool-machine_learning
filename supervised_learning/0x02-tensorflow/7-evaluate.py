#!/usr/bin/env python3
"""Program that evaluates the output of a neural network"""
import tensorflow as tf


def evaluate(X, Y, save_path):
    """Function that evaluates the output of a neural network"""
    savedNN = tf.train.import_meta_graph("{}.meta".format(save_path))
    with tf.Session() as sess:
        savedNN.restore(sess, save_path)
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_predict = tf.get_collection("y_predict")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        n_predict = sess.run(y_predict, feed_dict={x: X, y: Y})
        n_accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})
        n_loss = sess.run(loss, feed_dict={x: X, y: Y})
        return n_predict, n_accuracy, n_loss
