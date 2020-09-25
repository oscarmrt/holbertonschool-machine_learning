This project is about 0x09-transfer_learning
0-transfer.py - Write a python script that trains a convolutional neural network to classify the CIFAR 10 dataset:
You must use one of the applications listed in Keras Applications
Your script must save your trained model in the current working directory as cifar10.h5
Your saved model should be compiled
Your saved model should have a validation accuracy of 87% or higher
Your script should not run when the file is imported
Hint1: The training and tweaking of hyperparameters may take a while so start early!
Hint2: The CIFAR 10 dataset contains 32x32 pixel images, however most of the Keras applications are trained on much larger images. Your first layer should be a lambda layer that scales up the data to the correct size
Hint3: You will want to freeze most of the application layers. Since these layers will always produce the same output, you should compute the output of the frozen layers ONCE and use those values as input to train the remaining trainable layers. This will save you A LOT of time.
In the same file, write a function def preprocess_data(X, Y): that pre-processes the data for your model:
X is a numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data, where m is the number of data points
Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
Returns: X_p, Y_p
X_p is a numpy.ndarray containing the preprocessed X
Y_p is a numpy.ndarray containing the preprocessed Y
NOTE: About half of the points for this project are for the blog post in the next task. While you are attempting to train your model, keep track of what you try and why so that you have a log to reference when it is time to write your report.
