This project is about 0x00. Binary Classification
0-neuron.py - Write a class Neuron that defines a single neuron performing binary classification:
class constructor: def __init__(self, nx):
nx is the number of input features to the neuron
If nx is not an integer, raise a TypeError with the exception: nx must be an integer
If nx is less than 1, raise a ValueError with the exception: nx must be a positive integer
All exceptions should be raised in the order listed above
Public instance attributes:
W: The weights vector for the neuron. Upon instantiation, it should be initialized using a random normal distribution.
b: The bias for the neuron. Upon instantiation, it should be initialized to 0.
A: The activated output of the neuron (prediction). Upon instantiation, it should be initialized to 0.

1-neuron.py - Write a class Neuron that defines a single neuron performing binary classification (Based on 0-neuron.py):
class constructor: def __init__(self, nx):
nx is the number of input features to the neuron
If nx is not an integer, raise a TypeError with the exception: nx must be a integer
If nx is less than 1, raise a ValueError with the exception: nx must be positive
All exceptions should be raised in the order listed above
Private instance attributes:
__W: The weights vector for the neuron. Upon instantiation, it should be initialized using a random normal distribution.
__b: The bias for the neuron. Upon instantiation, it should be initialized to 0.
__A: The activated output of the neuron (prediction). Upon instantiation, it should be initialized to 0.
Each private attribute should have a corresponding getter function (no setter function).

2-neuron.py - Write a class Neuron that defines a single neuron performing binary classification (Based on 1-neuron.py):
Add the public method def forward_prop(self, X):
Calculates the forward propagation of the neuron
X is a numpy.ndarray with shape (nx, m) that contains the input data
nx is the number of input features to the neuron
m is the number of examples
Updates the private attribute __A
The neuron should use a sigmoid activation function
Returns the private attribute __A

3-neuron.py - Write a class Neuron that defines a single neuron performing binary classification (Based on 2-neuron.py):
Add the public method def cost(self, Y, A):
Calculates the cost of the model using logistic regression
Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
A is a numpy.ndarray with shape (1, m) containing the activated output of the neuron for each example
To avoid division by zero errors, please use 1.0000001 - A instead of 1 - A
Returns the cost

4-neuron.py - Write a class Neuron that defines a single neuron performing binary classification (Based on 3-neuron.py):
Add the public method def evaluate(self, X, Y):
Evaluates the neuron’s predictions
X is a numpy.ndarray with shape (nx, m) that contains the input data
nx is the number of input features to the neuron
m is the number of examples
Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
Returns the neuron’s prediction and the cost of the network, respectively
The prediction should be a numpy.ndarray with shape (1, m) containing the predicted labels for each example
The label values should be 1 if the output of the network is >= 0.5 and 0 otherwise

5-neuron.py - Write a class Neuron that defines a single neuron performing binary classification (Based on 4-neuron.py):
Add the public method def gradient_descent(self, X, Y, A, alpha=0.05):
Calculates one pass of gradient descent on the neuron
X is a numpy.ndarray with shape (nx, m) that contains the input data
nx is the number of input features to the neuron
m is the number of examples
Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
A is a numpy.ndarray with shape (1, m) containing the activated output of the neuron for each example
alpha is the learning rate
Updates the private attributes __W and __b

6-neuron.py - Write a class Neuron that defines a single neuron performing binary classification (Based on 5-neuron.py):
Add the public method def train(self, X, Y, iterations=5000, alpha=0.05):
Trains the neuron
X is a numpy.ndarray with shape (nx, m) that contains the input data
nx is the number of input features to the neuron
m is the number of examples
Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
iterations is the number of iterations to train over
if iterations is not an integer, raise a TypeError with the exception iterations must be an integer
if iterations is not positive, raise a ValueError with the exception iterations must be a positive integer
alpha is the learning rate
if alpha is not a float, raise a TypeError with the exception alpha must be a float
if alpha is not positive, raise a ValueError with the exception alpha must be positive
All exceptions should be raised in the order listed above
Updates the private attributes __W, __b, and __A
You are allowed to use one loop
Returns the evaluation of the training data after iterations of training have occurred

7-neuron.py - Write a class Neuron that defines a single neuron performing binary classification (Based on 6-neuron.py):
Update the public method train to def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
Trains the neuron by updating the private attributes __W, __b, and __A
X is a numpy.ndarray with shape (nx, m) that contains the input data
nx is the number of input features to the neuron
m is the number of examples
Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
iterations is the number of iterations to train over
if iterations is not an integer, raise a TypeError with the exception iterations must be an integer
if iterations is not positive, raise a ValueError with the exception iterations must be a positive integer
alpha is the learning rate
if alpha is not a float, raise a TypeError with the exception alpha must be a float
if alpha is not positive, raise a ValueError with the exception alpha must be positive
verbose is a boolean that defines whether or not to print information about the training. If True, print Cost after {iteration} iterations: {cost} every step iterations:
Include data from the 0th and last iteration
graph is a boolean that defines whether or not to graph information about the training once the training has completed. If True:
Plot the training data every step iterations as a blue line
Label the x-axis as iteration
Label the y-axis as cost
Title the plot Training Cost
Include data from the 0th and last iteration
Only if either verbose or graph are True:
if step is not an integer, raise a TypeError with the exception step must be an integer
if step is not positive or is greater than iterations, raise a ValueError with the exception step must be positive and <= iterations
All exceptions should be raised in the order listed above
The 0th iteration should represent the state of the neuron before any training has occurred
You are allowed to use one loop
You can use import matplotlib.pyplot as plt
Returns: the evaluation of the training data after iterations of training have occurred

8-neural_network.py - Write a class NeuralNetwork that defines a neural network with one hidden layer performing binary classification:
class constructor: def __init__(self, nx, nodes):
nx is the number of input features
If nx is not an integer, raise a TypeError with the exception: nx must be an integer
If nx is less than 1, raise a ValueError with the exception: nx must be a positive integer
nodes is the number of nodes found in the hidden layer
If nodes is not an integer, raise a TypeError with the exception: nodes must be an integer
If nodes is less than 1, raise a ValueError with the exception: nodes must be a positive integer
All exceptions should be raised in the order listed above
Public instance attributes:
W1: The weights vector for the hidden layer. Upon instantiation, it should be initialized using a random normal distribution.
b1: The bias for the hidden layer. Upon instantiation, it should be initialized with 0’s.
A1: The activated output for the hidden layer. Upon instantiation, it should be initialized to 0.
W2: The weights vector for the output neuron. Upon instantiation, it should be initialized using a random normal distribution.
b2: The bias for the output neuron. Upon instantiation, it should be initialized to 0.
A2: The activated output for the output neuron (prediction). Upon instantiation, it should be initialized to 0.

9-neural_network.py - Write a class NeuralNetwork that defines a neural network with one hidden layer performing binary classification (based on 8-neural_network.py):
class constructor: def __init__(self, nx, nodes):
nx is the number of input features
If nx is not an integer, raise a TypeError with the exception: nx must be an integer
If nx is less than 1, raise a ValueError with the exception: nx must be a positive integer
nodes is the number of nodes found in the hidden layer
If nodes is not an integer, raise a TypeError with the exception: nodes must be an integer
If nodes is less than 1, raise a ValueError with the exception: nodes must be a positive integer
All exceptions should be raised in the order listed above
Private instance attributes:
W1: The weights vector for the hidden layer. Upon instantiation, it should be initialized using a random normal distribution.
b1: The bias for the hidden layer. Upon instantiation, it should be initialized with 0’s.
A1: The activated output for the hidden layer. Upon instantiation, it should be initialized to 0.
W2: The weights vector for the output neuron. Upon instantiation, it should be initialized using a random normal distribution.
b2: The bias for the output neuron. Upon instantiation, it should be initialized to 0.
A2: The activated output for the output neuron (prediction). Upon instantiation, it should be initialized to 0.
Each private attribute should have a corresponding getter function (no setter function).

10-neural_network.py - Write a class NeuralNetwork that defines a neural network with one hidden layer performing binary classification (based on 9-neural_network.py):
Add the public method def forward_prop(self, X):
Calculates the forward propagation of the neural network
X is a numpy.ndarray with shape (nx, m) that contains the input data
nx is the number of input features to the neuron
m is the number of examples
Updates the private attributes __A1 and __A2
The neurons should use a sigmoid activation function
Returns the private attributes __A1 and __A2, respectively

11-neural_network.py - Write a class NeuralNetwork that defines a neural network with one hidden layer performing binary classification (based on 10-neural_network.py):
Add the public method def cost(self, Y, A):
Calculates the cost of the model using logistic regression
Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
A is a numpy.ndarray with shape (1, m) containing the activated output of the neuron for each example
To avoid division by zero errors, please use 1.0000001 - A instead of 1 - A
Returns the cost

12-neural_network.py - Write a class NeuralNetwork that defines a neural network with one hidden layer performing binary classification (based on 11-neural_network.py):
Add the public method def evaluate(self, X, Y):
Evaluates the neural network’s predictions
X is a numpy.ndarray with shape (nx, m) that contains the input data
nx is the number of input features to the neuron
m is the number of examples
Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
Returns the neuron’s prediction and the cost of the network, respectively
The prediction should be a numpy.ndarray with shape (1, m) containing the predicted labels for each example
The label values should be 1 if the output of the network is >= 0.5 and 0 otherwise

13-neural_network.py - Write a class NeuralNetwork that defines a neural network with one hidden layer performing binary classification (based on 12-neural_network.py):
Add the public method def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
Calculates one pass of gradient descent on the neural network
X is a numpy.ndarray with shape (nx, m) that contains the input data
nx is the number of input features to the neuron
m is the number of examples
Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
A1 is the output of the hidden layer
A2 is the predicted output
alpha is the learning rate
Updates the private attributes __W1, __b1, __W2, and __b2

14-neural_network.py - Write a class NeuralNetwork that defines a neural network with one hidden layer performing binary classification (based on 13-neural_network.py):
Add the public method def train(self, X, Y, iterations=5000, alpha=0.05):
Trains the neural network
X is a numpy.ndarray with shape (nx, m) that contains the input data
nx is the number of input features to the neuron
m is the number of examples
Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
iterations is the number of iterations to train over
if iterations is not an integer, raise a TypeError with the exception iterations must be an integer
if iterations is not positive, raise a ValueError with the exception iterations must be a positive integer
alpha is the learning rate
if alpha is not a float, raise a TypeError with the exception alpha must be a float
if alpha is not positive, raise a ValueError with the exception alpha must be positive
All exceptions should be raised in the order listed above
Updates the private attributes __W1, __b1, __A1, __W2, __b2, and __A2
You are allowed to use one loop
Returns the evaluation of the training data after iterations of training have occurred

15-neural_network.py - Write a class NeuralNetwork that defines a neural network with one hidden layer performing binary classification (based on 14-neural_network.py):
Update the public method train to def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
Trains the neural network
X is a numpy.ndarray with shape (nx, m) that contains the input data
nx is the number of input features to the neuron
m is the number of examples
Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
iterations is the number of iterations to train over
if iterations is not an integer, raise a TypeError with the exception iterations must be an integer
if iterations is not positive, raise a ValueError with the exception iterations must be a positive integer
alpha is the learning rate
if alpha is not a float, raise a TypeError with the exception alpha must be a float
if alpha is not positive, raise a ValueError with the exception alpha must be positive
Updates the private attributes __W1, __b1, __A1, __W2, __b2, and __A2
verbose is a boolean that defines whether or not to print information about the training. If True, print Cost after {iteration} iterations: {cost} every step iterations:
Include data from the 0th and last iteration
graph is a boolean that defines whether or not to graph information about the training once the training has completed. If True:
Plot the training data every step iterations as a blue line
Label the x-axis as iteration
Label the y-axis as cost
Title the plot Training Cost
Include data from the 0th and last iteration
Only if either verbose or graph are True:
if step is not an integer, raise a TypeError with the exception step must be an integer
if step is not positive and less than or equal to iterations, raise a ValueError with the exception step must be positive and <= iterations
All exceptions should be raised in the order listed above
The 0th iteration should represent the state of the neuron before any training has occurred
You are allowed to use one loop
Returns the evaluation of the training data after iterations of training have occurred

16-deep_neural_network.py - Write a class DeepNeuralNetwork that defines a deep neural network performing binary classification:
class constructor: def __init__(self, nx, layers):
nx is the number of input features
If nx is not an integer, raise a TypeError with the exception: nx must be an integer
If nx is less than 1, raise a ValueError with the exception: nx must be a positive integer
layers is a list representing the number of nodes in each layer of the network
If layers is not a list, raise a TypeError with the exception: layers must be a list of positive integers
The first value in layers represents the number of nodes in the first layer, …
If the elements in layers are not all positive integers, raise a TypeError with the exception layers must be a list of positive integers
All exceptions should be raised in the order listed above
Sets the public instance attributes:
L: The number of layers in the neural network.
cache: A dictionary to hold all intermediary values of the network. Upon instantiation, it should be set to an empty dictionary.
weights: A dictionary to hold all weights and biased of the network. Upon instantiation:
The weights of the network should be initialized using the He et al. method and saved in the weights dictionary using the key W{l} where {l} is the hidden layer the weight belongs to
The biases of the network should be initialized to 0’s and saved in the weights dictionary using the key b{l} where {l} is the hidden layer the bias belongs to
You are allowed to use one loop

17-deep_neural_network.py - Write a class DeepNeuralNetwork that defines a deep neural network performing binary classification (based on 16-deep_neural_network.py):
class constructor: def __init__(self, nx, layers):
nx is the number of input features
If nx is not an integer, raise a TypeError with the exception: nx must be an integer
If nx is less than 1, raise a ValueError with the exception: nx must be a positive integer
layers is a list representing the number of nodes in each layer of the network
If layers is not a list, raise a TypeError with the exception: layers must be a list of positive integers
The first value in layers represents the number of nodes in the first layer, …
If the elements in layers are not all positive integers, raise a TypeError with the exception layers must be a list of positive integers
All exceptions should be raised in the order listed above
Sets the private instance attributes:
__L: The number of layers in the neural network.
__cache: A dictionary to hold all intermediary values of the network. Upon instantiation, it should be set to an empty dictionary.
__weights: A dictionary to hold all weights and biased of the network. Upon instantiation:
The weights of the network should be initialized using the He et al. method and saved in the __weights dictionary using the key W{l} where {l} is the hidden layer the weight belongs to
The biases of the network should be initialized to 0‘s and saved in the __weights dictionary using the key b{l} where {l} is the hidden layer the bias belongs to
Each private attribute should have a corresponding getter function (no setter function).
You are allowed to use one loop

18-deep_neural_network.py - Write a class DeepNeuralNetwork that defines a deep neural network performing binary classification (based on 17-deep_neural_network.py):
Add the public method def forward_prop(self, X):
Calculates the forward propagation of the neural network
X is a numpy.ndarray with shape (nx, m) that contains the input data
nx is the number of input features to the neuron
m is the number of examples
Updates the private attribute __cache:
The activated outputs of each layer should be saved in the __cache dictionary using the key A{l} where {l} is the hidden layer the activated output belongs to
X should be saved to the cache dictionary using the key A0
All neurons should use a sigmoid activation function
You are allowed to use one loop
Returns the output of the neural network and the cache, respectively

19-deep_neural_network.py - Write a class DeepNeuralNetwork that defines a deep neural network performing binary classification (based on 18-deep_neural_network.py):
Add the public method def cost(self, Y, A):
Calculates the cost of the model using logistic regression
Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
A is a numpy.ndarray with shape (1, m) containing the activated output of the neuron for each example
To avoid division by zero errors, please use 1.0000001 - A instead of 1 - A
Returns the cost

20-deep_neural_network.py - Write a class DeepNeuralNetwork that defines a deep neural network performing binary classification (based on 19-deep_neural_network.py):
Add the public method def evaluate(self, X, Y):
Evaluates the neural network’s predictions
X is a numpy.ndarray with shape (nx, m) that contains the input data
nx is the number of input features to the neuron
m is the number of examples
Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
Returns the neuron’s prediction and the cost of the network, respectively
The prediction should be a numpy.ndarray with shape (1, m) containing the predicted labels for each example
The label values should be 1 if the output of the network is >= 0.5 and 0 otherwise

21-deep_neural_network.py - Write a class DeepNeuralNetwork that defines a deep neural network performing binary classification (based on 20-deep_neural_network.py):
Add the public method def gradient_descent(self, Y, cache, alpha=0.05):
Calculates one pass of gradient descent on the neural network
Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
cache is a dictionary containing all the intermediary values of the network
alpha is the learning rate
Updates the private attribute __weights
You are allowed to use one loop

22-deep_neural_network.py - Write a class DeepNeuralNetwork that defines a deep neural network performing binary classification (based on 21-deep_neural_network.py):
Add the public method def train(self, X, Y, iterations=5000, alpha=0.05):
Trains the deep neural network
X is a numpy.ndarray with shape (nx, m) that contains the input data
nx is the number of input features to the neuron
m is the number of examples
Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
iterations is the number of iterations to train over
if iterations is not an integer, raise a TypeError with the exception iterations must be an integer
if iterations is not positive, raise a ValueError with the exception iterations must be a positive integer
alpha is the learning rate
if alpha is not a float, raise a TypeError with the exception alpha must be a float
if alpha is not positive, raise a ValueError with the exception alpha must be positive
All exceptions should be raised in the order listed above
Updates the private attributes __weights and __cache
You are allowed to use one loop
Returns the evaluation of the training data after iterations of training have occurred

23-deep_neural_network.py - Write a class DeepNeuralNetwork that defines a deep neural network performing binary classification (based on 22-deep_neural_network.py):
Update the public method train to def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
Trains the deep neural network by updating the private attributes __weights and __cache
X is a numpy.ndarray with shape (nx, m) that contains the input data
nx is the number of input features to the neuron
m is the number of examples
Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
iterations is the number of iterations to train over
if iterations is not an integer, raise a TypeError with the exception iterations must be an integer
if iterations is not positive, raise a ValueError with the exception iterations must be a positive integer
alpha is the learning rate
if alpha is not a float, raise a TypeError with the exception alpha must be a float
if alpha is not positive, raise a ValueError with the exception alpha must be positive
verbose is a boolean that defines whether or not to print information about the training. If True, print Cost after {iteration} iterations: {cost} every step iterations:
Include data from the 0th and last iteration
graph is a boolean that defines whether or not to graph information about the training once the training has completed. If True:
Plot the training data every step iterations as a blue line
Label the x-axis as iteration
Label the y-axis as cost
Title the plot Training Cost
Include data from the 0th and last iteration
Only if either verbose or graph are True:
if step is not an integer, raise a TypeError with the exception step must be an integer
if step is not positive and less than or equal to iterations, raise a ValueError with the exception step must be positive and <= iterations
All exceptions should be raised in the order listed above
The 0th iteration should represent the state of the neuron before any training has occurred
You are allowed to use one loop
Returns the evaluation of the training data after iterations of training have occurred
