This project is about 0x03-policy_gradients
policy_gradient.py - Write a function that computes to policy with a weight of a matrix.
policy_gradient.py - By using the previous function created policy, write a function that computes the Monte-Carlo policy gradient based on a state and a weight matrix.
Prototype: def policy_gradient(state, weight):
state: matrix representing the current observation of the environment
weight: matrix of random weight
Return: the action and the gradient (in this order)
train.py - By using the previous function created policy_gradient, write a function that implements a full training.
Prototype: def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
env: initial environment
nb_episodes: number of episodes used for training
alpha: the learning rate
gamma: the discount factor
Return: all values of the score (sum of all rewards during one episode loop)
Since the training is quite long, please print the current episode number and the score after each loop. To display these information on the same line, you can use end="\r", flush=False of the print function.
train.py - Update the prototype of the train function by adding a last optional parameter show_result (default: False).
When this parameter is True, render the environment every 1000 episodes computed.
