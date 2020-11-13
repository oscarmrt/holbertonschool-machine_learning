This project is about 0x02-hmm
0-markov_chain.py - Write the function def markov_chain(P, s, t=1): that determines the probability of a markov chain being in a particular state after a specified number of iterations:
P is a square 2D numpy.ndarray of shape (n, n) representing the transition matrix
P[i, j] is the probability of transitioning from state i to state j
n is the number of states in the markov chain
s is a numpy.ndarray of shape (1, n) representing the probability of starting in each state
t is the number of iterations that the markov chain has been through
Returns: a numpy.ndarray of shape (1, n) representing the probability of being in a specific state after t iterations, or None on failure

1-regular.py - Write the function def regular(P): that determines the steady state probabilities of a regular markov chain:
P is a is a square 2D numpy.ndarray of shape (n, n) representing the transition matrix
P[i, j] is the probability of transitioning from state i to state j
n is the number of states in the markov chain
Returns: a numpy.ndarray of shape (1, n) containing the steady state probabilities, or None on failure

2-absorbing.py - Write the function def absorbing(P): that determines if a markov chain is absorbing:
P is a is a square 2D numpy.ndarray of shape (n, n) representing the standard transition matrix
P[i, j] is the probability of transitioning from state i to state j
n is the number of states in the markov chain
Returns: True if it is absorbing, or False on failure

3-forward.py - Write the function def forward(Observation, Emission, Transition, Initial): that performs the forward algorithm for a hidden markov model:
Observation is a numpy.ndarray of shape (T,) that contains the index of the observation
T is the number of observations
Emission is a numpy.ndarray of shape (N, M) containing the emission probability of a specific observation given a hidden state
Emission[i, j] is the probability of observing j given the hidden state i
N is the number of hidden states
M is the number of all possible observations
Transition is a 2D numpy.ndarray of shape (N, N) containing the transition probabilities
Transition[i, j] is the probability of transitioning from the hidden state i to j
Initial a numpy.ndarray of shape (N, 1) containing the probability of starting in a particular hidden state
Returns: P, F, or None, None on failure
P is the likelihood of the observations given the model
F is a numpy.ndarray of shape (N, T) containing the forward path probabilities
F[i, j] is the probability of being in hidden state i at time j given the previous observations

4-viterbi.py - Write the function def viterbi(Observation, Emission, Transition, Initial): that calculates the most likely sequence of hidden states for a hidden markov model:
Observation is a numpy.ndarray of shape (T,) that contains the index of the observation
T is the number of observations
Emission is a numpy.ndarray of shape (N, M) containing the emission probability of a specific observation given a hidden state
Emission[i, j] is the probability of observing j given the hidden state i
N is the number of hidden states
M is the number of all possible observations
Transition is a 2D numpy.ndarray of shape (N, N) containing the transition probabilities
Transition[i, j] is the probability of transitioning from the hidden state i to j
Initial a numpy.ndarray of shape (N, 1) containing the probability of starting in a particular hidden state
Returns: path, P, or None, None on failure
path is the a list of length T containing the most likely sequence of hidden states
P is the probability of obtaining the path sequence

5-backward.py - Write the function def backward(Observation, Emission, Transition, Initial): that performs the backward algorithm for a hidden markov model:
Observation is a numpy.ndarray of shape (T,) that contains the index of the observation
T is the number of observations
Emission is a numpy.ndarray of shape (N, M) containing the emission probability of a specific observation given a hidden state
Emission[i, j] is the probability of observing j given the hidden state i
N is the number of hidden states
M is the number of all possible observations
Transition is a 2D numpy.ndarray of shape (N, N) containing the transition probabilities
Transition[i, j] is the probability of transitioning from the hidden state i to j
Initial a numpy.ndarray of shape (N, 1) containing the probability of starting in a particular hidden state
Returns: P, B, or None, None on failure
Pis the likelihood of the observations given the model
B is a numpy.ndarray of shape (N, T) containing the backward path probabilities
B[i, j] is the probability of generating the future observations from hidden state i at time j

6-baum_welch.py - Write the function def baum_welch(Observations, Transition, Emission, Initial, iterations=1000): that performs the Baum-Welch algorithm for a hidden markov model:
Observations is a numpy.ndarray of shape (T,) that contains the index of the observation
T is the number of observations
Transition is a numpy.ndarray of shape (M, M) that contains the initialized transition probabilities
M is the number of hidden states
Emission is a numpy.ndarray of shape (M, N) that contains the initialized emission probabilities
N is the number of output states
Initial is a numpy.ndarray of shape (M, 1) that contains the initialized starting probabilities
iterations is the number of times expectation-maximization should be performed
Returns: the converged Transition, Emission, or None, None on failure
