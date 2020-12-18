This project is about 0x0F-word_embeddings
0-bag_of_words.py - Write a function def bag_of_words(sentences, vocab=None): that creates a bag of words embedding matrix:
sentences is a list of sentences to analyze
vocab is a list of the vocabulary words to use for the analysis
If None, all words within sentences should be used
Returns: embeddings, features
embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
s is the number of sentences in sentences
f is the number of features analyzed
features is a list of the features used for embeddings

1-tf_idf.py - Write a function def tf_idf(sentences, vocab=None): that creates a TF-IDF embedding:
sentences is a list of sentences to analyze
vocab is a list of the vocabulary words to use for the analysis
If None, all words within sentences should be used
Returns: embeddings, features
embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
s is the number of sentences in sentences
f is the number of features analyzed
features is a list of the features used for embeddings

2-word2vec.py - Write a function def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5, cbow=True, iterations=5, seed=0, workers=1): that creates and trains a gensim word2vec model:
sentences is a list of sentences to be trained on
size is the dimensionality of the embedding layer
min_count is the minimum number of occurrences of a word for use in training
window is the maximum distance between the current and predicted word within a sentence
negative is the size of negative sampling
cbow is a boolean to determine the training type; True is for CBOW; False is for Skip-gram
iterations is the number of iterations to train over
seed is the seed for the random number generator
workers is the number of worker threads to train the model
Returns: the trained model

3-gensim_to_keras.py - Write a function def gensim_to_keras(model): that converts a gensim word2vec model to a keras Embedding layer:
model is a trained gensim word2vec models
Returns: the trainable keras Embedding

4-fasttext.py - Write a function def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5, cbow=True, iterations=5, seed=0, workers=1): that creates and trains a genism fastText model:
sentences is a list of sentences to be trained on
size is the dimensionality of the embedding layer
min_count is the minimum number of occurrences of a word for use in training
window is the maximum distance between the current and predicted word within a sentence
negative is the size of negative sampling
cbow is a boolean to determine the training type; True is for CBOW; False is for Skip-gram
iterations is the number of iterations to train over
seed is the seed for the random number generator
workers is the number of worker threads to train the model
Returns: the trained model

5-elmo - When training an ELMo embedding model, you are training:
The internal weights of the BiLSTM
The character embedding layer
The weights applied to the hidden states
In the text file 5-elmo, write the letter answer, followed by a newline, that lists the correct statements:
A. 1, 2, 3
B. 1, 2
C. 2, 3
D. 1, 3
E. 1
F. 2
G. 3
H. None of the above
