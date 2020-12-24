This project is about 0x10-nlp_metrics
0-uni_bleu.py - Write the function def uni_bleu(references, sentence): that calculates the unigram BLEU score for a sentence:
references is a list of reference translations
each reference translation is a list of the words in the translation
sentence is a list containing the model proposed sentence
Returns: the unigram BLEU score

1-ngram_bleu.py - Write the function def ngram_bleu(references, sentence, n): that calculates the n-gram BLEU score for a sentence:
references is a list of reference translations
each reference translation is a list of the words in the translation
sentence is a list containing the model proposed sentence
n is the size of the n-gram to use for evaluation
Returns: the n-gram BLEU score

2-cumulative_bleu.py - Write the function def cumulative_bleu(references, sentence, n): that calculates the cumulative n-gram BLEU score for a sentence:
references is a list of reference translations
each reference translation is a list of the words in the translation
sentence is a list containing the model proposed sentence
n is the size of the largest n-gram to use for evaluation
All n-gram scores should be weighted evenly
Returns: the cumulative n-gram BLEU score
