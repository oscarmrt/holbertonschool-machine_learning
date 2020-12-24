#!/usr/bin/env python3
"""Program that calculates the cumulative n-gram BLEU score for a sentence"""
import numpy as np


def create_ngram(sentence, n):
    """Function that creates an ngram"""
    new_sentence = []
    for i in range(len(sentence) - n + 1):
        n_gram = ""
        for j in range(n):
            n_gram += sentence[i + j]
            if not j + 1 == n:
                n_gram += " "
        new_sentence.append(n_gram)
    return new_sentence


def ngram_bleu(references, sentence, n):
    """Function that calculates the n-gram BLEU score for a sentence"""
    c = len(sentence)
    r_list = np.array([abs(len(ref) - c) for ref in references])
    mask = np.where(r_list == r_list.min())
    r = np.array([len(ref) for ref in references])[mask].min()
    cn = create_ngram(sentence, n)
    candidate = {x: 0 for x in cn}
    references = [create_ngram(ref, n) for ref in references]
    max_match = 0
    for ref in references:
        match = 0
        ref_dict = {x: ref.count(x) for x in ref}
        for key in ref_dict.keys():
            if key in candidate:
                match += 1
            if match > max_match:
                max_match = match
    P = max_match / len(cn)
    return P, r


def cumulative_bleu(references, sentence, n):
    """Function that calculates the cumulative
    n-gram BLEU score for a sentence"""
    c = len(sentence)
    precisions = []
    for i in range(n):
        P, r = ngram_bleu(references, sentence, i + 1)
        precisions.append(P)
    if c > r:
        BP = 1
    else:
        BP = np.exp(1-(r / c))
    return BP * np.exp(np.log(precisions).sum() * (1/n))
