#!/usr/bin/env python3
"""class Dataset"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """class Dataset"""
    def __init__(self):
        """class constructor"""
        data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                               split='train', as_supervised=True)
        data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                               split='validation', as_supervised=True)
        tokenizer_pt, tokenizer_en = self.tokenize_dataset(data_train)
        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en
        self.data_train = data_train.map(self.tf_encode)
        self.data_valid = data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """Instance method that creates sub-word tokenizers for our dataset"""
        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data), target_vocab_size=2**15)
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data), target_vocab_size=2**15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Instance method that encodes a translation into tokens"""
        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size+1]
        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size+1]
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """Instance method that acts as a tensorflow wrapper for the encode
        instance method"""
        res_pt, res_en = tf.py_function(self.encode, [pt, en],
                                        [tf.int64, tf.int64])
        res_pt.set_shape([None])
        res_en.set_shape([None])
        return res_pt, res_en
