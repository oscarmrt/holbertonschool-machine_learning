#!/usr/bin/env python3
"""class Dataset"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset():
    """class Dataset"""
    def __init__(self, batch_size, max_len):
        """class constructor"""
        def filter_max_length(x, y, max_length=max_len):
            """Function that uses the filter method"""
            return tf.logical_and(tf.size(x) <= max_length,
                                  tf.size(y) <= max_length)
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        data_train, data_valid = examples['train'], examples['validation']
        tokenizer_pt, tokenizer_en = self.tokenize_dataset(data_train)
        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en
        data_train = data_train.map(self.tf_encode)
        data_train = data_train.filter(filter_max_length)
        data_train = data_train.cache()
        BUFFER_SIZE = metadata.splits['train'].num_examples
        data_train = data_train.shuffle(BUFFER_SIZE).\
            padded_batch(batch_size, padded_shapes=([None], [None]))
        self.data_train = data_train.prefetch(tf.data.experimental.AUTOTUNE)
        data_valid = data_valid.map(self.tf_encode)
        data_valid = data_valid.filter(filter_max_length).\
            padded_batch(batch_size, padded_shapes=([None], [None]))
        self.data_valid = data_valid

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
