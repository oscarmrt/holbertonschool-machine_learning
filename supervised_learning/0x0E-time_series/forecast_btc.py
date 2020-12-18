#!/usr/bin/env python3
"""Program that creates, trains, and validates a keras model
for the forecasting of BTC"""
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    """creates slide windows in an array """
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])
    if single_step:
        labels.append(target[i + target_size])
    else:
        labels.append(target[i:i + target_size])
    return np.array(data), np.array(labels)


def univariate_data(dataset, start_index, end_index, history_size,
                    target_size):
    """creates slide windows in an array"""
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)


def univariate():
    """univariate"""
    univariate_past_history = 24
    univariate_future_target = 0
    x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                               univariate_past_history,
                                               univariate_future_target)
    x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                           univariate_past_history,
                                           univariate_future_target)
    print('Single window of past history')
    print(x_train_uni.shape)
    print(x_train_uni[0])
    print(x_train_uni[1])
    print('\n Target price to predict')
    print(y_train_uni[0])


def multivariate():
    """multivariate"""
    df_mul.plot(subplots=True)
    dataset = df_mul.values
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    dataset = (dataset - data_mean) / data_std
    past_history = 24
    future_target = 0
    STEP = 1
    x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 2],
                                                       0, TRAIN_SPLIT,
                                                       past_history,
                                                       future_target,
                                                       STEP,single_step=True)
    x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 2],
                                                   TRAIN_SPLIT,
                                                   None, past_history,
                                                   future_target, STEP,
                                                   single_step=True)


if __name__ == '__main__':
    preprocess = __import__('preprocess_data').preprocessor
    txt = './coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
    df_W, df_mul = preprocess(txt, 730)
    df = df_W.values
    TRAIN_SPLIT = int(days * 0.8 * 24)
    tf.random.set_seed(13)
    uni_train_mean = df[:TRAIN_SPLIT].mean()
    uni_train_std = df[:TRAIN_SPLIT].std()
    uni_data = (df - uni_train_mean) / uni_train_std
    plt.figure(figsize=(14, 4))
    plt.plot(df_W.index[:TRAIN_SPLIT], df[:TRAIN_SPLIT])
    plt.plot(df_W.index[TRAIN_SPLIT:], df[TRAIN_SPLIT:])
    plt.legend(["Training", "Test"])
    plt.grid()
