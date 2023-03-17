# coding=utf-8
# Copyright 2020 Beijing BluePulse Corp.
# Created by Zhang Guanqun on 2020/6/5


import matplotlib.pyplot as plt
import os
import tensorflow as tf
from typing import Union, List
import unicodedata


def preprocess_paths(paths: Union[List, str]):
    if isinstance(paths, list):
        return [os.path.abspath(os.path.expanduser(path)) for path in paths]
    return os.path.abspath(os.path.expanduser(paths)) if paths else None


def get_reduced_length(length, reduction_factor):
    return tf.cast(tf.math.ceil(tf.divide(length, tf.cast(reduction_factor, dtype=length.dtype))), dtype=tf.int32)


def merge_two_last_dims(x):
    b, _, f, c = shape_list(x)
    return tf.reshape(x, shape=[b, -1, f * c])


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


# draw loss pic
def plot_metric(history, metric, pic_file_name):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.savefig(pic_file_name)


# against LAS loop decoding
def text_no_repeat(s):
    repeat_times = 0
    repeat_pattern = ''
    for i in range(1, len(s) // 2):
        pos = i
        if s[0 - 2 * pos:0 - pos] == s[0 - i:]:
            tmp_repeat_pattern = s[0 - i:]
            tmp_repeat_times = 1
            while pos * (tmp_repeat_times + 2) <= len(s) \
                    and s[0 - pos * (tmp_repeat_times + 2):0 - pos * (tmp_repeat_times + 1)] == s[0 - i:]:
                tmp_repeat_times += 1
            if tmp_repeat_times * len(tmp_repeat_pattern) > repeat_times * len(repeat_pattern):
                repeat_times = tmp_repeat_times
                repeat_pattern = tmp_repeat_pattern
    # print(repeat_pattern, '*', repeat_times)
    if len(repeat_pattern) != 1:
        s = s[:0 - repeat_times * len(repeat_pattern)] if repeat_times > 0 else s
    # print(s)
    return s

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator