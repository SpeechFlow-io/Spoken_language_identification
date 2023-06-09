# Copyright 2023 by zhongying

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.keras.optimizers.schedules import ExponentialDecay


class TransformerLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ Transformer learning rate schedule """

    def __init__(self, d_model, init_steps=0, warmup_steps=4000, max_lr=None):
        super(TransformerLRSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.init_steps = init_steps

    def __call__(self, step):
        # lr = (d_model^-0.5) * min(step^-0.5, step*(warm_up^-1.5))
        step += self.init_steps
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        if self.max_lr is not None:
            return tf.math.minimum(self.max_lr, lr)
        return lr

    def get_config(self):
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "max_lr": self.max_lr
        }


class SANSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lamb, d_model, warmup_steps=4000):
        super(SANSchedule, self).__init__()

        self.lamb = tf.cast(lamb, tf.float32)
        self.d_model = tf.cast(d_model, tf.float32)

        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        arg1 = step / (self.warmup_steps ** 1.5)
        arg2 = 1 / tf.math.sqrt(step)

        return (self.lamb / tf.math.sqrt(self.d_model)) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            "lamb": self.lamb,
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps
        }


class BoundExponentialDecay(ExponentialDecay):
    def __init__(self, min_lr=0.0, **kwargs):
        super().__init__(**kwargs)
        self.min_lr = min_lr

    def __call__(self, step):
        with ops.name_scope_v2(self.name or "ExponentialDecay") as name:
            initial_learning_rate = ops.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            decay_steps = math_ops.cast(self.decay_steps, dtype)
            decay_rate = math_ops.cast(self.decay_rate, dtype)

            global_step_recomp = math_ops.cast(step, dtype)
            p = global_step_recomp / decay_steps
            if self.staircase:
                p = math_ops.floor(p)
            new_lr = math_ops.multiply(
                initial_learning_rate, math_ops.pow(decay_rate, p), name=name)
            return math_ops.maximum(self.min_lr, new_lr)
