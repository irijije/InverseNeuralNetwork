import sys
import tensorflow as tf
import numpy as np


BINARY_SEARCH_STEPS = 9
MAX_ITERATIONS = 10000
ABORT_EARLY = True
LEARNING_RATE = 1e-2
TARGETED = True
CONFIDENCE = 0
INITIAL_CONST = 1e-3
BATCH_SIZE = 1
FEATURE_SIZE = 78
NUM_LABELS = 15


class CarliniL2:
    def __init__(self, model, boxmin = -0.5, boxmax = 0.5):
        self.repeat = BINARY_SEARCH_STEPS >= 10

        shape = (BATCH_SIZE, FEATURE_SIZE)
        modifier = tf.Variable(np.zeros(shape, dtype=np.float32))

        self.test_traffic = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.test_label = tf.Variable(np.zeros((BATCH_SIZE, NUM_LABELS)), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(BATCH_SIZE), dtype=tf.float32)

        self.assign_test_traffic = tf.placeholder(tf.float32, shape)
        self.assign_test_label = tf.placeholder(tf.float32, (BATCH_SIZE, NUM_LABELS))
        self.assign_const = tf.placeholder(tf.float32, [BATCH_SIZE])

        self.boxmul = (boxmax - boxmin) / 2.
        self.boxplus = (boxmin + boxmax) / 2.
        self.new_traffic = tf.tanh(modifier + self.test_traffic)*self.boxmul + self.boxplus

        self.output = model.predict(self.new_traffic)

        self.l2dist = tf.reduce_sum(tf.square(self.ne                            w_traffic-(tf.tanh(self.test_traffic)*self.boxmul + self.boxplus)), [1,2,3])

        real = tf.reduce_sum((self.test_label)*self.output, 1)
        other = tf.reduce_max((1-self.test_label)*self.output - (self.tlab*10000), 1)

        if self.TARGETED:
            loss1 = tf.maximum(0.0, other-real+CONFIDENCE)
        else:
            loss1 = tf.maximum(0.0, real-other+CONFIDENCE)

            self.loss2 = tf.reduce_sum(self.l2dist)
            self.loss1 = tf.reduce_sum(self.const*loss1)
            self.loss = self.loss1 + self.losss2