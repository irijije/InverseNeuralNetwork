import os
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
NN_MODEL_NAME = "models/small.h5"
INN_MODEL_NAME = "models/small_inv.h5"
BUFFER_SIZE = 60000
BATCH_SIZE = 256

class InverseNN:
    def __init__(self):
        (self.train_x, self.train_y), (self.test_x, self.test_y) = self.load_dataset("dataset/CICIDS2018_small.csv")

        self.train_dataset = tf.data.Dataset.from_tensor_slices(self.train_x.values).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

        self.inn_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(15,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(78, activation='sigmoid')
        ])

        self.nn_model = tf.keras.models.load_model(NN_MODEL_NAME)

        self.optimizer = tf.keras.optimizers.Adam(1e-4)


    def loss(self, x, x_, y, y_, C):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        generate_loss = tf.reduce_mean(tf.square(tf.abs(x - x_)))
        classify_loss = cross_entropy(y, 1-y_)
        total_loss = generate_loss + C*classify_loss

        return total_loss

    @tf.function
    def train_step(self, x, C):
        with tf.GradientTape() as tape:
            y = self.nn_model(x)
            x_ = self.inn_model(y)
            y_ = self.nn_model(x_)

            loss = self.loss(x, x_, y, y_, C)

        gradients = tape.gradient(loss, self.inn_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.inn_model.trainable_variables))

    def train(self, epochs, C):
        for epoch in range(epochs):
            print("epoch: {} training".format(epoch))
            for batch in self.train_dataset:
                self.train_step(batch, C)

        self.inn_model.save(INN_MODEL_NAME)
    
    def norm(self, data, stats):
        #return ((data - stats['mean']) / (stats['std']+0.00001))
        return (data-stats['min']) / (stats['max']-stats['min']+0.00001)

    def load_dataset(self, filepath):
        df = pd.read_csv(filepath).replace([np.inf, -np.inf], np.nan).dropna().astype('float32')
        desc = df.describe().drop(['Label'], axis=1).transpose()
        train_data, test_data = train_test_split(df, test_size=0.2)
        train_labels = train_data.pop('Label')
        test_labels = test_data.pop('Label')
        train_data = self.norm(train_data, desc)
        test_data = self.norm(test_data, desc)

        return (train_data, train_labels), (test_data, test_labels)


if __name__ == "__main__":
    inn = InverseNN()
    inn.train(5, 0.1)