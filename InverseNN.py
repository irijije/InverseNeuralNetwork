import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocess import load_dataset
from metadata import FEATURES

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
BASE_PATH = "dataset/"

C = 10**-10
NN_MODEL_NAME = "models/nn.h5"
INN_MODEL_NAME = "models/inv.h5"
TRAIN_DATA = "dataset/CICIDS2018_mal_train.csv"
TEST_DATA = "dataset/CICIDS2018_mal_test.csv"


class InverseNN:
    def __init__(self):
        self.train_x, self.train_y = load_dataset(TRAIN_DATA)
        self.test_x, self.test_y = load_dataset(TEST_DATA)

        self.train_dataset = tf.data.Dataset.from_tensor_slices(self.train_x.values).shuffle(60000).batch(1024)

        self.inn_model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(15,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(78, activation='sigmoid')
        ])

        self.nn_model = tf.keras.models.load_model(NN_MODEL_NAME)

        self.optimizer = tf.keras.optimizers.Adam(1e-4)

        self.C = C

    def loss(self, x, x_, y, y_):
        cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        generate_loss = tf.reduce_mean(tf.square(tf.abs(x - x_)))
        
        onehot_t = tf.one_hot(tf.zeros(tf.shape(y)[0], dtype=tf.int32), depth=15)
        classify_loss1 = cross_entropy(onehot_t, y_)

        onehot_y = tf.one_hot(tf.argmax(y, axis=1), depth=15)
        classify_loss2 = -cross_entropy(onehot_y, y_)

        classify_loss = classify_loss1
        # + classify_loss2
        
        total_loss = generate_loss + self.C*classify_loss

        return total_loss, generate_loss

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            y = self.nn_model(x, training=False)
            x_ = self.inn_model(y, training=False)
            y_ = self.nn_model(x_, training=False)

            loss, generate_loss = self.loss(x, x_, y, y_)

        gradients = tape.gradient(loss, self.inn_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.inn_model.trainable_variables))

        return generate_loss

    def train(self, C, epochs=3, filename=INN_MODEL_NAME):
        self.C = C
        losses = []
        for epoch in range(epochs):
            print("epoch: {} training".format(epoch))
            for batch in self.train_dataset:
                loss = self.train_step(batch)
            losses.append(loss)
        print("loss: ", end="")
        tf.print(loss)
        self.inn_model.save(filename)


if __name__ == "__main__":
    inn = InverseNN()
    inn.train(0.000001, 5)

    x, _ = load_dataset(TEST_DATA)
    nn_model = tf.keras.models.load_model(NN_MODEL_NAME)
    inn_model = tf.keras.models.load_model(INN_MODEL_NAME)

    y = nn_model(np.array(x), training=False)
    x_ = inn_model(y, training=False)
    
    print("\nx: {}".format(np.array2string(np.array(x)[1], prefix="x: ",
            formatter={'float_kind':lambda x: "%7.4f" % x})))
    print("\ny: {}".format(np.array2string(np.array(x_)[1], prefix="x_: ",
            formatter={'float_kind':lambda x: "%7.4f" % x})))
    stats = pd.read_csv(BASE_PATH+"stats.csv")
    stats.index = FEATURES

    x *= (stats['max']-stats['min']+0.000001)
    x += stats['min']
    x = np.around(x, 6)
    x_ *= (stats['max']-stats['min']+0.000001)
    x_ += stats['min']
    x_ = np.around(x_, 6)
    
    #pd.DataFrame(x).to_csv(BASE_PATH+"test_real.csv", header=FEATURES, index=None)
    #pd.DataFrame(x_).to_csv(BASE_PATH+"test_input.csv", header=FEATURES, index=None)