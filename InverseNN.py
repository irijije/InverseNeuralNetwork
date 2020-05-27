import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from preprocess import load_dataset


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
NN_MODEL_NAME = "models/nn.h5"
INN_MODEL_NAME = "models/inv_001.h5"
BUFFER_SIZE = 60000
BATCH_SIZE = 256


class InverseNN:
    def __init__(self):
        self.train_x, self.train_y = load_dataset("dataset/CICIDS2018_mal_train.csv")
        self.test_x, self.test_y = load_dataset("dataset/CICIDS2018_mal_test.csv")

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
        
        max_y = tf.argmax(y, axis=1)
        onehot_y = tf.reshape(tf.one_hot(max_y, depth=15), shape=[-1, 15])
        unpacked_y = tf.unstack(onehot_y, axis=1)
        unpacked_y[0] = tf.ones(tf.shape(max_y)[0])
        packed_y = tf.transpose(tf.stack(unpacked_y, axis=0))
        y_ = tf.abs(y_ - packed_y)
        
        # untargeted
        # y_ = 1-y_

        classify_loss = cross_entropy(y, y_)
        
        total_loss = generate_loss + C*classify_loss

        return total_loss, generate_loss

    @tf.function
    def train_step(self, x, C):
        with tf.GradientTape() as tape:
            y = self.nn_model(x)
            x_ = self.inn_model(y)
            y_ = self.nn_model(x_)

            loss, generate_loss = self.loss(x, x_, y, y_, C)

        gradients = tape.gradient(loss, self.inn_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.inn_model.trainable_variables))

        return generate_loss

    def train(self, epochs=5, C=0.1, filename=INN_MODEL_NAME):
        losses = []
        for epoch in range(epochs):
            print("epoch: {} training".format(epoch))
            for batch in self.train_dataset:
                loss = self.train_step(batch, C)
            losses.append(loss)
        print("loss: ", end="")
        tf.print(loss)
        self.inn_model.save(filename)

        return np.array(loss)


if __name__ == "__main__":
    inn = InverseNN()
    loss = inn.train(5, 0.1**5, "models/inv_03.h5")
    print("loss: {}".format(loss))