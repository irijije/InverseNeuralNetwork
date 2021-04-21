import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from preprocess import load_dataset


os.environ["CUDA_VISIBLE_DEVICES"] = '1'

C = 10**-10
NN_MODEL_NAME = "models/nn.h5"
INN_MODEL_NAME = "models/inv_add_"+str(C)+".h5"
AE_ENCODER_NAME = "models/ae_encoder.h5"
TRAIN_DATA = "dataset/CICIDS2018_mal_train.csv"
TEST_DATA = "dataset/CICIDS2018_mal_test.csv"


class InverseNN_add:
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

        self.encoder = tf.keras.models.load_model(AE_ENCODER_NAME)

        self.nn_model = tf.keras.models.load_model(NN_MODEL_NAME)

        self.optimizer = tf.keras.optimizers.Adam(1e-4)

        self.C = C

    def loss(self, x, x_, y, y_):
        cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        generate_loss = tf.reduce_mean(tf.square(tf.abs(x - x_)))
        
        onehot_t = tf.one_hot(tf.zeros(tf.shape(y)[0], dtype=tf.int32), depth=15)
        classify_loss1 = cross_entropy(onehot_t, y_)

        y_ = tf.multiply(y_, y)
        
        classify_loss2 = -cross_entropy(y, y_)

        classify_loss = classify_loss1 + classify_loss2
        
        total_loss = generate_loss + self.C*classify_loss

        return total_loss

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            y = self.nn_model(x, training=False)
            # max_y = tf.argmax(y, axis=1)
            # onehot_y = tf.reshape(tf.one_hot(max_y, depth=15), shape=[-1, 15])
            #x_f = self.encoder(x, training=False)
            #x_ = self.inn_model(tf.concat([onehot_y, x_f], axis=1), training=False)
            x_ = self.inn_model(y, training=False)
            y_ = self.nn_model(x_, training=False)

            loss = self.loss(x, x_, y, y_)

        gradients = tape.gradient(loss, self.inn_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.inn_model.trainable_variables))


    def train(self, C, epochs=5, filename=INN_MODEL_NAME):
        self.C = C
        for epoch in range(epochs):
            print("epoch: {} training".format(epoch))
            for batch in self.train_dataset:
                self.train_step(batch)
        self.inn_model.save(filename)


if __name__ == "__main__":
    inn = InverseNN_add()
    C = 2**-(20-20)
    model_name = "models/inv_add_"+str(0)+".h5"
    inn.train(0, 5, model_name)