import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
MODEL_NAME = "models/small.h5"

class NN:
    def __init__(self):
        (self.train_x, self.train_y), (self.test_x, self.test_y) = self.load_dataset("dataset/CICIDS2018_small.csv")

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(78,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(15, activation='softmax')
        ])
        
        self.model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
        )

    def train(self):
        self.model.fit(self.train_x, self.train_y, batch_size=200, epochs=5)
        self.model.save(MODEL_NAME)
    
    def test(self, n_test_traffic=1):
        model = tf.keras.models.load_model(MODEL_NAME)
        _, test_acc = model.evaluate(self.test_x, self.test_y, verbose=2)
        print('Test accuracy:', test_acc)

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
    nn = NN()
    nn.train()