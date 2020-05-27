import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as score

from preprocess import load_dataset
from metadata import LABELS


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
MODEL_NAME = "models/nn.h5"


class NN:
    def __init__(self):
        self.train_x, self.train_y = load_dataset("dataset/CICIDS2018_train.csv")
        self.test_x, self.test_y = load_dataset("dataset/CICIDS2018_test.csv")

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(78,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(15, activation='softmax')
        ])

        self.model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'],
        )

    def show_result(self, hist):
        _, loss_ax = plt.subplots()
        acc_ax = loss_ax.twinx()
        loss_ax.plot(hist.history['loss'], 'r', label='train loss')
        acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
        loss_ax.plot(hist.history['val_loss'], 'y', label='val loss')
        acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
        loss_ax.set_xlabel('epoch')
        loss_ax.set_ylabel('loss')
        acc_ax.set_ylabel('accuray')
        loss_ax.legend(loc='upper left')
        acc_ax.legend(loc='lower left')
        plt.show()

    def train(self):
        hist = self.model.fit(self.train_x, self.train_y, validation_split=0.2, batch_size=200, epochs=10)
        self.model.save(MODEL_NAME)
        self.show_result(hist)
    
    def test(self, filename=None):
        if filename:
            self.model = tf.keras.models.load_model(filename)
        test_y_ = self.model(np.array(self.test_x), training=False)
        test_y_ = tf.argmax(test_y_, 1)
        precision, recall, f1, _ = score(self.test_y, test_y_, zero_division=1)

        ax = plt.subplot(111)
        plt.title("scores")
        ind = np.arange(15)
        w = 0.3
        ax([])
        ax.bar(ind, precision, width=w, color='g', label='precision')
        ax.bar(ind+w, recall, width=w, color='b', label='recall')
        ax.bar(ind+2*w, f1, width=w, color='r', label='f1')
        ax.set_xlabel('kind of attacks')
        ax.set_ylabel('scores')
        ax.set_xticks(ind+w)
        ax.set_xticklabels(LABELS)
        ax.legend(loc='upper right')
        ax.autoscale(tight=True)
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.show()
    

if __name__ == "__main__":
    nn = NN()
    nn.train()
    nn.test(MODEL_NAME)