import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from preprocess import load_dataset
from InverseNN import InverseNN


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
NN_MODEL_NAME = "models/nn.h5"
INN_MODEL_NAME = "models/inv_010.h5"


def show_test():
    x, _ = load_dataset("test.csv")
    nn_model = tf.keras.models.load_model(NN_MODEL_NAME)
    inn_model = tf.keras.models.load_model(INN_MODEL_NAME)

    y = nn_model(np.array(x), training=False)
    x_ = inn_model(y, training=False)
    y_ = nn_model(np.array(x_), training=False)

    print("\nx: {}".format(np.array2string(np.array(x), prefix="x: ",
            formatter={'float_kind':lambda x: "%7.4f" % x})))
    print("\ny: {}".format(np.array2string(np.array(y), prefix="y: ",
            formatter={'float_kind':lambda x: "%7.4f" % x})))
    print("\nx_: {}".format(np.array2string(np.array(x_), prefix="x_: ",
            formatter={'float_kind':lambda x: "%7.4f" % x})))
    print("\ny_: {}".format(np.array2string(np.array(y_), prefix="y_: ",
            formatter={'float_kind':lambda x: "%7.4f" % x})))

    y = np.argmax(y)
    y_ = np.argmax(y_)

    if y != y_:
        print("\nattack success y: {} y_: {} \n".format(y, y_))
    else:
        print("\nattack failed y: {} y_: {} \n".format(y, y_))

def test(filename):
    x, _ = load_dataset("dataset/CICIDS2018_mal_test.csv")
    nn_model = tf.keras.models.load_model(NN_MODEL_NAME)
    inn_model = tf.keras.models.load_model(filename)

    y = nn_model(np.array(x), training=False)
    x_ = inn_model(y, training=False)
    y_ = nn_model(x_, training=False)

    y = tf.math.argmax(y, axis=1)
    y_ = tf.math.argmax(y_, axis=1)

    acc = tf.keras.metrics.Accuracy()
    acc.update_state(y, y_)
    print("attack success probability: {}".format(1-acc.result().numpy()))

    return 1-acc.result().numpy()


if __name__ == "__main__":
    losses = []
    success_rate = []
    for i in range(10):
        model_name = "models/inv_0"+str(i)+".h5"
        print("model {} is training".format(model_name))
        inn = InverseNN()
        losses.append(inn.train(5, 0.1**(10-i), model_name))
        success_rate.append(test(model_name))

    _, rate_ax = plt.subplots()
    loss_ax = rate_ax.twinx()
    rate_ax.set_ylim(0.0, 1.0)
    loss_ax.set_ylim(0.0, 1.0)
    l1, = rate_ax.plot(["10^-"+str(10-x) for x in range(10)], success_rate, 'b')
    l2, = loss_ax.plot(["10^-"+str(10-x) for x in range(10)], losses, 'r')
    rate_ax.set_xlabel('C values')
    rate_ax.set_ylabel('attack success rate')
    loss_ax.set_ylabel('generate loss')
    plt.legend([l1, l2], ["rate", "loss"], loc='upper left')
    plt.savefig('c_values.png')
    plt.show()

    #test("models/inv_03.h5")
    #show_test()