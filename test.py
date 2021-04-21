import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from preprocess import load_dataset
from InverseNN import InverseNN
from InverseNN_add import InverseNN_add


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
NN_MODEL_NAME = "models/nn.h5"
AE_ENCODER_NAME = "models/ae_encoder.h5"
INN_MODEL_NAME = "models/inv_small_test.h5"
TEST_DATA = "dataset/CICIDS2018_mal_test.csv"

BLACKBOX = True


def show_test():
    x, _ = load_dataset("dataset/test.csv")
    x = np.array(x)
    nn_model = tf.keras.models.load_model(NN_MODEL_NAME)
    encoder = tf.keras.models.load_model(AE_ENCODER_NAME)
    inn_model = tf.keras.models.load_model(INN_MODEL_NAME)

    if BLACKBOX:
        y = nn_model(x, training=False)
        max_y = tf.argmax(y, axis=1)
        onehot_y = tf.reshape(tf.one_hot(max_y, depth=15), shape=[-1, 15])
        x_f = encoder(x, training=False)
        x_ = inn_model(tf.concat([onehot_y, x_f], axis=1), training=False)
        y_ = nn_model(x_, training=False)
    else:
        y = nn_model(x, training=False)
        x_ = inn_model(y, training=False)
        y_ = nn_model(x_, training=False)

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

def test(filename=INN_MODEL_NAME):
    x, _ = load_dataset(TEST_DATA)
    x = np.array(x)
    nn_model = tf.keras.models.load_model(NN_MODEL_NAME)
    encoder = tf.keras.models.load_model(AE_ENCODER_NAME)
    inn_model = tf.keras.models.load_model(filename)

    if BLACKBOX:
        y = nn_model(x, training=False)
        # max_y = tf.argmax(y, axis=1)
        # onehot_y = tf.reshape(tf.one_hot(max_y, depth=15), shape=[-1, 15])
        # x_f = encoder(x, training=False)
        #x_ = inn_model(tf.concat([onehot_y, x_f], axis=1), training=False)
        x_ = inn_model(y, training=False)
        y_ = nn_model(x_, training=False)
    else:
        y = nn_model(x, training=False)
        x_ = inn_model(y, training=False)
        y_ = nn_model(x_, training=False)
    
    y = tf.math.argmax(y, axis=1)
    y_ = tf.math.argmax(y_, axis=1)

    #print(y)
    #print(y_)

    generate_loss = tf.reduce_mean(tf.square(tf.abs(x - x_)))

    acc = tf.keras.metrics.Accuracy()
    acc.update_state(y_, tf.zeros_like(y))
    print("attack success probability: {}".format(acc.result().numpy()))

    return generate_loss, acc.result().numpy()


if __name__ == "__main__":
    losses = []
    success_rate = []
    for i in range(21):
        C = 2**-(20-i)
        if BLACKBOX:
            model_name = "models/inv_add_my2"+str(C)+".h5"
            inn = InverseNN_add()
        else:
            model_name = "models/inv_cw_"+str(C)+".h5"
            inn = InverseNN()
        print("model {}".format(model_name))
        inn.train(C, 3, model_name)
        loss, rate = test(model_name)
        losses.append(loss)
        success_rate.append(rate)

    superscript_map = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
    "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻"}

    trans = str.maketrans(''.join(superscript_map.keys()), ''.join(superscript_map.values()))

    plt.title("Evasion Attack with Autoencoder")
    loss_ax = plt.gca()
    xs = ["2"+("-"+str(20-x)).translate(trans) for x in range(21)]
    rate_ax = loss_ax.twinx()
    rate_ax.set_ylim(0.0, 1.05)
    loss_ax.set_ylim(0.0, 0.0105)
    rate_ax.set_yticks(np.arange(0, 1.1, 0.1))
    loss_ax.set_yticks(np.arange(0, 0.011, 0.001))
    l1, = rate_ax.plot(xs, success_rate, 'C0', marker='.')
    l2, = loss_ax.plot(xs, losses, 'C1', marker='.')
    loss_ax.set_xlabel('C values')
    rate_ax.set_ylabel('attack success rate')
    loss_ax.set_ylabel('generate loss')
    plt.legend([l1, l2], ["rate", "loss"], loc='upper left')
    plt.grid(b=True, which='major', linestyle='--')
    plt.savefig('c_values.png')
    plt.show()

    #test()
    #show_test()