import os
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
NN_MODEL_NAME = "models/small.h5"
INN_MODEL_NAME = "models/small_inv.h5"


def norm(data, stats):
    #return ((data - stats['mean']) / (stats['std']+0.00001))
    return (data-stats['min']) / (stats['max']-stats['min']+0.00001)

def load_data(filepath):
    df_all = pd.read_csv("dataset/CICIDS2018_small.csv").replace([np.inf, -np.inf], np.nan).dropna().astype('float32')
    desc = df_all.describe().drop(['Label'], axis=1).transpose()
    
    df = pd.read_csv("test.csv").replace([np.inf, -np.inf], np.nan).dropna().astype('float32')
    labels = df.pop('Label')
    data = norm(df, desc)

    return data, labels
    
def test():
    x, _ = load_data("test.csv")
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

if __name__ == "__main__":
    test()

