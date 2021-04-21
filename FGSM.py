import os
import numpy as np
import tensorflow as tf

from preprocess import load_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

TARGET_MODEL_NAME = "models/nn.h5"
SUR_MODEL_NAME = "models/nn_tiny.h5"
TEST_DATA = "dataset/CICIDS2018_tiny_mal_test.csv"

target_model = tf.keras.models.load_model(TARGET_MODEL_NAME)
sur_model = tf.keras.models.load_model(SUR_MODEL_NAME)

x, _ = load_dataset(TEST_DATA)
x = tf.convert_to_tensor(np.array(x))
target = 0
onehot_target = tf.one_hot(target, depth=15)

loss_object = tf.keras.losses.CategoricalCrossentropy()
with tf.GradientTape() as tape:
    tape.watch(x)
    prediction = sur_model(x, training=False)
    loss = -1*loss_object(onehot_target, prediction)

gradient = tape.gradient(loss, x)
perturbations = tf.sign(gradient)

x_ = x + 0.1*perturbations
x_ = tf.clip_by_value(x_, 0, 1)

y = tf.math.argmax(target_model(x, training=False), axis=1)
y_ = tf.math.argmax(target_model(x_, training=False), axis=1)
print(y)
print(y_)

acc = tf.keras.metrics.Accuracy()
acc.update_state(y_, tf.zeros_like(y_))
print("attack success probability: {}".format(acc.result().numpy()))


# epsilons = [0, 0.01, 0.1, 0.15]
# for i, eps in enumerate(epsilons):
#     x_ = x + eps*perturbations
#     x_ = tf.clip_by_value(x_, 0, 1)
#     print(model(x_))
#     print("\nx: {}".format(np.array2string(np.array(x), prefix="x: ",
#             formatter={'float_kind':lambda x: "%7.4f" % x})))
#     print("\nx_: {}".format(np.array2string(np.array(x_), prefix="x_: ",
#             formatter={'float_kind':lambda x_: "%7.4f" % x_})))