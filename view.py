import matplotlib.colors
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

Activations = [tf.nn.sigmoid, tf.nn.softmax, tf.nn.elu, tf.nn.softplus, tf.nn.relu6, tf.nn.tanh]

datas = []
for activation in Activations:
    # 之前的模型...etc
    hidden_layer1 = activation(tf.matmul(input_features, hidden_W) + hidden_b)
    # 训练部分
    with ...:
        #计算cost
        datas.append(sess.run(cost))
#图像显示
