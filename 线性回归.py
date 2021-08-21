from typing import Dict, List, Any

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

t_x = np.linspace(-1, 1, 100)
t_y = 2 * t_x + np.random.rand(*t_x.shape) * 0.3

# plt.plot(t_x, t_y, 'ro', label="Origin data")
# plt.legend()
# plt.show()

X = tf.placeholder("float")
Y = tf.placeholder("float")

w = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1], name="bias"))

z = tf.multiply(X, w) + b
cost = tf.reduce_min(tf.square(Y - z))
learningrate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learningrate).minimize(cost)

init = tf.global_variables_initializer()
training_epochs = 20
display_step = 2
with tf.Session() as sess:
    print("[#]Start training.")
    sess.run((init))
    plotdata = {"batchsize": [], "loss": []}
    for epoch in range(training_epochs):
        for (x, y) in zip(t_x, t_y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X: t_x, Y: t_y})
            print("[+]Epochs: ", epoch + 1, " cost= ", cost, " W= ", sess.run(w), " b= ", sess.run(b))
            if not (loss == "NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append("loss")
    print("[!]Finished.")
    print("[#]cost= ", sess.run(cost, feed_dict={X: t_x, Y: t_y}), " W= ", sess.run(w), " b= ", sess.run(b))
    # Graphic
    plt.plot(t_x, t_y, 'ro', label='Origin data')
    plt.plot(t_x, sess.run(w) * t_x + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
