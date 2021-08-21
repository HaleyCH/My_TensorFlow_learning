import matplotlib.colors
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(39)


def to_one_hot(class_num, i):
    return [1 if i == r else 0 for r in range(class_num)]


def make_data(size):
    num_class = 2
    each_class_size = int(size / num_class)
    X0 = np.random.multivariate_normal(np.random.randn(2), np.eye(2), each_class_size)
    X1 = np.random.multivariate_normal(np.random.randn(2) + [-1, -2], np.eye(2), each_class_size)
    Y0 = np.zeros(each_class_size)
    Y1 = np.ones(each_class_size)
    X0 = np.concatenate((X0, X1))
    Y0 = np.concatenate((Y0, Y1))
    Y = [to_one_hot(num_class, y) for y in Y0]

    # plt.plot(X0[:, 0], X0[:, 1], 'ro')
    # plt.legend()
    # plt.show()
    return X0, Y


# make_data(1000)
# print(to_one_hot(3, 1))
t_x, t_y = make_data(2000)
# colors = ['r' if i[0] == 1 else 'b' for i in t_y]
# # print(colors)
# plt.scatter(t_x[:, 0], t_x[:, 1], c=colors)
# plt.show()

input_dim = 2
lab_dim = 2

input_features = tf.placeholder(tf.float32, [None, input_dim])
input_labels = tf.placeholder(tf.float32, [None, lab_dim])

W = tf.Variable(tf.random_normal([input_dim, lab_dim]), name="weight")
b = tf.Variable(tf.zeros(lab_dim), name="bias")

output = tf.nn.softmax(tf.matmul(input_features, W) + b)
o1 = tf.argmax(output, axis=1)  # 用于可视化

loss = tf.reduce_mean(tf.square(output - input_labels))
optimizer = tf.train.AdamOptimizer(0.05)
train = optimizer.minimize(loss)

maxEpochs = 50
minibitchSize = 25
display_step = 4

loss_data = []

with tf.Session() as sess:
    print("[#]Start...")
    sess.run(tf.global_variables_initializer())

    for epoch in range(maxEpochs):
        for i in range(np.int32(len(t_y) / minibitchSize)):
            x1 = t_x[i * minibitchSize:(i + 1) * minibitchSize]
            y1 = t_y[i * minibitchSize:(i + 1) * minibitchSize]
            _, loss_val = sess.run([train, loss], feed_dict={input_features: x1, input_labels: y1})
            loss_data.append(loss_val)
            if epoch % display_step == 0:
                print("[+]Epoch:", epoch + 1, " time =", i + 1, " loss= ", loss_val)

    print("[!]W:", sess.run(W), " b:", sess.run(b))
    t_x, t_y = make_data(200)
    xs1 = np.linspace(-7, 7, 200)
    xs2 = np.linspace(-7, 7, 200)
    xx, yy = np.meshgrid(xs1, xs2)

    c_p = np.zeros((200, 200))
    for i in range(200):
        for j in range(200):
            c_p[i, j] = sess.run(o1, feed_dict={input_features: [[xx[i, j], yy[i, j]]]})
            # print(c_p[i, j])

    cmap = matplotlib.colors.ListedColormap([matplotlib.colors.colorConverter.to_rgba('r', alpha=0.2),
                                             matplotlib.colors.colorConverter.to_rgba('b', alpha=0.2)])
    plt.contourf(xx, yy, c_p, cmap=cmap)
    # plt.show()
    colors = ['r' if i[0] == 1 else 'b' for i in t_y]
    # print(colors)
    plt.scatter(t_x[:, 0], t_x[:, 1], c=colors)
    x = np.linspace(-10, 10, 300)
    y = -x * (sess.run(W)[0][0] / sess.run(W)[1][0]) - sess.run(b)[0] / sess.run(W)[1][0]
    plt.plot(x, y, label="Feature line1")
    plt.legend()
    y = -x * (sess.run(W)[0][1] / sess.run(W)[1][1]) - sess.run(b)[1] / sess.run(W)[1][1]
    plt.plot(x, y, label="Feature line2")
    plt.legend()
    plt.show()
