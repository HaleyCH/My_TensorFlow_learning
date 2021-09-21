import matplotlib.colors
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(39)
#tf.random.set_random_seed(39)
num_class = 5
diff = [[0, 2], [0, -2], [2, 0], [-2, 0], [2,2]]


def to_one_hot(class_num, i):
    return [1 if i == r else 0 for r in range(class_num)]


def make_data(size, diff):
    global num_class
    each_class_size = int(size / num_class)
    X0 = np.random.multivariate_normal(np.random.randn(2) + diff[0],
                                       np.eye(2), each_class_size)
    Y0 = np.zeros(each_class_size)
    for i in range(num_class - 1):
        tmp_X = np.random.multivariate_normal(
            np.random.randn(2) + diff[i + 1],
            np.eye(2), each_class_size)
        tmp_Y = np.full(each_class_size, i + 1)
        X0 = np.concatenate((X0, tmp_X))
        Y0 = np.concatenate((Y0, tmp_Y))
    Y = [to_one_hot(num_class, y) for y in Y0]
    # plt.plot(X0[:, 0], X0[:, 1], 'ro')
    # plt.legend()
    # plt.show()
    return X0, Y


# make_data(1000)
# print(to_one_hot(3, 1))
t_x, t_y = make_data(5000, diff)
# colors = ['r' if i[0] == 1 else 'b' for i in t_y]
# # print(colors)
# plt.scatter(t_x[:, 0], t_x[:, 1], c=colors)
# plt.show()

input_dim = 2
lab_dim = num_class
hidden_dim = 1024

input_features = tf.placeholder(tf.float32, [None, input_dim])
input_labels = tf.placeholder(tf.float32, [None, lab_dim])

hidden_W = tf.Variable(tf.random_normal([input_dim, hidden_dim]), name="h_weight")
hidden_b = tf.Variable(tf.zeros(hidden_dim), name="h_b")

W = tf.Variable(tf.random_normal([hidden_dim, lab_dim]), name="weight")
b = tf.Variable(tf.zeros(lab_dim), name="bias")

hidden_layer1 = tf.nn.softmax(tf.matmul(input_features, hidden_W) + hidden_b)

output = tf.nn.sigmoid(tf.matmul(hidden_layer1, W) + b)
o1 = tf.argmax(output, axis=1)  # 用于可视化

loss = tf.reduce_mean(tf.square(output - input_labels))
global_step = tf.Variable(0., trainable=False)
add_global = global_step.assign_add(1)
optimizer = tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(0.05, global_step, 1000, 0.05))
train = optimizer.minimize(loss)

maxEpochs = 50
minibitchSize = 100
display_step = 100

loss_data = []

with tf.Session() as sess:
    print("[#]Start...")
    sess.run(tf.global_variables_initializer())

    for epoch in range(maxEpochs):
        for i in range(np.int32(len(t_y) / minibitchSize)):
            x1 = t_x[i * minibitchSize:(i + 1) * minibitchSize]
            y1 = t_y[i * minibitchSize:(i + 1) * minibitchSize]
            _, loss_val, _ = sess.run([train, loss, add_global], feed_dict={input_features: x1, input_labels: y1})
            loss_data.append(loss_val)
            if epoch % display_step == 0:
                print("[+]Epoch:", epoch + 1, " time =", i + 1, " loss= ", loss_val)

    print("[!]W:", sess.run(W), " b:", sess.run(b))
    t_x, t_y = make_data(200, diff)
    print("[*]cost:",sess.run(tf.reduce_mean(sess.run(loss, feed_dict={input_features: t_x, input_labels: t_y}))))

    t_x, t_y = make_data(200, diff)
    xs1 = np.linspace(-7, 7, 200)
    xs2 = np.linspace(-7, 7, 200)
    xx, yy = np.meshgrid(xs1, xs2)

    c_p = np.zeros((200, 200))
    for i in range(200):
        for j in range(200):
            c_p[i, j] = sess.run(o1, feed_dict={input_features: [[xx[i, j], yy[i, j]]]})
            # print(c_p[i, j])

    cmap = matplotlib.colors.ListedColormap([matplotlib.colors.colorConverter.to_rgba('r', alpha=0.2),
                                             matplotlib.colors.colorConverter.to_rgba('b', alpha=0.2),
                                             matplotlib.colors.colorConverter.to_rgba('g', alpha=0.2),
                                             matplotlib.colors.colorConverter.to_rgba('c', alpha=0.2),
                                             matplotlib.colors.colorConverter.to_rgba('y', alpha=0.2),
                                             matplotlib.colors.colorConverter.to_rgba('m', alpha=0.2),
                                             ])
    plt.contourf(xx, yy, c_p, cmap=cmap)
    # plt.show()
    color_cast = ['r', 'b', 'g', 'c', 'y', 'm']
    colors = [color_cast[np.argmax(i, axis=0)] for i in t_y]
    # print(colors)
    plt.scatter(t_x[:, 0], t_x[:, 1], c=colors)
    # x = np.linspace(-10, 10, 300)
    # for i in range(num_class):
    #     y = -x * (sess.run(W)[0][i] / sess.run(W)[1][i]) - sess.run(b)[i] / sess.run(W)[1][i]
    #     plt.plot(x, y, label="Feature line" + str(i))
    #     plt.legend()

    plt.show()
