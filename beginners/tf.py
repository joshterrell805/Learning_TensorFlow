import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x_train, y_train = mnist.train.images, mnist.train.labels
# first dimension is rows, second is the pixels for x, classes for y
assert len(x_train.shape) == 2, len(y_train.shape) == 2

nrow_train = x_train.shape[0]
assert y_train.shape[0] == nrow_train

npixels = x_train.shape[1]
nclasses = y_train.shape[1]

tf_x = tf.placeholder(tf.float32, [None, npixels])
tf_y = tf.placeholder(tf.float32, [None, nclasses])

tf_W = tf.Variable(tf.zeros([npixels, nclasses]))
tf_b = tf.Variable(tf.zeros([nclasses]))
tf_y_pred = tf.nn.softmax(tf.matmul(tf_x, tf_W) + tf_b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf_y * tf.log(tf_y_pred), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    indexes = np.random.choice(nrow_train, replace=True, size=1000)
    batch_xs, batch_ys = x_train[indexes], y_train[indexes]
    sess.run(train_step, feed_dict={tf_x: batch_xs, tf_y: batch_ys})

correct_prediction = tf.equal(tf.argmax(tf_y, 1), tf.argmax(tf_y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={tf_x: mnist.test.images, tf_y: mnist.test.labels}))
