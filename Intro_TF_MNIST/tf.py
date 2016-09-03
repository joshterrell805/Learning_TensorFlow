def main_beginner():
    mnist = load_data()

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

def main_expert():
    def create_weight_var(shape):
        """create a weight variable initialize with some noise"""
        # truncated enforces the value to be within two standard deviations of the mean (0)
        initial_value = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial_value)

    def create_bias_var(shape):
        """
        create a bias variable initialize with a slightly positive value to help prevent
        "dead neurons"
        """
        initial_value = tf.constant(0.1, shape=shape)
        return tf.Variable(initial_value)
    
    def conv2d(x, W):
        """default our convolution to stepping one pixel in x and y"""
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    def max_pool_2x2(x):
        """pool using a 2x2 input and use the max algorithm"""
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # ph = place holder
    ph_x = tf.placeholder(tf.float32, [None, 28*28])
    ph_y = tf.placeholder(tf.float32, [None, 10])
    x_image = tf.reshape(ph_x, [-1, 28, 28, 1])

    # First Layer
    # 5x5 is the size of the local receptive field, 1 channel deep, 32 output units per 5x5 grid!
    # 32 low-level "features"?
    W_conv1 = create_weight_var([5, 5, 1, 32])
    # one bias per output of the first layer
    b_conv1 = create_bias_var([32])

    # relu does max(x, 0)
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # max_pool takes the maximum value in every 2x2 grid in the input and maps it to 
    # 1 pixel in the output
    h_pool1 = max_pool_2x2(h_conv1)

    # Second Layer
    W_conv2 = create_weight_var([5, 5, 32, 64])
    b_conv2 = create_bias_var([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Third Layer
    # densly (fully) connected 7x7x64 (3136) -> 1024
    W_fc1 = create_weight_var([7*7*64, 1024])
    b_fc1 = create_bias_var([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    # relu will make sure no value is less than 0
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Employ dropout to reduce overfitting
    # does it matter where this goes? why didn't we place it on the final layer?
    ph_keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, ph_keep_prob)

    # Final layer, softmax
    W_fc2 = create_weight_var([1024, 10])
    b_fc2 = create_bias_var([10])
    y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ph_y * tf.log(y_pred), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_predictions = tf.equal(tf.argmax(ph_y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.initialize_all_variables())

        # train
        mnist = load_data()
        for i in range(20*1000):
            batch = mnist.train.next_batch(50)
            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    ph_x: batch[0], ph_y: batch[1], ph_keep_prob: 1.})
                print('training accuracy on step {}: {:3.2f}%'.format(i, train_accuracy * 100))
            train_step.run(feed_dict={ph_x: batch[0], ph_y: batch[1], ph_keep_prob: 0.5})

        # evaluate on test set
        # this is a workaround evaluating `test_step` of the data at a time
        # sice my computer is running out of memory when evaluating the whole test set
        test_x, test_y = mnist.test.images, mnist.test.labels
        ntest = test_x.shape[0]
        test_step = 100
        n_correct = 0
        for start in range(0, ntest, test_step):
            test_batch_x = test_x[start:start+test_step]
            test_batch_y = test_y[start:start+test_step]
            c_preds = correct_predictions.eval(feed_dict={ph_x: test_batch_x, ph_y: test_batch_y,
                                     ph_keep_prob: 1.0})
            n_correct += sum(c_preds)
        test_accuracy = n_correct / ntest
        print('test accuracy: {:3.2f}%'.format(test_accuracy * 100))

def load_data():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    return mnist

def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('mode', nargs='?', choices=('softmax', 'cnn'), default='softmax',
                        help='run single softmax (beginner) or cnn (expert) version of code.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    import tensorflow as tf
    import numpy as np

    if args.mode == 'softmax':
        main_beginner()
    else:
        main_expert()
