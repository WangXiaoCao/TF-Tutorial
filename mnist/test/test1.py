from tensorflow.examples.tutorials.mnist.input_data import *
import tensorflow as tf

# read data
mnist = read_data_sets("MNIST_data/", one_hot=True)

# create n* 784 placeholder(as the number of factor is 784)
x = tf.placeholder(tf.float32, [None, 784])

# create variables:weight and bias(the initial value is 0)
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# build the softmax model
y = tf.nn.softmax(tf.matmul(x, w) + b)

# create n* 10 placeholder which is used to place actual labels
y_ = tf.placeholder("float", [None, 10])

# compute the loss
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# training, learning rate is 0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# initial all variables
init = tf.initialize_all_variables()
# create a session
sess = tf.Session()
sess.run(init)

# do 1000 circulations of trainning, select 100 observations at each time.
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})













