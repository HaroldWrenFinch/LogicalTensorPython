import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[4, 2], name='x')
y = tf.placeholder(tf.float32, shape=[4, 1], name='y')

weight1 = tf.Variable(tf.random_uniform([2, 6], -1, 1), name="hidden1")
weight2 = tf.Variable(tf.random_uniform([6, 1], -1, 1), name="hidden2")

bias1 = tf.Variable(tf.zeros([6]), name="Bias1")
bias2 = tf.Variable(tf.zeros([1]), name="Bias2")

MAX_ERROR = 0.01

with tf.name_scope("hidden_layer") as scope:
    hiddenCalculated = tf.sigmoid(tf.matmul(x, weight1) + bias1)

with tf.name_scope("output_layer") as scope:
    output = tf.sigmoid(tf.matmul(hiddenCalculated, weight2) + bias2)

with tf.name_scope("cost") as scope:
    cost = - tf.reduce_mean((y * tf.log(output)) + (1 - y) * tf.log(1.0 - output))

with tf.name_scope("train") as scope:
    trainStep = tf.train.GradientDescentOptimizer(MAX_ERROR).minimize(cost)

AND_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
AND_Y = [[0], [0], [0], [1]]

init = tf.global_variables_initializer()
sess = tf.Session()

# writer = tf.summary.FileWriter("./log/and_logs", sess.graph_def)

sess.run(init)

i = 0
score = 1

while score > MAX_ERROR:
    i = i + 1

    if i % 10000 == 0:
        sess.run(trainStep, feed_dict={x: AND_X, y: AND_Y})
        print('iteration: ', i)
        print('Output: ')
        print(sess.run(output, feed_dict={x: AND_X, y: AND_Y}))
        score = sess.run(cost, feed_dict={x: AND_X, y: AND_Y})
        print('cost: ', score)
