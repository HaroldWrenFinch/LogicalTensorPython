import tensorflow as tf

sess = tf.Session()

hello = tf.constant("Hello L")
print(sess.run(hello))

a = tf.constant(1)
b = tf.constant(2)

c = sess.run(a * b)

print('a + b = {0}'.format(c))
