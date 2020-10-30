import tensorflow as tf
import matplotlib.pyplot as plt

random_uniform = tf.random_uniform([100],0,)
#tf.random_uniform([列(可略),行],下限值,上限值)

with tf.Session() as sess:
    print(sess.run( random_uniform ))
    plt.hist(sess.run( random_uniform ))
    plt.show()

