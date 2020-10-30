import tensorflow as tf
import matplotlib.pyplot as plt

#截斷的常態分佈(範圍不超過2倍的標準差)
n = 5000000
A = tf.truncated_normal([n,])
# tf.truncated_normal([列(可略),行],平均值,標準差)

B = tf.random_normal([n,])
with tf.Session() as sess:
    a, b = sess.run([A, B])
    plt.hist(b, 100, (-5, 5));
    plt.show()
    plt.hist(a, 100, (-5, 5));
    plt.show()
