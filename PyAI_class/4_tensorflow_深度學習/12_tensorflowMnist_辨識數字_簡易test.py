from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def data_to_matrix(data):
    return np.reshape(data, (28, 28))


x = tf.placeholder(tf.float32, shape=[None, 784])       #x表示image輸入,欄位有28*28個
# Y = tf.placeholder(tf.float32, shape=[None, 10])        #y表示label輸出,欄位分別代表one-hot表示的數字0~9,共10欄

W = tf.Variable(tf.zeros([784,10]))                     #權重值w中的[]為[x的欄位數,最後y輸出的比數]
b = tf.Variable(tf.zeros([10]))                         #偏移值b中的[]為[最後y輸出的比數]

#模型公式y_=x*w=b
y_ = tf.matmul(x, W) + b

# 設定模型存放位置
model_path = "tmp/model.ckpt"           #.ckpt:為tensorflow的副檔名
saver = tf.train.Saver()

#開始之前都要初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # 載入保存的模型
    saver.restore(sess, model_path)

    prediction_result = sess.run(tf.argmax(y_, 1), feed_dict={x: mnist.test.images})

    #將要預測的數字畫出(人工辨識)
    matrix = data_to_matrix(mnist.test.images[0])
    plt.figure()
    plt.imshow(matrix)
    plt.show()

    # 第一張圖像的預測結果(電腦預測結果)
    print(prediction_result[0:1])
