import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#step1:準備資料------------------------------------------
xs = np.arange(0, 3, 0.01).reshape(-1,1)
print(xs)
print('xs',xs.shape)
noise_data = 0.1*(2*np.random.normal(0, 2, 300).astype(np.float32)-1).reshape(-1,1)
ys = xs*xs + xs + 1 + noise_data
print('ys',ys.shape)
plt.title("curve")
plt.plot(xs,ys)
plt.show()

#step2:設計有中間層的神經網路------------------------------------------
x = tf.placeholder(tf.float32,[None,1])
y_ = tf.placeholder(tf.float32,[None,1])
print(x, y_)
w1 = tf.get_variable("w1",initializer=tf.random_normal([1,32]))
w2 = tf.get_variable("w2",initializer=tf.random_normal([32,1]))
b1 = tf.get_variable("b1",initializer=tf.zeros([1,32]))
b2 = tf.get_variable("b2",initializer=tf.zeros([1,1]))
print(w1, w2, b1, b2)
l1 = tf.matmul(x, w1) + b1
print(l1)
l1 = tf.nn.relu(l1)
print(l1)
y = tf.matmul(l1, w2) + b2
print(y)

#step3:演算法設計------------------------------------------
# 損失函數 lost
loss = tf.reduce_mean( tf.square ( y - y_ ) )
#優化方法: 梯度下降法
optimizer = tf.train.GradientDescentOptimizer ( learning_rate = 0.01 )
#找出最小的損失
train = optimizer.minimize ( loss )

#step4:訓練模型------------------------------------------
init = tf.global_variables_initializer()
#開始訓練模型
with tf.Session() as sess:
    sess.run(init)
    loss_list = []
    for e in range(1001):
        loss_val, _ = sess.run([loss, train], {x: xs, y_: ys})

        if (e % 200 == 0):
            print("%d steps, loss is %f" % (e, loss_val))
            loss_list.append(loss_val)
            ys_pre = sess.run(y, {x: xs})
            #顯示預測結果
            plt.title("curve")
            plt.plot(xs,ys_pre,label='model train step={}'.format(e))

            plt.legend("ys","ys_pre")

plt.plot(xs,ys,label='model first')
plt.legend()
plt.show()

# 將誤差值畫出來
plt.title("loss")
plt.plot(loss_list, lw=2)
plt.show()