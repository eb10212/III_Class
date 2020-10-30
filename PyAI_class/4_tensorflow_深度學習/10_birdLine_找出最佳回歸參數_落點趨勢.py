import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#1.隨機產生100個棲息座標x_data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 8
# plt.plot(x_data,y_data)
# plt.show()
#會是一個完整的線性圖案

# 2.假設設計100個偏移值(模擬真實情況),當作統計100點"實際飛鳥棲息"的"偏移值"
noise_data = np.random.normal(0.0, 0.5, 100).astype(np.float32)
# plt.hist(noise_data)
# plt.show()

# 落點公式(把偏移值也考慮進去)
y_data2 = x_data * 8 + noise_data

# 最後畫出飛鳥實際棲息的分布圖
# plt.plot(x_data, y_data, 'o', label='data: y_data=x_data*8 + noise_data')
# plt.plot(x_data, y_data2, 'o', label='data: y_data2=x_data*8 + noise_data')
# plt.legend()
# plt.show()


# 3.有了資料後,開始做數據的分析(訓練出1個模型-此題就是找出最佳回歸線y=ab+b中的a和b)-演算法設計
a = tf.Variable( tf.random_uniform([1], -1.0, 1.0 ) )       #初始化(第1個)權重值a：先隨機產生"一個"-1到1之間的值
b = tf.Variable( tf.zeros([1]) )                            #初始化權重值b ：產生"一個"0值

# 機器要學習的特徵模型y為預測值
y = a * x_data + b

#設定演算法
lost = tf.reduce_mean( tf.square ( y - y_data ) )                       #損失函數 lost=Σ(y - y_data)^2/n
optimizer = tf.train.GradientDescentOptimizer (learning_rate = 0.5)     #優化方法: 梯度下降法(優化器)
train = optimizer.minimize ( lost )                                     #找出最小的損失



#執行階段
init = tf.global_variables_initializer()        #初始化所有變數((很重要~~))

with tf.Session() as sess:
    sess.run(init)
    loss_list = []
    for step in range(100):     #訓練100次
        sess.run(train)         #將每次訓練的誤差值收集起來
        loss_list.append(sess.run(lost))
        if step % 10 == 0:      #只是不想每一筆都要show出來看
            print(step, sess.run(a), sess.run(b))           # 每10次把當時的權重值a,b印出來
            plt.plot(x_data, sess.run(a)* x_data+ sess.run(b),label="model train step={}".format(step))     # 每10次把迴歸線畫出來
    print('所有loss_list',loss_list)

    # for step in range(10):     #訓練10次
    #     sess.run(train)         #將每次訓練的誤差值收集起來
    #     loss_list.append(sess.run(lost))
    #
    #     print(step, sess.run(a), sess.run(b))
    #     plt.plot(x_data, sess.run(a)* x_data+ sess.run(b),label="model train step={}".format(step))



    # 把原始的候鳥落點分布圖畫出來
    plt.plot(x_data, y_data2, 'o', label='data: y_data2=x_data*8 + noise_data')
    plt.legend()
    plt.show()
    # 將誤差值畫出來
    plt.plot(loss_list, lw=2)
    plt.show()
