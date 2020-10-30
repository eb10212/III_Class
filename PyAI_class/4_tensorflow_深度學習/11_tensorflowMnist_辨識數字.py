#step1:下載手寫數字辨識資料集-----------------------------
from tensorflow.examples.tutorials.mnist import input_data

#載入mnist資料(只需下載一次即可)-存在MNIST_data的資料夾中
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#one_hot=True:使用0或1來表示圖片,[0,1,2,3.....,9]
#                             [1,0,0,0......0]表示數字0的意思

# 讀取MNIST資料為Tuple形式, x_train為影像資料, y_train為標籤資料
# X shape (60,000 28x28), y shape (10,000, )

# # 檢視結構
# print('訓練資料集的圖像的結構{}'.format( mnist.train.images.shape))
# print('訓練資料集的圖像標註的結構{}'.format( mnist.train.labels.shape))
# print('訓練資料集的圖像的總數={}'.format( len( mnist.train.images)))
# print('訓練資料集的第一個圖像(張量)= ={}'.format( mnist.train.images[1]))
# print('訓練資料集的第一個圖像標註的張量顯示={}'.format( mnist.train.labels[1]))


#step2:將圖形轉為可視之矩陣-----------------------------
import matplotlib.pyplot as plt
import numpy as np

#將圖像資料轉換成28x28矩陣
def data_to_matrix(data):
    return np.reshape(data, (28, 28))
#轉換 訓練資料集的第一個圖像數組 為矩陣
matrix = data_to_matrix(mnist.train.images[1])

#畫出矩陣方法1
plt.figure(num='訓練資料集的第一個圖像數組的矩陣')
plt.imshow(matrix)
plt.title('the first image and label is {}'.format( np.argmax(mnist.train.labels[1])))
                                                    #np.argmax:取得數組裡面,值最大的位置
#畫出矩陣方法2
plt.matshow(matrix, cmap=plt.get_cmap('gray'))
plt.title("the first image and label is {}".format(np.argmax(mnist.train.labels[1])))

plt.show()


#step3:設定參數(資料準備)-----------------------------
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 784])       #x表示image輸入,欄位有28*28個
Y = tf.placeholder(tf.float32, shape=[None, 10])        #y表示label輸出,欄位分別代表one-hot表示的數字0~9,共10欄
#tf.placeholder:接收執行期間的數據,shape=[None, 784]:假設不知道要輸入幾筆資料(所以可先填none)

W = tf.Variable(tf.zeros([784,10]))                     #權重值w中的[]為[x的欄位數,最後y輸出的比數]
b = tf.Variable(tf.zeros([10]))                         #偏移值b中的[]為[最後y輸出的比數]

#模型公式y_=x*w=b
y_ = tf.matmul(x, W) + b


#step4:演算法設計-----------------------------
lr = 0.5            #學習率:需 <1
batch_size = 1000   #批量:一次訓練所有資料太花時間,分批訓練
epochs = 1000       #總訓練次數
#可自行調整以上3個參數值,觀察最後結果的accuracy正確率(上述為每次(每批)訓練1000個,共訓練1000次)
epoch_list=[]
accuracy_list=[]
loss_list=[]

#設定成本函數(選擇損失函數)與梯度下降演算法
loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_))    #labels:實際值,logits:預測值
train = tf.train.GradientDescentOptimizer(lr).minimize(loss)

#計算正確率accuracy
correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
#tf.argmax(vector, 1)：返回的是vector中的最大值的"索引号"
#如果vector是一个向量，那就返回一个值，如果是一个矩阵，那就返回一个向量，这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号。
cp = tf.cast(correct_prediction, tf.float32)                    #實際值y=輸出值y_,則cp=1,否則cp=0
accuracy = tf.reduce_mean(cp)                                   #cp的平均值


#step5:執行運算-----------------------------
# 設定模型存放位置
model_path = "tmp/model.ckpt"           #.ckpt:為tensorflow的副檔名
saver = tf.train.Saver()

#開始之前都要初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)       # 每次訓練會隨機輸入一個batch_size的數量做訓練
        _, accuracy_, loss_, cp_ = sess.run([train, accuracy, loss, cp], feed_dict={x: batch_x, Y: batch_y})
        epoch_list.append(epoch)
        accuracy_list.append(accuracy_)
        loss_list.append(loss_)

        if epoch % 100 == 0:        #不想每筆都記錄(每100筆記錄一次)
            # print(“cp_len={} cp={}”.format(len(cp_),cp_)) #查看一個訓練批次的cp值
            print("accuracy={} loss={} epochs={}".format(accuracy_, loss_,epoch))

            plt.subplot(1, 2, 1)
            plt.plot(epoch_list, accuracy_list, lw=2)
            plt.xlabel("epoch")
            plt.ylabel("accuracy ")
            plt.title("train set: lr={} batch_size={} epochs={}".format(lr, batch_size, epochs))

            plt.subplot(1, 2, 2)
            plt.plot(epoch_list, loss_list, lw=2)
            plt.xlabel("epoch")
            plt.ylabel("loss ")
            plt.title("train set: lr={} batch_size={} epochs={}".format(lr, batch_size, epochs))
            plt.show()

    print("訓練結束!!")

    # 將模型保存在指定的位置
    save_path = saver.save(sess, model_path)
    print("模型保存在: {}".format(save_path))


# step6:評估模型-----------------------------
    # 1.在測試集上的準確率
    accu_test = sess.run(accuracy, feed_dict={x: mnist.test.images,Y: mnist.test.labels})
    print("Test Accuracy:", accu_test)

    # 2.在驗證集上的準確率
    accu_validation = sess.run(accuracy, feed_dict={x: mnist.validation.images,Y: mnist.validation.labels})
    print("valid Accuracy:", accu_validation)

    # 3.訓練集上的準確率
    accu_train = sess.run(accuracy, feed_dict={x: mnist.train.images,Y: mnist.train.labels})
    print("train Accuracy:", accu_train)


#step7:隨機測試一張圖,使用訓練的模型-----------------------------
# 載入保存的模型
with tf.Session() as sess:
    sess.run(init)
    # 載入保存的模型
    saver.restore(sess, model_path)

    # # 評估在測試集上的準確率(與118行結果一致)
    # print("Test Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images,Y: mnist.test.labels}))

    # 使用訓練好的模型來預測測試集裡圖像的數字
    prediction_result = sess.run(tf.argmax(y_, 1), feed_dict={x: mnist.test.images})
    # print(prediction_result)      #辨識出來的數字排名

    # 畫出測試集的第一張圖像(方便人工辨識)
    matrix = data_to_matrix(mnist.test.images[0])
    plt.figure()
    plt.imshow(matrix)
    plt.show()

    # 第一張圖像的預測結果(模型預測結果)
    print(prediction_result[0:1])       #辨識出來的數字排名"第1"
