'''
使用CNN卷積神經網路提升辨識準確度
共7層
可比較1_KerasMnistMLP_多層感知器神經網路.py
'''
import keras
from keras.datasets import mnist        #匯入MNIST資料
from keras.models import Sequential     #使用Sequential模型(多層神經元)
from keras.layers.core import Dense, Dropout, Activation        #與MLP相同-Dense:全連結層(神經元個數),Dropout:防止過擬合,Activation:激活函數
from keras.layers import Flatten, Conv2D, MaxPooling2D          #與MLP不一樣的地方!!!-Flatten:平坦層,Conv2D:CNN神經網路(卷積層),MaxPooling2D:池化層
from keras.callbacks import EarlyStopping, CSVLogger
import matplotlib.pyplot as plt

#step1:準備資料------------------------------------
batch_size = 128
num_classes = 10
epochs = 20

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#step2:整理資料------------------------------------
#1.正規化-將資料轉為keras可用的格式
x_train = x_train.reshape(60000,28,28,1).astype('float32')      #28x28x"1":1為通道的"灰階圖"
x_test = x_test.reshape(10000,28,28,1).astype('float32')

x_train /= 255
x_test /= 255

#2.把y變成了one-hot的形式
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# #印出形狀
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)


#step3:定義模型架構------------------------------------
#1.一層一層的去建立(搭建)神經層,產生新的模型
model = Sequential()

#2.加入第1層卷積層(Conv2D)
model.add(Conv2D(filters=10,                #filters:定義過濾器數量=10個(一張圖分10個區域的概念)
                 kernel_size=(3,3),         #kernel_size:卷積核的大小為3*3
                 padding='same',            #padding='same':將圖片外圍補0,避免重要的特徵被忽略
                 input_shape=(28,28,1),     #原始影像大小(由19/20行):只有第一層需要
                 activation='relu'))        #激活函數

#3.加入第1層-池化層(MaxPooling2D)
model.add(MaxPooling2D(pool_size=(2,2)))    #池化大小,2*2矩陣中找最大
#經過池化後影像大小=28/2=14

#4.加入第2層-卷積層(Conv2D)
model.add(Conv2D(filters=20,
                 kernel_size=(3,3),
                 padding='same',
                 activation='relu'))        #不用在input_shape=(28,28,1)

#5.加入第3層-池化層(MaxPooling2D)
model.add(MaxPooling2D(pool_size=(2,2)))
#經過池化後影像大小=14/2=7

#6.加入dropout隨機在訓練時關閉輸入單元與權重的影響,防止過擬合,0.2表示有2成輸入單元被關閉(丟棄)
model.add(Dropout(0.2))

#7.加入第5層-平坦層(Flatten):轉換為1維矩陣
model.add(Flatten())

#8.加入第6層-全連結層(Dense):
model.add(Dense(256,activation='relu'))         #256:神經元的數量(可自行調整)

model.add(Dropout(0.2))

#9.加入第7層-全連結層(Dense)/輸出層,10個神經元(分別對應0~9)
model.add(Dense(10,activation='softmax'))

#10.顯示出模型摘要
model.summary()


#step4:定義模型的優化方式------------------------------------
# metrics，裡面可以放入accuracy
model.compile(loss='categorical_crossentropy',          # 損失函數用交叉熵
              optimizer='adam',                         # 優化器用adam:Adagradrm具有特定參數學習率，根據訓練的更新"頻率"自行調整(参数接收的更新越多,更新越小)
              metrics=['accuracy'])                     # metrics，放入需要觀察的accuracy(準確度)/cost/score等

logger = CSVLogger('3_model_KerasMnistCNN.log')           #將epoch的訓練結果保存在csv文件中
estop = EarlyStopping(monitor='val_loss', patience=3)   #當監測值val_loss不再改善時，如發現損失沒有下降，則經過3個epoch後停止訓練


#step5:模型訓練(放入訓練資料x,y)------------------------------------
hist = model.fit(x_train, y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 verbose=2,
                 validation_split=0.2,          #validation_split:為0~1之間的浮點數,用來指定訓練集當作驗證集的比例
                 callbacks=[logger,estop])


#step6:進行模型評估------------------------------------
score = model.evaluate(x_test, y_test, verbose=0)
print('test loss:', score[0])
print('test acc:', score[1])


#step7:畫出學習曲線,保存模型參數------------------------------------
# 顯示acc訓練結果
accuracy = hist.history['acc']
val_accuracy = hist.history['val_acc']
plt.plot(range(len(accuracy)), accuracy, marker='.', label='accuracy(training data)')
plt.plot(range(len(val_accuracy)), val_accuracy, marker='.', label='val_accuracy(evaluation data)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# 顯示loss訓練結果
loss = hist.history['loss']
val_loss = hist.history['val_loss']
plt.plot(range(len(loss)), loss, marker='.', label='loss(training data)')
plt.plot(range(len(val_loss)), val_loss, marker='.', label='val_loss(evaluation data)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

#step7:存儲模型與權重------------------------------------
model.save('3_model_KerasMnistCNN.h5')

del model