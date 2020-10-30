'''
使用keras打造2層MLP神經網路-辨識MNIST手寫數字
可比較4_tensorflow_深度學習/11_和12_.py
'''

import keras
from keras.datasets import mnist        #匯入MNIST資料
from keras.models import Sequential                             #使用Sequential模型(多層神經元)
from keras.layers.core import Dense, Dropout, Activation        #每層神經元的相關功能"Dense:全連結層(神經元個數),Dropout:防止過擬合,Activation:激活函數
from keras.optimizers import RMSprop                            #優化器(梯度下降演算法)
from keras.callbacks import EarlyStopping, CSVLogger            #訓練神經網路中的相關功能(EarlyStopping:若有得到步錯的結果可提早停止訓練,CSVLogger:記錄訓練過程)
import matplotlib.pyplot as plt

#step1:準備資料------------------------------------
batch_size = 128        # 每一批次讀入128張資料(每次讀幾筆)
num_classes = 10        # 數字為0~9所以共10個類別(要分幾類)
epochs = 20             # 使用反向傳播法進行訓練，總共訓練20次(訓練回合數)

# 讀取MNIST資料為Tuple形式, x_train為影像資料, y_train為標籤資料
# X shape (60,000 28x28), y shape (10,000, )
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)  #(60000, 28, 28)      #kears所提供的x的訓練影樣張數(28*28像素)
print(y_train.shape)  #(60000,)
print(x_test.shape)   #(10000, 28, 28)
print(y_test.shape)   #(10000,)

# #先使用plt顯示錢6張影像
# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.title("image {}.".format(i))
#     plt.imshow(x_train[i].reshape(28, 28))
# plt.show()

#step2:整理資料------------------------------------
# 將輸入的資料正規劃(將資料轉為keras可用的格式)
x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')

# 輸入的x共有60,000*784個數值(像素數值=色階 皆為0到255之間,所以每個除以255來進行標準化(變為0~1之間)
x_train /= 255
x_test /= 255

#把lebel變成one-hot的形式
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# # 印出形狀
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)


#step3:定義模型架構------------------------------------
#1.一層一層的去建立(搭建)神經層,產生新的模型
model = Sequential()

#2.加入第1層-隱藏層(全連接層Dense:神經元),輸入為任意筆數的784維資料(28*28), 256個神經元(可自行調整,只要符合2的n次方的數及可)
model.add(Dense(input_dim=784,
                units=256,                      #神經元的數量(同上述,可自行調整)
                kernel_initializer='normal',    #權重值的初始化
                bias_initializer='zeros',       #權重值的初始化
                activation='relu',              #relu激勵函數(隱藏層)<-隱藏層老師建議的激活函數
                name='hidden1'                  #此層的名稱
                ))

#3.加入dropout隨機在訓練時關閉輸入單元與權重的影響,防止過擬合,0.2表示有2成輸入單元被關閉(丟棄)
model.add(Dropout(0.2))

#4.加入第2層-隱藏層(全連接層), 256個神經元
model.add(Dense(units=256,
                kernel_initializer='normal',
                bias_initializer='zeros',
                activation='relu',
                name='hidden2'
                ))

model.add(Dropout(0.2))

#5.加入第3層-輸出層(亦為全連接層), 10個神經元(分別對應0~9)
model.add(Dense(units=10,
                kernel_initializer='normal',
                bias_initializer='zeros',
                activation='softmax',             #softmax激勵函數(輸出層)<-輸出層老師建議的激活函數
                name='output'
                ))
#6.顯示出模型摘要
model.summary()


#step4:定義模型的優化方式------------------------------------
model.compile(loss='categorical_crossentropy',      # 損失函數用交叉熵
              optimizer=RMSprop(),                  # 優化器用RMSprop(梯度下降法)
              metrics=['accuracy'])                 # metrics，放入需要觀察的accuracy(準確度)/cost/score

logger = CSVLogger('1_model_KerasMnistMLP.log')             #將每批次訓練的結果(log)紀錄在csv文件中
estop = EarlyStopping(monitor='val_loss', patience=3)       #當監測值val_loss不再改善時，如發現損失沒有下降，則經過3次(epoch)後停止訓練


#step5:模型訓練(放入訓練資料x,y)------------------------------------
hist = model.fit(x_train, y_train,          #x:圖片特徵矩陣,y:圖片標籤(真實代表的數字one-hot)
                 batch_size=batch_size,     #第14行先宣告的變數=128,一回合抓幾張
                 epochs=epochs,             #第16行先宣告的變數=10,共幾回合
                 verbose=1,                 #verbose:日誌顯示=1為輸出進度條記錄(預設)/0為不在標準輸出流輸出日誌信息/2為每個epoch輸出一行記錄
                 validation_split=0.1,      #validation_split：0~1之間的浮點數，用來指定訓練集當作驗證集的比例。
                 callbacks=[logger,estop])  #96/97行

#step6:進行模型評估------------------------------------
score = model.evaluate(x_test, y_test, verbose=0)       #放入的x,y為”測試“的資料集,且verbose=0(不需輸入log訊息)
print('score',score)
print('test loss:', score[0])
print('test acc:', score[1])


#step7:畫出學習曲線,保存模型參數------------------------------------
# 顯示acc學習結果
accuracy = hist.history['acc']
val_accuracy = hist.history['val_acc']
plt.plot(range(len(accuracy)), accuracy, marker='.', label='accuracy(training data)')
plt.plot(range(len(val_accuracy)), val_accuracy, marker='.', label='val_accuracy(evaluation data)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# 顯示loss學習結果
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
model.save('1_model_KerasMnistMLP.h5')        #.h5為kears的副檔名

del model      #已將模型存成檔案,所以就可以釋放記憶體容量(清除所佔記憶體)
