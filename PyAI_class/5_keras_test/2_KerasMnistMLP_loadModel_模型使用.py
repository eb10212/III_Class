'''
使用訓練好的模型,進行圖像辨識
'''

import keras
from keras.datasets import mnist            #匯入MNIST資料
import matplotlib.pyplot as plt
from keras.models import load_model         #載入模型
import numpy as np
import pandas as pd

#step1:讀取訓練過的模型&資料集-------------------------------------------
model = load_model('1_model_KerasMnistMLP.h5')    #載入模型
model.summary()                                   #顯示出模型摘要

# 讀取MNISt的資料(為Tuple形式:x為影像資料,y為標籤資料(x shape=(60,000 28x28),y shape=(10,000, ) )
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#取用測試集(test)的資料作後續的"驗證"

keep_y_test = y_test        #保持原始數據,與之後混淆矩陣做比較用

x_test = x_test.reshape(10000, 784).astype('float32')       #mnist的測試集的資料量
x_test /= 255                                               #正規化
y_test = keras.utils.to_categorical(y_test, num_classes=10) #數字0~9,轉成one-hot形式


#step2:驗證模型,使用測試(test中的資料)-------------------------------------------
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#show出test資料集中的預測結果
predictions = model.predict_classes(x_test)
print(predictions.shape)                    #共10000筆資料
print("All predictions: ",predictions)      #顯示每一筆predictions的結果(為list的格式)

# 顯示出前15個測試影像, 預測結果, 與原始答案
for i in range(15):
    plt.subplot(3,5,i+1)                        #總共3*5=15張圖
    plt.title("pred.={} label={}".format(predictions[i],np.argmax(y_test[i])))
    plt.imshow(x_test[i].reshape(28, 28))       #印出實際的圖案
plt.show()


#step3:查看預測錯誤的部分-------------------------------------------
errorList = []
for i in range(len(predictions)):
    if predictions[i] != np.argmax(y_test[i]):
        print("Image[%d] : label=%d, but prediction=%d" % (i, np.argmax(y_test[i], axis=0), predictions[i]))
        errorList.append(i)
print("-----------------------")
print("total number of error prediction is %d" % len(errorList))        #顯示錯誤筆數
print("-----------------------")
#印出比較表
print("%s\n" % pd.crosstab(keep_y_test, predictions, rownames=['label'], colnames=['predict']))

