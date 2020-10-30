import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import pandas as pd

#step1:讀取訓練過的模型&資料集-------------------------------------------
model = load_model('3_model_KerasMnistCNN.h5')
model.summary()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

label_y_test = y_test

# 圖片正規劃
x_test = x_test.reshape(10000,28,28,1).astype('float32')
x_test /= 255
y_test = keras.utils.to_categorical(y_test,num_classes=10)

#step2:驗證模型,使用測試(test中的資料)-------------------------------------------
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

predictions = model.predict_classes(x_test)
print(x_test.shape)
print(predictions)

# 顯示出前15個測試影像, 預測結果, 與原始答案
for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.title("pred.={} label={}".format(predictions[i],np.argmax(y_test[i])))
    plt.imshow(x_test[i].reshape(28, 28), cmap=None)
plt.show()


#step3:查看預測錯誤的部分-------------------------------------------
print("Error prediction:")
errorList = []
for i in range(len(predictions)):
    if predictions[i] != np.argmax(y_test[i]):
        print("Image[%d] : label=%d, but prediction=%d" % (i, np.argmax(y_test[i]), predictions[i]))
        errorList.append(i)
print("-----------------------")
print("total number of error prediction is %d" % len(errorList))
#印出比較表
print("%s\n" % pd.crosstab(label_y_test, predictions, rownames=['label'], colnames=['predict']))
