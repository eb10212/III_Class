'''
VGG16:使用ImageNet的預訓練模型(圖片集)
      13個3*3的卷積層+13個2*2的池化層+3個全連結層,共16層
'''
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
                        #.vgg16:可透用成其他keras的預訓練模型(如ResNet50)
                                     #VGG16:根據使用的模型後"大寫",preprocess_input:前處理(轉成VGG16可讀取的格式),decode_predictions:預測圖片使用
from keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#step1:載入VGG16------------------------------------
model = VGG16(weights='imagenet')           #weights='imagenet':權重為ImageNet
model.summary()                             #顯示出模型摘要

#step2:定義預測圖片的函示------------------------------------
def predict(filename, rank):
    img = image.load_img(filename, target_size=(224, 224))      #將每張圖片(不管大小)載入,並轉成224*224的大小(為vgg16所使用的訓練大小)
    x = image.img_to_array(img)                                 #將圖片轉為數字矩陣
    print(x.shape)                                              #(244,244,3):3為彩色圖片--為單一張圖片

    #在 x array的第0維(索引值0的位置),新增一個資料
    x = np.expand_dims(x, axis=0)                               #np.expand_dims:用於擴充維度
    print(x.shape)                                              #(1,244,244,3),最外層的舵加一層框[]:表示每張圖片皆會是獨立一個,[圖片a,圖片b,....]

    preds = model.predict(preprocess_input(x))                  #轉換成VGG16可以讀的格式(preprocess_input:前處理工具,vgg16的工具)
    print(preds.shape)                                          #(1,1000):ImageNet中的分類共有1000種

    results = decode_predictions(preds,top=rank)[0]             #decode_predictions:預測圖片(vgg16的工具),top=rank:取前幾名排序
                                                                #results=[[],none]的形式,目前只有一張圖片取[0]
    return results

#step3:預測圖片------------------------------------
filename = "vgg16TestPic/1.jpg"         #先在資料夾中加入要辨識的圖片

plt.figure()
im = Image.open(filename)
im_list = np.asarray(im)
plt.title("predict")
plt.axis("off")
plt.imshow(im_list)
plt.show()

results = predict(filename, 10)         #show出前10筆預測結果
for result in results:
    print(result)
