'''
ResNet50:解決退化問題,共50層
'''
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, decode_predictions
import glob

#step1:讀取圖片------------------------------------
file_path = 'vgg16TestPic/'
files = glob.glob(file_path + '*.jpg')

imgs = []
for file in files:
    img = image.load_img(file, target_size=(224, 224))
    arr_img = image.img_to_array(img)
    arr_img = np.expand_dims(arr_img, axis=0)
imgs.append(arr_img)                        #把圖片數組加到串列裡

x = np.concatenate([x for x in imgs])       #把所有圖片數組合併在一起
                                            #concatenate:把shape為(0,224,224,3)的每張圖片打包成shape為(batch,224,224,3)實現批次預測或批次訓練

#step2:ResNet50------------------------------------
model = ResNet50(weights='imagenet')
model.summary()

#批次預測
results = decode_predictions( model.predict(x), top=1)
i=0
for result in results:
    #印出檔名 與 辨識結果
    print(files[i], result)
    i+=1
