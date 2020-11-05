'''
簡單使用ImageDataGenerator()函式:自動增加學習樣本!!!
為了解決資料及不夠時,模型過擬合的狀態
'''
import glob
import matplotlib.pyplot as plt
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

#step1:設定資料的相對路徑----------------------------------
train_dir = 'kagglecatdog/train'
test_dir = 'kagglecatdog/test'
validation_dir = 'kagglecatdog/validation'

#step2:讀取圖片----------------------------------
files=glob.glob('kagglecatdog/train/dog/*.jpg')
test_files=files[1]                                    #隨機選取一張圖來測試(選索引第10張)
img=image.load_img(test_files,target_size=(150,150))    #將圖片轉成所要的尺寸
x=image.img_to_array(img)                               #將圖片轉成陣列形式
x=np.expand_dims(x,axis=0)                              #將圖片陣列增加一個維度

#step3:使用ImageDataGenerator()函式----------------------------------
increase_test_data=ImageDataGenerator(
    rescale=1./255,                                     #將圖片像素縮放在0~1間
    preprocessing_function=preprocess_input,
    rotation_range=45,                                  #將圖片旋轉幾度(0~180度)
    width_shift_range=0.2,                              #水平平移(相對總寬度的比例)
    height_shift_range=0.2,                             #垂直平移(相對總寬度的比例)
    shear_range=0.2,                                    #隨機傾斜的角度
    zoom_range=0.2,                                     #雖積縮放的範圍
    horizontal_flip=True,                               #一半影像水平翻轉(左右顛倒)
    fill_mode='nearest'                                 #產生的影像若有空白,填補像素
)

#step4:用一張圖片產生多張圖片的結果----------------------------------
i=0
for batch in increase_test_data.flow(x,batch_size=1):
    plt.figure(i)
    imgplot=plt.imshow(image.array_to_img(batch[0]))
    i+=1
    if i%4==0:
        break
plt.show()