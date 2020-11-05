'''
使用vgg16做照片相似度排序
將vgg16TestPic目錄內的每一張jpg取出其特徵向量,並相互比較,利用cosine函數計算兩張照片特徵向量的角度(越接近1越相似)
'''
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import numpy as np
import os

#step1:定義相似矩陣函數(cosine定理的公式推導過程)---------------------------------
def cosine_similarity(featuresVector):
    sim = featuresVector.dot(featuresVector.T)      #與自己(featuresVector)的轉置矩陣(featuresVector.T)做內積運算(dot)
    if not isinstance(sim, np.ndarray):             #isinstance:判斷是否相同類型
        sim = sim.toarray()                         #轉成陣列格式
    norms = np.array([np.sqrt(np.diagonal(sim))])   #np.diagonal:取對角線,np.sqrt:取平方根
    return (sim/norms/norms.T)

#step2:相似矩陣的計算---------------------------------
images_filename_list = []
images_data_tuple = []
for img_path in os.listdir("vgg16TestPic"):         #自vgg16TestPic 目錄找出所有JPEG檔案
    if img_path.endswith(".jpg"):
        img = image.load_img("vgg16TestPic/" + img_path, target_size=(224, 224))
        images_filename_list.append(img_path)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        if len(images_data_tuple) == 0:
            images_data_tuple = x                   #每張圖片的特徵陣列
        else:
            images_data_tuple = np.concatenate((images_data_tuple, x))      #np.concatenate:tuple的拼接(將每張圖的特徵x作疊加的動作)

print('檔案目錄:',images_filename_list)
images_data_tuple = preprocess_input(images_data_tuple)                     #轉圖片為VGG的格式(前處理)


#step3:載入VGG16---------------------------------
model = VGG16(weights='imagenet', include_top=False)            #include_top=False:表示只計算出特徵,不使用最後3層的全連接層(只需要特徵不需要分類,所以不使用原來的分類器)
model.summary()                                                 #顯示出模型摘要

#使用vgg16的模型-預測出特徵
features = model.predict(images_data_tuple)

#計算特徵向量
featuresVector = features.reshape(len(images_filename_list), 7*7*512)   #由於最後一層vgg16的池化層輸出的張量大小為7*7*512(model.summary中看出)

#step4:計算相似矩陣(套用函示)---------------------------------
sim = cosine_similarity(featuresVector)
print(sim)                                  #印出所有照片之間的特徵值(越接近1.0,照片越相近)

#step5:測試單一張圖片---------------------------------
testPicID = 2                               #用在images_filename_list中索引位置=2的圖來當測試圖
top = np.argsort(-sim[testPicID], axis=0)   #由大到小的排序(np.argsort:進行由小到大的排序,-sim:表示將資料列反向)
print(top)

rank = [images_filename_list[i] for i in top]
print('目錄內的所有照片與測試檔案"{}"的相似度排序(越前面的越相似):{}'.format(images_filename_list[testPicID],rank))


