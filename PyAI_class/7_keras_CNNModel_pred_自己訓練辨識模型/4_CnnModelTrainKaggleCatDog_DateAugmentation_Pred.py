'''
隨機挑一張圖,測試模型是否能判斷是貓還是狗
'''
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from PIL import Image               #顯示圖片
import matplotlib.pyplot as plt

# #step1:設定資料的相對路徑&建立訓練資料與測試資料-------------------------------
# train_dir = 'kagglecatdog/train'
# test_dir = 'kagglecatdog/test'
# validation_dir = 'kagglecatdog/validation'
#
# train_datagen =  ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow_from_directory(train_dir )
#
# print('='*30)
# print('訓練的分類：',train_generator.class_indices)
# print('='*30)
#
# labels = train_generator.class_indices
#
# labels = dict((v,k) for k,v in labels.items())      #將分類做成字典方便查詢(後續結果調用)
# print(labels)

labels={0:'cat',1:'dog'}

#step2:定義待測圖片轉數據的方式-------------------------------
def read_image(img_path):                           #將圖片轉為待測數據
    try:
        img = image.load_img(img_path, target_size=(150, 150))
    except Exception as e:
        print(img_path,e)

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


#step3:選擇一張圖來作後續的預測分類用-------------------------------
filename = "../6_keras_PretrainedModel_預先訓練模型/vgg16TestPic/1.jpg"

plt.figure()
im = Image.open(filename)
im_list = np.asarray(im)
plt.title("predict")
plt.axis("off")
plt.imshow(im_list)
plt.show()


#step4:使用模型預測分類-------------------------------
model = load_model('3_model_CnnModelTrainKaggleCatDog_DateAugmentation.h5')
img = read_image(filename)
pred = model.predict(img)[0]           #結果不是0就是1(list的形式,所以要[0,]or[1.])
print('辨識結果:',labels[pred[0]])      #使用labels字典搜尋結果(0代表cat,1代表dog)
# print('辨識結果:',labels[int(pred)])


