'''
使用Fine-tune方式微調InceptionV3模型
    辨識4種葉子
'''
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.preprocessing.image import ImageDataGenerator       #資料產生器
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping,ModelCheckpoint      #ModelCheckpoint(輔助函數):挑出最好的權重值
#以上是建立CNN所需,以下是針對微調的預訓練模型
from keras.applications.inception_v3 import InceptionV3

#取得資料位置
train_dir = 'leaf/train'
test_dir = 'leaf/test'

class_numbers=len(os.listdir(train_dir))           #總分類數
print('總共有幾類:',class_numbers)                   #class_numbers=4 總共有四種葉子


#step1:微調方式(設定網路結構, 使用在imagenet上訓練的參數作為初始參數)-----------------------------------------
#1.建立整體Backbone(骨幹)
Backbone=InceptionV3(weights='imagenet',input_shape =(299, 299,3),include_top=False)   #include_top=False:不使用預先的分類器
Backbone.trainable = True                   #設定所有層為可訓練,方便後續使用
set_trainable = False                       #凍結布林變數

#2.再將Backbone(原InceptionV3)中的前249層全部凍結(不更動),再微調訓練249層之後(解凍)
for layer in Backbone.layers[:249]:
   layer.trainable = False
for layer in Backbone.layers[249:]:             #部分微調Fine-tune的部分
   layer.trainable = True


#step2:建立模型-----------------------------------------
model = Sequential()

#建立自己的分類器(因為不使用InceptionV3的分類器,25行中:include_top=False)
model.add(Backbone)                         #使用InceptionV3的預訓練模型
model.add(Flatten())                        #加入自己的平坦層(轉一維陣列的概念)
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))                     #丟棄0.5的神經元
model.add(Dense(class_numbers, activation='softmax'))       #class_numbers=4

model.summary()


#step3:建立訓練資料與測試資料-----------------------------------------
#使用ImageDataGenerator產生器並"增加訓練資料"的學習樣本
train_datagen=ImageDataGenerator(
    rescale=1./255,                             #指定將影象像素縮放到0~1之間
    #preprocessing_function=preprocess_input,
    rotation_range=45,                          #影象旋轉的角度值(0~180度)
    width_shift_range=0.2,                      #水平平移(相對總寬度的比例)
    height_shift_range=0.2,                     #垂直平移(相對總高度的比例)
    shear_range=0.2,                            #隨機錯切換角度
    zoom_range=0.2,                             #隨機縮放範圍
    horizontal_flip=True,                       #一半影像水平翻轉
    fill_mode = 'nearest'                       #產生新的影像若有出現空白處,以"最接近的像素"填補像素
)
#測試資料的產生器
test_datagen = ImageDataGenerator(rescale=1./255)

#操作方式:train/test產生器所產生的資料
train_generator =train_datagen.flow_from_directory(
    train_dir,                      #訓練取用的資料來源路徑(一開始設定的變數)
    target_size=(299, 299),         #圖片尺寸
    batch_size=10,                  #每批次產生10張(每次10張訓練)
    class_mode='categorical'        #class_mode(分類):超過兩類使用categorical,若只有兩類使用binary
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=10,
    class_mode='categorical',
    shuffle = False                 #是否隨機從data裡讀資料
)

print('='*30)
print(train_generator.class_indices)
print('='*30)


#step5:訓練模型,保存權重-----------------------------------------
#ModelCheckpoint:輔助函數(會挑出最好的權重值:val_acc)
checkpoint = ModelCheckpoint('1_mode_iv3LeafFinetune.h5',verbose=1,monitor='val_acc', save_best_only=True, mode='auto')
model.compile(loss='categorical_crossentropy',      #損失函數用多元交叉熵
              optimizer=RMSprop(lr=1e-4),           #(lr=1e-4):梯度下降的學習率
              metrics=['acc'])

estop = EarlyStopping(monitor='val_loss', patience=5)       #當監測值val_loss不再改善時(如發現損失沒有下降)則經過5個epoch後停止訓練(patience=5)

#開始訓練模型(使用批量生成器)
H = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples/train_generator.batch_size,     #每一回合從訓練集中抓取訓練樣本訓練,總30次
    epochs=10,                                                              #一共訓練回合
    validation_data=test_generator,                                         #驗證資料集來源
    validation_steps=50,                                                    #驗證次數
    callbacks=[checkpoint, estop],
    verbose=1
)


#step6:畫出曲線-----------------------------------------
epochs = range(len(H.history['acc']))

plt.figure()
plt.plot(epochs, H.history['acc'], 'b',label='Training acc')
plt.plot(epochs, H.history['val_acc'], 'r',label='validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('1_leaf_acc_iv3.png')               #將繪製出的曲線圖存檔
plt.show()

plt.figure()
plt.plot(epochs, H.history['loss'], 'b',label='Training loss')
plt.plot(epochs, H.history['val_loss'], 'r',label='validation loss')
plt.title('Training and validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('1_leaf_loss_iv3.png')

plt.show()

del model
