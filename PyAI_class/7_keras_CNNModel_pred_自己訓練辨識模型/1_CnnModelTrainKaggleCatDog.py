'''
下載好kagglecatdog.zip
建立自己的CNN-訓練貓/狗圖片分類
'''
import matplotlib.pyplot as plt
from keras.models import Sequential                             #使用Sequential模型(多層神經元)
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D   #全連結/平坦/卷積/池化 層
from keras.preprocessing.image import ImageDataGenerator        #從圖片轉為可訓練的資料(特徵矩陣)
from keras.optimizers import RMSprop                            #優化器(梯度下降演算法)
from keras.callbacks import EarlyStopping

#設定資料的相對路徑
train_dir = 'kagglecatdog/train'
test_dir = 'kagglecatdog/test'
validation_dir = 'kagglecatdog/validation'

#step1:建立CNN網路模型----------------------------------
model = Sequential()
#加入各層(共11層)
model.add(Conv2D(filters=32,
                 kernel_size=(3,3),
                 input_shape=(150,150,3),       #一開始輸入的圖片大小(自己設定要將原始的圖片要修改成怎樣的統一規格),只有第一層需要設定
                 activation='relu'
                ))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=128,
                 kernel_size=(3,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=128,
                 kernel_size=(3,3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(1,activation='sigmoid'))        #1個激活函數sigmoid:2分法(不是貓就是狗)

model.summary()


#step2:定義模型的優化方式----------------------------------
estop = EarlyStopping(monitor='val_loss', patience=3)

model.compile(loss='binary_crossentropy',       #損失函數用2元交叉熵(不是0就是1)
              optimizer=RMSprop(lr=1e-4),       #(lr=1e-4):學習率
              metrics=['acc'] )                 #metrics:放入需要觀察的accuracy(準確度)/cost/score等


#step3:建立訓練資料與測試資料----------------------------------
train_datagen =  ImageDataGenerator(rescale=1./255)    #將影像轉為批次量的張量,並指定將影象像素縮放到0~1之間
test_datagen = ImageDataGenerator(rescale=1./255)
#操作方式:產生器
train_generator = train_datagen.flow_from_directory(train_dir,                  #參數1:訓練取用的資料來源路徑
                                                    target_size=(150, 150),     #參數2:尺寸
                                                    batch_size=20,              #每批次產生20張
                                                    class_mode='binary')        #class_mode(分類):超過兩類使用categorical,若只有兩類使用binary

validation_generator = test_datagen.flow_from_directory(validation_dir,         #參數1:測試取用的資料來源路徑(使用驗證資料集)
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')


#step4:訓練模型與保存參數----------------------------------
#因為使用資料產生器ImageDataGenerator,所以訓練模型時需使用fit_generator
print(train_generator.samples)
print(train_generator.batch_size)
H = model.fit_generator(train_generator,                            #使用所產生的訓練資料
                        steps_per_epoch=train_generator.samples/train_generator.batch_size, #相除過後,每回合訓練100次
                        epochs=30,                                  #共30回合
                        validation_data=validation_generator,       #使用所產生的驗證資料
                        validation_steps=50,                        #驗證50次
                        callbacks=[estop]
                        )

model.save('1_model_CnnModelTrainKaggleCatDog.h5')

#step5:畫出訓練結果----------------------------------
# 顯示acc學習結果
accuracy = H.history['acc']
val_accuracy = H.history['val_acc']
plt.plot(range(len(accuracy)), accuracy, marker='.', label='accuracy(training data)')
plt.plot(range(len(val_accuracy)), val_accuracy, marker='.', label='val_accuracy(evaluation data)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# 顯示loss學習結果
loss = H.history['loss']
val_loss = H.history['val_loss']
plt.plot(range(len(loss)), loss, marker='.', label='loss(training data)')
plt.plot(range(len(val_loss)), val_loss, marker='.', label='val_loss(evaluation data)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

'''
圖形結果:訓練的正確率曲線與驗證的曲線相距太多-->發生"過擬合"(我們要的是驗證的那條曲線)
    -原因可能是訓練的資料量太少!!
'''