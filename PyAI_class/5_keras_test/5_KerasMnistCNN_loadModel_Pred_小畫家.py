'''
自製一張28*28的數字圖檔(jpg)
進行辨識
'''
import matplotlib.pyplot as plt
import glob, cv2
from keras.preprocessing import image
from keras.models import load_model
import numpy as np


def show_predictions(images, labels, predictions):
    plt.gcf().set_size_inches(6, 5) #建立畫布視窗大小

    for i in range(len(predictions)):
        figs = plt.subplot(3, 3, 1 + i)
        # 顯示黑白圖片
        figs.imshow(images[i], cmap='binary')

        # 有預測結果, 才顯示預測結果
        if (len(predictions) > 0):
            title = 'predic. = ' + str(predictions[i])
            # 正確顯示(o), 錯誤顯示(x)
            title += (' (o)' if predictions[i] == labels[i] else ' (x)')
            title += ' label = ' + str(labels[i])
        # 沒有預測結果, 只顯示真實數值
        else:
            title = 'label = ' + str(labels[i])

        # 不顯示刻度
        figs.set_title(title, fontsize=6)
        figs.set_xticks([])
        figs.set_yticks([])
        #start_id += 1
    plt.show()

# 讀取目錄下的圖片檔路徑, 存在files list裡
files = glob.glob("minstTestPic/*.jpg")
print(files)
test_images_list = []
test_label_list = []

for file in files:
    img = image.load_img(file, target_size=(28, 28))                #將圖片轉為28*28像素
    img = image.img_to_array(img)
    print(img.shape)                                                #(28,28,3):3表示彩色
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                     #轉成灰階
    print(img.shape)                                                #(28,28)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)    #轉為反相黑白
    test_images_list.append(img)
    label = file[13:14]                                             #"minstTestPic/1.jpg":取第13個索引位置字元作為label
    test_label_list.append(int(label))

test_images_list = np.array(test_images_list)           #串列轉為矩陣
test_label_list = np.array(test_label_list)             #串列轉為矩陣

# 將待測的圖片list轉換為 n*28*28*1 的4維(含顏色)矩陣(CNN)
test_image_mtx = test_images_list.reshape(len(test_images_list), 28, 28, 1).astype('float32')

# 將待測的圖片list轉換為 784個 float 數字的 1 維向量 (如果模型是多層感知機, 使用下面的正規化)
#test_image_mtx = test_images_list.reshape(len(test_images_list), 784).astype('float32')

print(test_image_mtx.shape)  # (3,28,28,1)=(張數,長,寬,顏色)

# 將待測的圖片標準化, 準備做預測
test_image_norm = test_image_mtx / 255

# 載入模型
print("載入模型 3_model_KerasMnistCNN.h5")
model = load_model('3_model_KerasMnistCNN.h5')

# 預測結果
prediction = model.predict_classes(test_image_norm)
print("預測結果")
print(prediction)

# 顯示待測圖像、顯示預測值、顯示真實值(使用一開始定義的函示)
show_predictions(test_images_list, test_label_list, prediction)