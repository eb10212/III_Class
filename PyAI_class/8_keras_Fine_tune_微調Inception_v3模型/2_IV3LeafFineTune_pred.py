'''
使用模型辨識葉子的分類
可先去下載中文字形檔https://www.cns11643.gov.tw/AIDB/Open_Data.zip
'''
from keras.preprocessing import image
import glob
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import os
from keras.models import load_model
import matplotlib.pyplot as plt


def read_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(299, 299))
    except Exception as e:
        print(img_path,e)

    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img/255

#將圖片加上文字後存檔
def draw_save(img_path, label, out='tmp/'):
    img = Image.open(img_path)
    os.makedirs(os.path.join(out,label),exist_ok=True)      #檔案位置設定
    if img is None:
        return None
    draw = ImageDraw.Draw(img)                              #在圖片上加入文字
    font = ImageFont.truetype("TW-Kai-98_1.ttf",160)        #使用中文字形TW-Kai-98_1.ttf
    draw.text((10,10), label, fill='#FFFF00', font=font)    #fill文字顏色:黃色
    img.save(os.path.join(out,label,"test.jpg"))

#將圖片加上文字後show出(不存檔)
def show_img(img_path,label):
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)  # 在圖片上加入文字
    font = ImageFont.truetype("TW-Kai-98_1.ttf", 160)
    draw.text((10, 10), label, fill='#FFFF00', font=font)
    plt.figure()
    im_list = np.asarray(img)
    plt.title("predict")
    plt.axis("off")
    plt.imshow(im_list)
    plt.show()


labels = {'紅葉子': 0, '綠葉子': 1, '褐色葉子': 2, '黃綠葉子': 3}
labels= {str(v): k for k,v in labels.items()}
print(labels)

files = glob.glob("leaf/test/紅葉子/*.JPG")
print(files[1])     #隨意選一個照片(選擇索引位置1)

model = load_model('1_mode_iv3LeafFinetune.h5') #辨識樹葉
img = read_image(files[1])

pred = model.predict(img)[0]
print(pred)

#推論出機率最高的分類, 取得所在位置
index = np.argmax(pred)

print(files[1], labels[str(index)], pred[index])
# draw_save(files[1], labels[str(index)], out='tmp/')       #將預測出的圖加上文字後存檔
show_img(files[1], labels[str(index)])                      #將預測出的圖加上文字後直接顯示
