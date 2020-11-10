# encoding:utf-8
'''
將彩色圖片轉為灰階圖像
'''
import cv2

imgdata = cv2.imread('face/face0.jpg')
print('原始影像圖片size與通道數',imgdata.shape)           #包含R,G,B:"3種"元色,每種元色各256種階度

img_gray1 = cv2.cvtColor(imgdata, cv2.COLOR_BGR2GRAY)  #將圖像轉為灰階,降低運算複雜度
print('灰階影像圖片size與通道數',img_gray1.shape)          #包含"1個"灰階,共256個階度

# imgdata2 = cv2.imread('face/face0.jpg',cv2.IMREAD_GRAYSCALE)      #另外一種灰階方式:直接設定參數cv2.IMREAD_GRAYSCALE
# print('灰階影像圖片size與通道數',imgdata2.shape)

#將灰階圖像顯示在視窗
cv2.namedWindow('img-color', cv2.WINDOW_NORMAL)
cv2.imshow('img-color',imgdata)
cv2.namedWindow('img_gray1', cv2.WINDOW_NORMAL)
cv2.imshow('img_gray1',img_gray1)
# cv2.namedWindow('img_gray2', cv2.WINDOW_NORMAL)
# cv2.imshow('img_gray2',imgdata2)

# 等待隨機一個按鍵
cv2.waitKey(0)

#另存一個新的檔案
cv2.imwrite("capture/save_face0_gray.jpg",img_gray1)

#關閉視窗
cv2.destroyAllWindows()