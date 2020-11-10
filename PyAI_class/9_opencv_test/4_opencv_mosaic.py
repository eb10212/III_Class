# encoding:utf-8
'''
將影像打馬賽克
'''
import cv2
import numpy as np

imgdata = cv2.imread('face/face0.jpg')
# print(imgdata.shape)

roi_image =imgdata[750:1200,200:2000]

#取得原始影像要馬賽克的ROI範圍設定馬賽克資訊, 1200-750=450,2000-200=1800
# #產生一個馬賽克圖片大小為 450x1800
mask =np.random.randint(0,256,(450,1800,3)) #(最小值,最大值,圖片的形狀)
imgdata[750:1200,200:2000] = mask

#將圖像顯示在視窗
cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.imshow('mask',imgdata)
# cv2.namedWindow('after-mask', cv2.WINDOW_NORMAL)
# cv2.imshow('after-mask',roi_image)

# 等待隨機一個按鍵
cv2.waitKey(0)

#另存一個新的檔案
cv2.imwrite("capture/save_face0_mosaic.jpg.jpg",imgdata)

#關閉視窗
cv2.destroyAllWindows()
