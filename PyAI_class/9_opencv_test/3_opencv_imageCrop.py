# encoding:utf-8
'''
靜態圖像裁切
'''
import cv2

imgdata = cv2.imread('face/face0.jpg')
print(imgdata.shape)
print(imgdata.size)         #影像的size(長x寬x色彩)

#擷取部分範圍ROI (Region of Interest)
crop_image = imgdata[300:600,  #圖片左上角Y座標值200~下Y座標值400的位置
                     600:950]  #圖片左上角X座標值300~下X座標值600的位置


cv2.namedWindow('My image', cv2.WINDOW_NORMAL)
cv2.imshow('My image',imgdata)
cv2.namedWindow('crop', cv2.WINDOW_NORMAL)
cv2.imshow('crop',crop_image)

# 等待隨機一個按鍵
cv2.waitKey(0)

#另存一個新的檔案
cv2.imwrite("capture/save_face0_crop.jpg",crop_image)

#關閉視窗
cv2.destroyAllWindows()