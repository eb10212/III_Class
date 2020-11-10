# encoding:utf-8
import cv2

# 輸入一張圖片路徑
imgdata = cv2.imread('face/face0.jpg')
print(imgdata.shape)

#將彩色影像抽出
b,g,r = cv2.split(imgdata)

cv2.imshow('imgdata', imgdata )
cv2.imshow('b channel', b )
cv2.imshow('g channel', g )
cv2.imshow('r channel', r)

bgr = cv2.merge([b,g,r])
cv2.imshow('bgr',bgr )

# 等待隨機一個按鍵
cv2.waitKey(0)
#關閉視窗
cv2.destroyAllWindows()
