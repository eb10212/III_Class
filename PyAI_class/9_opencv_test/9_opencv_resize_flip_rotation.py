'''
影像處理
    -1.調整尺寸大小:resize
    -2.圖片翻轉(90/180度):flip
    -3.圖片旋轉(360度):rotation
'''
import cv2

cv2.namedWindow('0_imgdata',cv2.WINDOW_NORMAL)
imgdata = cv2.imread('face/face0.jpg')              #原始圖片
cv2.imshow('0_imgdata', imgdata )

#1.調整尺寸大小(resize)-------------------------------------------
cv2.namedWindow('1_resize_image',cv2.WINDOW_NORMAL)
resize_image = cv2.resize(imgdata,(1000,500))       #調整大小後
cv2.imshow('1_resize_image', resize_image )

#2.圖片翻轉(flip)-------------------------------------------
cv2.namedWindow('2_1_flip_image1',cv2.WINDOW_NORMAL)
flip_image = cv2.flip(imgdata,0)                    #重直翻轉,參數=0
cv2.imshow('2_1_flip_image1', flip_image )

cv2.namedWindow('2_2_flip_image2',cv2.WINDOW_NORMAL)
flip_image = cv2.flip(imgdata,1)                    #水平翻轉,參數=1
cv2.imshow('2_2_flip_image2', flip_image )

cv2.namedWindow('2_3_flip_image3',cv2.WINDOW_NORMAL)
flip_image = cv2.flip(imgdata,-1)                   #同時水平與重直翻轉,參數=-1
cv2.imshow('2_3_flip_image3', flip_image )

#3.圖片旋轉(rotation)-------------------------------------------
#先設定旋轉影像-產生矩陣資料
height, width = imgdata.shape[:2]            #imgdata.shape:為tuple,如(1378, 2068, 3),只要前兩個的資料height=1378,width=2068
M = cv2.getRotationMatrix2D((height/2,width/2),60,0.6)  #旋转中心:在圖片的中心,逆時針旋轉:60度,旋轉後的影像呈現:60%(變小)/1:不變

cv2.namedWindow('3_rotation_image',cv2.WINDOW_NORMAL)
rotation_image = cv2.warpAffine(imgdata,M,(height, width))  #輸入原始資料與旋轉矩陣資料,輸出的尺寸
cv2.imshow('3_rotation_image', rotation_image )


# 等待隨機一個按鍵
cv2.waitKey(0)
#關閉視窗
cv2.destroyAllWindows()