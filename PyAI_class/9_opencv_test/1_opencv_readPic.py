'''
基本使用OpenCV讀取儲存靜態影像
    pip install opencv-python
    pip install opencv-contrib-python
'''
# encoding:utf-8
import cv2

#step1:讀取圖片----------------------------------------
imgdata = cv2.imread('face/face0.jpg')          #cv2.imread(路徑):讀進來的資料會自動儲存成一個NumPy的陣列
print('imread的格式:',type(imgdata))

#step2:顯示圖片----------------------------------------
cv2.namedWindow('My image',cv2.WINDOW_NORMAL)   #1.可自由調整視窗的大小(預設為自動依照影像的大小顯示視窗)
cv2.imshow('My image',imgdata)                  #2.將圖像顯示在視窗,imshow(視窗名稱,圖片),名稱要一致


#step3:寫入圖片----------------------------------------
cv2.imwrite("capture/save_face0.jpg",imgdata)   #cv2.imwrite(指定存檔路徑,圖片資訊)

#step4:關閉視窗----------------------------------------
cv2.waitKey(0)                    #cv2.waitKey:等待使用者按下的鍵盤上任意按鍵的時間(單位為毫秒),若=0:表示持續等待至使用者按下按鍵為止
                                  #若無等待,則圖片會一閃而過
cv2.destroyAllWindows()           #關閉視窗
# cv2.destroyWindow('My image')     #關閉單一視窗(依視窗名稱)