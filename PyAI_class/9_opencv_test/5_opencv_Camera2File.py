'''
將攝影機的影像轉存為圖片
'''
import cv2
import datetime

#打開攝影機
cap = cv2.VideoCapture(0)
print(cap)
#假如攝影機已經打開，則執行while迴圈
while(cap.isOpened()):
    ret, frameData =cap.read()          #讀取攝影機的一張畫面frame
    cv2.imshow('frame',frameData)       #顯示攝影機的一張畫面frame
    key = cv2.waitKey(1000)             #每1秒等待按鍵
    ISOTIMEFORMAT = '%Y%m%d%H%M%S'
    theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    print('save')
    cv2.imwrite("capture\image"+theTime+".jpg", frameData)
    if key== ord('x'):                  #按下按鍵x
        print('break')
        break                           #離開攝影機的存取

cap.release()                           #釋放攝影機資源
cv2.destroyAllWindows()                 #關閉所有視窗