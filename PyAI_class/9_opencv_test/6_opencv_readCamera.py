'''
讀取/播放 攝像機或錄影機的影像
'''
import cv2

#打開攝影機
cap = cv2.VideoCapture(0)
print(cap)
#假如攝影機已經打開，則執行while迴圈
while(cap.isOpened()):
    ret, frameData =cap.read()          #讀取攝影機的一張畫面frame
    cv2.imshow('frame',frameData)       #顯示攝影機的一張畫面frame
    key = cv2.waitKey(1)                #等待按鍵,每1秒釋放一次CPU time,給其他程式執行
    if key== ord('q'):                  #按下按鍵q
        print('save')
        cv2.imwrite("capture\capture.jpg", frameData)
    elif key== ord('x'):                #按下按鍵x
        print('break')
        break                           #離開攝影機的存取

cap.release()                           #釋放攝影機資源
cv2.destroyAllWindows()                 #關閉所有視窗