import cv2

#載入偵測人臉模組
face_cascade = cv2.CascadeClassifier('c://haarcascades//haarcascade_frontalface_default.xml')

#打開攝影機
cap = cv2.VideoCapture(0)
print(cap)
#假如攝影機已經打開，則執行while迴圈
while(cap.isOpened()):
    ret, img = cap.read()  # 輸入一張圖像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 將圖像轉為灰階, 降低運算複雜度

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,  # 有的人臉離鏡頭近，會比其他人臉更大
        minNeighbors=5,  # 檢測矩形其周圍有多少物體進行偵測
        minSize=(30, 30),  # 檢測矩形的大小
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        #照片/添加的文字/左上角坐標/字體/字體大小/顏色/字體粗細
        cv2.putText(img, 'face', (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
        img = cv2.rectangle(img,
                            (x, y),  # 左上角座標
                            (x + w, y + h),  # 寬與高
                            (255, 0, 0),  # 顏色
                            2)  # 線框粗細

        cv2.imshow('img', img)

    key = cv2.waitKey(100) #等待按鍵, 每1秒釋放一次CPU time, 給其他程式執行
    if key== ord('q'): #按下按鍵q
        print('save')
        cv2.imwrite("capture\capture.jpg", img)
    elif key== ord('x'):#按下按鍵x
        print('break')
        break #離開攝影機的存取

cap.release() #釋放攝影機資源
cv2.destroyAllWindows() #關閉所有視窗