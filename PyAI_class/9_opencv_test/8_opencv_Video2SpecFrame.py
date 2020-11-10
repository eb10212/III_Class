'''
將影片video轉存為圖片(指定某一幀)
    只截取一張
'''
import cv2

videoPath = "opencvtest/165297.mp4"                     #影片路徑
cap = cv2.VideoCapture(videoPath)                       #打開影片文件
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   #影片的總帧数
fps = cap.get(cv2.CAP_PROP_FPS)                         #影片的帧率
dur = total_frames / fps                                #影片的時間長度
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)               #影片的寬度
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)             #影片的高度
print('total_frames',total_frames)
print('fps',fps)
print('dur',dur)
print('width',width)
print('height',height)


#設定目前影片帧的位置
cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
position = cap.get(cv2.CAP_PROP_POS_FRAMES)
print('frame position',position)
#讀取目前影片帧位置的frame, 並儲存
success, image = cap.read()
cv2.imwrite("capture\one"+str(position)+".jpg", image)

position = cap.get(cv2.CAP_PROP_POS_FRAMES)
print('now frame position',position)

cap.release() #釋放攝影機資源
cv2.destroyAllWindows() #關閉所有視窗
