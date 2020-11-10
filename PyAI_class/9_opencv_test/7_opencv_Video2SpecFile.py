'''
將影片video轉存為圖片(設定某範圍的幀)
    依時間間隔固定擷取圖片(多張)
'''
import cv2

videoPath = "opencvtest/165297.mp4"

videoPath = "opencvtest/165297.mp4"                     #影片路徑
cap = cv2.VideoCapture(videoPath)                       #打開影片文件
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   #影片的總帧数
fps = cap.get(cv2.CAP_PROP_FPS)                         #影片的帧率
dur = total_frames / fps                                #影片的時間長度
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)               #影片的寬度
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)             #影片的高度
# print('total_frames',total_frames)
# print('fps',fps)
# print('dur',dur)
# print('width',width)
# print('height',height)

split_num = 10              #預計要從視頻中平均間隔切出10張照片

frameIndexPerStep = int(total_frames/split_num)
print(frameIndexPerStep)

for i in range(0,total_frames,frameIndexPerStep):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    success, image = cap.read()
    if success:
        print('save', i)
        cv2.imwrite("capture\image" + str(i) + ".jpg", image)


cap.release()           #釋放攝影機資源
cv2.destroyAllWindows() #關閉所有視窗