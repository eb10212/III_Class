'''
在圖片上加上文字
'''
import numpy as np
import cv2

img = np.zeros((400, 600, 3), np.uint8)   #產生空白的圖片
img.fill(0)

text = 'Sato Takeru : One night!'         # 文字

# 參數依順序為:照片,添加的文字,左上角坐標,字體,字體大小,顏色,字體粗細,類型
cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(img, text, (10, 80), cv2.FONT_HERSHEY_PLAIN,1, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(img, text, (10, 120), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(img, text, (10, 160), cv2.FONT_HERSHEY_COMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(img, text, (10, 200), cv2.FONT_HERSHEY_TRIPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(img, text, (10, 240), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(img, text, (10, 280), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(img, text, (10, 320), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

cv2.imshow('font', img)
cv2.waitKey(0)
cv2.destroyAllWindows()