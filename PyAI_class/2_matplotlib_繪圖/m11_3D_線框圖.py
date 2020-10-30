import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from mpl_toolkits.mplot3d.axes3d import get_test_data

# 設定視角(角度)
fig = plt.figure(figsize=plt.figaspect(0.3))

#第一個子圖(列,行,第幾個位置,投影方式)
ax = fig.add_subplot(1, 2, 1, projection='3d')

X = np.arange(-6,6,0.25)
Y = np.arange(-5,5,0.25)
# print(X,Y)
X, Y = np.meshgrid(X, Y)    #用作標點畫出隔線(平面網格)
# plt.plot(X,Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
# plt.plot(X,Y,Z)

#圖表訊息
plt.xlabel("X-axis", fontsize=16)
plt.ylabel("Y-axis", fontsize=16)

#畫3D曲面圖
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.rainbow,linewidth=0, antialiased=False)
                                # rstride=1,cstride=1:預設步數(一開始X,Y所設定的步長=.25)表示行列隔多少個取樣點建一個小面
                                # cmap:顏色
#設定Z軸範圍
ax.set_zlim(-1.01, 1.01)
#設定邊條顏色註解
fig.colorbar(surf, shrink=0.5, aspect=10)

#第二個子圖
ax = fig.add_subplot(1, 2, 2, projection='3d')

X, Y, Z = get_test_data(0.5)

#ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)

#畫線框圖
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.rainbow, linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=10)

plt.show()

