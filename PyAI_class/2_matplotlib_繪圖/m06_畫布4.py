import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-np.pi,np.pi,200)
y1 = np.cos(x)
y2 = np.sin(x)

plt.figure(figsize=(18,4))

# 依照 [left, bottom, width, height] 設定百分比
plt.axes([.1, .1, .8, .8])
plt.plot(x,y1)

plt.axes([.4, .3, .2, .2])
plt.plot(x,y2)

plt.show()



