import matplotlib.pyplot as plt
import numpy as np

X = np.arange(20)
Y = np.random.uniform(0.5,1.0, 20)#隨機產生0.5~1機率相同的20個數值

for x, y in zip(X, Y):              #zip:產生兩兩一組的(x,y)
    plt.text(x , y , '%.2f' % y, ha='center', va='bottom', fontsize=6)
                                # 長條圖上面的數(文)字對齊方式:ha水平//va垂直
plt.bar(X,Y,facecolor='#9999ff', edgecolor='black')

plt.show()
