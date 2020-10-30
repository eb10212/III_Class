import matplotlib.pyplot as plt
import numpy as np

#設定資料
labels = ['Q1','Q2','Q3','Q4']
sizes = [20,30,40,10]
explode = (0, 0.1, 0, 0)    #每一塊區域距離中心的距離(間格)

plt.axis('equal') #圓形

cmap = plt.cm.viridis       #viridis:顏色色票(可設定區間)
colors = cmap(np.linspace(0., .5, len(labels)))

#設定屬性(圓餅圖,餅外的標示,餅內的標示)-可略
wedges, texts, autotexts = plt.pie(sizes, colors=colors, explode=explode, labels=labels, autopct='%1.2f%%', shadow=False, startangle=90)
plt.setp(autotexts, size=14, weight="bold", color='red' )
plt.setp(texts, size=18, weight="bold" )

##直接畫圓餅圖
# plt.pie(sizes, colors=colors, explode=explode, labels=labels, autopct='%1.2f%%', shadow=False, startangle=90)

plt.title('Pie Chart', weight="bold")

plt.show()
