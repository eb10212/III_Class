import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

x_raw = pd.period_range(pd.datetime.now(),periods=30, freq='d') #時間區間,從今天開始的日期往後的 30(週期)天(頻率)
print(x_raw)

x = x_raw.to_timestamp().to_pydatetime()    #姜pandas時間格式轉換成python的時間格式
print(x)

y1 = np.random.randint(50,80,30)
y2 = np.random.randint(20,45,30)

g = plt.gca()   #得到當前的GCA當前的座標軸

#設定刻度標籤----------------------------------------------------------
for tick in g.xaxis.get_majorticklabels():  # x軸的日期格式大小
    tick.set_fontsize(6)
for tick in g.yaxis.get_majorticklabels():  # example for xaxis
    tick.set_fontsize(6)
#設定畫布刻度的日期格式
g.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
plt.gcf().autofmt_xdate()   #設定自動旋轉日期標籤
#--------------------------------------------------------------------

plt.title('Random Trends')
plt.grid(True)

plt.plot(x,y1)
plt.plot(x,y2)

plt.show()