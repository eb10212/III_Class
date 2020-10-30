import tensorflow as tf
import matplotlib.pyplot as plt

random_normal = tf.random_normal( [100] , 0 , 1)
#常態性質,tf.random_normal([列,行],平均值,標準差):內容為隨機常態100個
#                         [行,]
with tf.Session() as sess:
  print( random_normal.eval() )
  plt.hist( random_normal.eval() )  #直方圖
  plt.show()
