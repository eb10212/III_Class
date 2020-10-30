#陣列的齊他索引方式
import numpy as np

x=np.arange(25).reshape(5,5)
print(x)
print('')

print( x[[0,1,2],[0,1,0]])  #(0,0) (1,1) (2.0)
print('')

print( x[[0],[1,2,4]])
print('')

print(x[[2,4]]) #row 2 4
print('')

print(x[ x > 20 ])  #布林索引
print(x[ x % 2 == 0 ])  #布林索引


