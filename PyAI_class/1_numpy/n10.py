# NumpPy 二維陣列的索引
import numpy as np
a = np.arange(0,36).reshape(6,6)
print(a)
print('')

print(a[4,5])
print(a[4,-1])
print('')

print(a[3, 1:4 ])
print('')

print(a[ :3 , 3: ])
print('')

print(a[2,:])
print('')

print(a[:,3])  #理論上應是行向量, 但NumPy輸出是列向量
print('')

print(a[:, ::2 ])
print('')

print(a[::2,::3])

