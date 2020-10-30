import numpy as np

a = np.array([1,2,3,4,5,6,7,8,9,10,11,12])

a_2d = a.reshape(3,4)
print(a_2d)
print(a_2d.ndim)
print(a_2d.shape)
print('\n')

a_2d_col = a.reshape(3,4, order='F')
print(a_2d_col)
print('\n')

a_3d = a.reshape(3,2,2)
print(a_3d)
print('\n')

a_3d = a.reshape(3,2,2, order='F')
print(a_3d)
print('\n')

a_3d = a.reshape(3,2,2)
print(a_3d)
print('\n')