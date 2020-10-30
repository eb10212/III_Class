import numpy as np

dt = np.dtype('f2')
print(dt)

people = np.dtype([('name','S20'), ('height', 'i4'), ('weight', 'f2')])

a = np.array([('amy', 156, 50),('bob', 175, 72)], dtype = people )
print(a)
print(a['name'])