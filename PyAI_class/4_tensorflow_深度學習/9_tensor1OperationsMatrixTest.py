import tensorflow as tf

# 二維張量 2*3
A = tf.constant( [ [1,2,3],[4,5,6] ] )

#二維張量 3*2
B = tf.constant( [ [1,2], [3,4],[5,6] ] )

matmulAB = tf.matmul(A, B) #矩陣的乘法
matmulBA = tf.matmul(B, A) #矩陣的乘法

with tf.Session() as sess:
    print("matrixAB為{}矩陣, 數值為{}".format(matmulAB.get_shape(), sess.run(matmulAB)))
    print("matrixBA為{}矩陣, 數值為{}".format(matmulBA.get_shape().as_list(), sess.run(matmulBA)))