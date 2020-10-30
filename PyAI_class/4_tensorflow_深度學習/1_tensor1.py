import tensorflow as tf

c0 = tf.constant( 1 )
c1 = tf.constant( [1, 2] )
c2 = tf.constant( [[1],[2]] )

print("零階張量形狀")
print(c0)
print("一階張量形狀")
print(c1)
print("二階張量形狀")
print(c2)


with tf.Session() as sess:                #tensorflow的版本在2.0.0以上不能直接使用.Session()
    print("零階張量執行後結果")
    print(sess.run(c0))
    print("一階張量執行後結果")
    print(sess.run(c1))
    print("二階張量執行後結果")
    print(sess.run(c2))

# print('======')
# tf.compat.v1.disable_eager_execution()      #tensorflow的版本在2.0.0以上要加此段
# c0 = tf.constant( 1 )
# c1 = tf.constant( [1, 2] )
# c2 = tf.constant( [[1],[2]] )
# sess=tf.compat.v1.Session()
# print(sess.run(c0))
# print(sess.run(c1))
# print(sess.run(c2))
