import tensorflow as tf
tf.compat.v1.disable_eager_execution()
c0 = tf.constant( [0, 0, 0, 1, 0, 0, 0, 0, 0, 0] );
c1 = tf.constant(  [[1, 2, 3],[4, 5, 6]] );
c2 = tf.constant( [[1, 2],[3, 4],[4, 5]]  );

# print("c0張量形狀")
# print(c0)
# print("c1張量形狀")
# print(c1)
# print("c2張量形狀")
# print(c2)

with tf.compat.v1.Session() as sess:
    print("c0張量執行後結果")
    print(sess.run(c0))
    print("c1張量執行後結果")
    print(sess.run(c1))
    print("c2張量執行後結果")
    print(sess.run(c2))
