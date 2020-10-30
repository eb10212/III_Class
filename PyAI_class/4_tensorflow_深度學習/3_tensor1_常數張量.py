import tensorflow as tf
#建立形狀
zeros = tf.zeros([2,5])     #內容為0的常數張量 [幾維/幾列,幾個/幾行]

ones = tf.ones([1,3])       #內容為1的常數張量 [幾維/幾列,幾個/幾行]

fill = tf.fill([1,3],5)     #內容為"特定值"的常數張量 ([幾維/幾列,幾個/幾行],特定值)

range1 = tf.range( 5 )          #內容為"一個數值範圍且距離街等差"的常數張量,預設開頭為0等差為1,此題為0~5間(不含5)的等差數列張量[0 1 2 3 4]
range2 = tf.range( 5, delta=2 ) #此題為0~5間(不含5)且等差=2的數列張量[0 2 4],寫法等同tf.range( 5,2 )

linspace = tf.linspace(1.0, 5.0, 3)     #(下界,上界,產生多少個元素)

#顯示
with tf.Session() as sess:
    print(sess.run(zeros))
    print('==============')
    print(sess.run(ones))
    print('==============')
    print(sess.run(fill))
    print('==============')
    print(sess.run(range1))
    print('==============')
    print(sess.run(range2))
    print('==============')
    print(sess.run(linspace))