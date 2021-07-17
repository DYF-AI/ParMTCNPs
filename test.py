import tensorflow as tf
 
c = tf.truncated_normal(shape=[2,1], mean=0.5, stddev=0.5)
 
with tf.Session() as sess:
    print(sess.run(c))