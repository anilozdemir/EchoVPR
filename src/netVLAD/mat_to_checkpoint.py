# TODO: purpose? if not, remove it

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


from net_from_mat import netFromMat
from nets import defaultCheckpoint

tf.reset_default_graph()
layers = netFromMat()
saver = tf.train.Saver()

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
saver.save(sess, defaultCheckpoint())