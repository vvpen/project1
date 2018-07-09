# 用一张图片来识别，通过模型训练好的数据


import tensorflow as tf
from datasets import dataset_factory
from preprocessing import preprocessing_factory
from nets import nets_factory

import numpy as np

tf.app.flags.DEFINE_string('datasets_name', 'cifar10', '')
tf.app.flags.DEFINE_string('datasets_dir', 'd:/temp/ai/cifar10', '')

tf.app.flags.DEFINE_string('model_name', 'cifarnet', '')
tf.app.flags.DEFINE_string('output_file', 'd:/temp/ai/pb/output.pb', '')

tf.app.flags.DEFINE_string('cheakpoint_path', 'd:/temp/ai/cifarnet-model', '')

tf.app.flags.DEFINE_string('pic_path', 'c:/Users/vvpen/Desktop/素材/dog.png', '')

FLAGS = tf.app.flags.FLAGS

is_training = False
preprocessing_name = FLAGS.model_name

graph = tf.Graph().as_default()

dataset = dataset_factory.get_dataset(FLAGS.datasets_name, 'train', FLAGS.datasets_dir)

image_preprocessing_fn = preprocessing_factory.get_preprocessing(FLAGS.model_name, is_training=False)

network_fn = nets_factory.get_network_fn(FLAGS.model_name, num_classes=dataset.num_classes, is_training=False)

if hasattr(network_fn, 'default_image_size'):
    image_size = network_fn.default_image_size
    print(1)
    print(image_size)
else:
    image_size = FLAGS.default_image_size
    print(2)
    print(image_size)
#

# 读取输入进来的图片，并进行解码
imgIn = tf.placeholder(name="input", dtype=tf.string)
image = tf.image.decode_jpeg(imgIn, channels=3)
print(image)
# 图片预处理工作
image = image_preprocessing_fn(image, image_size, image_size)
print(image)
# 增加一个维度
image = tf.expand_dims(image, 0)
print(image)
# 送入模型进行预测
logit, end_points = network_fn(image)

#
sess = tf.Session()
# 读取最后一次的模型文件
saver = tf.train.Saver()
cheakpoint_path = tf.train.latest_checkpoint(FLAGS.cheakpoint_path)
saver.restore(sess, cheakpoint_path)

# 读取一张图片，并且运行session
image_value = open(FLAGS.pic_path, 'rb').read()
logit_value = sess.run([logit], feed_dict={imgIn: image_value})

# 结果
print(logit_value)
print(np.argmax(logit_value))
