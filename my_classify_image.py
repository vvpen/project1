import tensorflow as tf

import numpy as np

FLAGS = tf.app.flags.FLAGS


def create_graph(model_file=None):
    if not model_file:
        model_file = FLAGS.model_file
    #
    with open(model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    #
#


# 文件地址
image_path = "c:/Users/vvpen/Desktop/素材/dog.png"
model_file = "d:/temp/ai\pb/frozen_graph.pb"

# 初始化模型
create_graph(model_file)

with tf.Session() as sess:
    # 读取图片文件
    image_data = open(image_path, 'rb').read()
    # 读取输入进来的图片，并进行解码
    imgIn = tf.placeholder(name="input", dtype=tf.string)
    image = tf.image.decode_jpeg(imgIn, channels=3)
    # 增加一个维度
    image = tf.expand_dims(image, 0)
    # 获取图片矩阵 1 * 32 * 32 * 3
    image_v = sess.run(image, feed_dict={imgIn: image_data})
    print(image_v.shape)
    print(type(image_v))

    # 拿到图片矩阵数据后，直接调用模型
    softmax_tensor = sess.graph.get_tensor_by_name("CifarNet/Predictions/Softmax:0")
    perdictions = sess.run(softmax_tensor, {"input:0": image_v})
    # perdictions = sess.run(softmax_tensor, {"CifarNet:0": image_v})
    perdictions = np.squeeze(perdictions)
    print(perdictions)
    print(np.argmax(perdictions))
#
