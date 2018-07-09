"""
DenseNet神经网络
版本：20180708
作者：vvpen

"""
#
import tensorflow as tf

#
slim = tf.contrib.slim

# 一些全局参数
epsilon = 0.001  # BN中的epsilon
debug = False


# 打印控制台日志
def log_debug(d):
    if debug:
        print(d)
    #


#


def densenet(images, num_classes=10, is_training=False, dropout_keep_prob=0.8, scope='densenet'):
    """
    入口函数
    参数:
        indata: 输入数据
        reduction: 输出缩小因子
    """
    return densenet_base(indata=images,
                         filter_num=16,
                         dense_block_items=[6, 12],
                         classes=num_classes,
                         reduction=0.5,
                         is_training=is_training,
                         dropout=dropout_keep_prob)


#


def densenet_base(indata, filter_num, dense_block_items, classes, reduction=0.5, is_training=True, dropout=False):
    """
    参数:
        indata: 输入数据
        reduction: 输出缩小因子
    """
    log_debug("MyDenseNet======================")
    #
    end_points = {}

    # 初始层：卷积
    end_point = 'conv_0'
    with tf.variable_scope(end_point):
        indata = slim.conv2d(inputs=indata,
                             num_outputs=filter_num,
                             kernel_size=[7, 7],
                             stride=2,
                             padding='SAME',
                             activation_fn=tf.nn.relu,
                             normalizer_fn=slim.batch_norm,
                             scope=end_point)
        log_debug(indata)
    end_points[end_point] = indata
    # 初始层：池化
    end_point = 'pool_0'
    with tf.variable_scope(end_point):
        indata = slim.max_pool2d(indata, [3, 3], stride=2, padding='SAME', scope=end_point)
        log_debug(indata)
    end_points[end_point] = indata

    log_debug("===================")

    # 每个DenseBlock层
    dense_block_len = len(dense_block_items)
    for i in range(dense_block_len):

        log_debug("--------------")

        # 第i层DenseBlock
        end_point = 'DenseBlock_' + str(i)
        with tf.variable_scope(end_point):
            indata = dense_block(indata, i, filter_num, dense_block_items[i], is_training, dropout)
        end_points[end_point] = indata

        # 最后一层没有TransitionLayers
        if i >= (dense_block_len - 1):
            continue
        #

        # 输出缩小
        tl_filter_num = indata.get_shape().as_list()[3]  # 输出的filter
        tl_filter_num = int(tl_filter_num * (1 - reduction))
        log_debug(tl_filter_num)

        log_debug("--------------")

        # 第i层TransitionLayers
        end_point = 'TransitionLayers_' + str(i)
        with tf.variable_scope(end_point):
            indata = transition_layers(indata, i, tl_filter_num, is_training, dropout)
        end_points[end_point] = indata
    #

    log_debug("===================")

    # 全局平均池化(GlobalAveragePooling)
    end_point = 'GlobalAveragePooling'
    with tf.variable_scope(end_point):
        indata = slim.avg_pool2d(indata, indata.shape[1:3], stride=[1, 1], padding='VALID', scope=end_point)
        log_debug(indata)
    end_points[end_point] = indata

    # 全连接层
    end_point = 'FullyConnected'
    with tf.variable_scope(end_point):
        indata = slim.conv2d(inputs=indata, num_outputs=indata.shape[3], kernel_size=[1, 1], scope=end_point)
        log_debug(indata)
    end_points[end_point] = indata

    # 输出层
    end_point = 'Logits'
    with tf.variable_scope(end_point):
        logits = slim.conv2d(inputs=indata, num_outputs=classes, kernel_size=[1, 1], activation_fn=None,
                             scope='output_base')
        logits = tf.squeeze(logits, name=end_point)
    end_points[end_point] = logits
    end_points['Predictions'] = slim.softmax(logits, scope='Predictions')
    return logits, end_points


#


def dense_block(indata, i, filter_num, items_num, is_training, dropout=None):
    """
    每个DenseBlock
    参数:
        indata: 输入数据
        dense_block_items: 一个数组，包含每个dense_block数量
    """
    concat_indata = indata
    # 循环处理每个item
    for j in range(items_num):
        # item处理
        indata = dense_items(concat_indata, i, j, filter_num, is_training)
        # 合并
        concat_indata = tf.concat([concat_indata, indata], 3, name='Concat_DB_' + str(i) + "_" + str(j))
        log_debug(concat_indata.get_shape())
    #
    return concat_indata


#


def dense_items(indata, i, j, filter_num, is_training, dropout=None):
    """
    每一个DenseBlock的块
    参数:
        indata: 输入数据
        dense_block_items: 一个数组，包含每个dense_block数量
    """
    # 瓶颈层：1 * 1 卷积
    # BN −> ReLU -> Conv(1*1)
    indata = slim.batch_norm(inputs=indata, is_training=is_training, scope="BN_DB_" + str(i) + "_" + str(j) + "_a")
    log_debug(indata)
    indata = tf.nn.relu(features=indata, name="RuLU_DB_" + str(i) + "_" + str(j) + "_a")
    log_debug(indata)
    indata = slim.conv2d(inputs=indata,
                         num_outputs=filter_num,
                         kernel_size=[1, 1],
                         stride=1,
                         padding='SAME',
                         activation_fn=None,
                         scope="Conv_DB_" + str(i) + "_" + str(j) + "_a")
    log_debug(indata)

    # dropout处理
    if dropout:
        indata = tf.nn.dropout(indata, dropout)
    #

    # 卷积层： 3 * 3
    # BN −> ReLU −> Conv(3*3)
    indata = slim.batch_norm(inputs=indata, is_training=is_training, scope="BN_DB_" + str(i) + "_" + str(j) + "_b")
    log_debug(indata)
    indata = tf.nn.relu(features=indata, name="RuLU_DB_" + str(i) + "_" + str(j) + "_b")
    log_debug(indata)
    indata = slim.conv2d(inputs=indata,
                         num_outputs=filter_num,
                         kernel_size=[3, 3],
                         stride=1,
                         padding='SAME',
                         activation_fn=None,
                         scope="Conv_DB_" + str(i) + "_" + str(j) + "_b")
    log_debug(indata)

    # dropout处理
    if dropout:
        indata = tf.nn.dropout(indata, dropout)
    #

    #
    return indata


#


def transition_layers(indata, i, filter_num, is_training, dropout=None):
    """
    TransitionLayers层
    参数:
        indata: 输入数据
    """
    # BN −> Conv(1×1)
    indata = slim.batch_norm(inputs=indata, is_training=is_training, scope="BN_TL_" + str(i))
    log_debug(indata)
    indata = slim.conv2d(inputs=indata,
                         num_outputs=filter_num,
                         kernel_size=[1, 1],
                         stride=1,
                         padding='SAME',
                         activation_fn=None,
                         scope="Conv_TL_" + str(i))
    log_debug(indata)

    # dropout处理
    if dropout:
        indata = tf.nn.dropout(indata, dropout)

    # −> averagePooling(2×2)
    indata = slim.avg_pool2d(indata, [2, 2], stride=2, padding='VALID', scope="AvgP_TL_" + str(i))
    log_debug(indata)

    #
    return indata


#


# 批量处理
def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
            [slim.conv2d],
            weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_IN', uniform=False),
            activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


# 定义本网络处理的图片大小
densenet.default_image_size = 32
