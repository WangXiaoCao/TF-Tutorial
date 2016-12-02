#!/usr/bin/python
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import os
import tensorflow as tf

# 定义图像大小
Image_SIZE = 24

# 设置描述数据集的全局变量
NUM_CLASSES = 10  # 中共10个类别
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500000  # 共50000个训练样本
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000  # 共10000个测试样本


def read_cifar10(filename_queue):


    class CIFAR10Record():
        pass
    result = CIFAR10Record()

    # 输入数据的格式
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes

    # 根据文件名读取数据啦
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # 将数据string类型转变为uint8的向量类型，向量打长度是record_bytes的长度
    record_bytes = tf.decode_raw(value, tf.uint8)

    # 将第一字节：label，转变成int32
    result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

    # 将剩余的字节：image从本来的height*width*depth转变成(height, width, depth)
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])
    # 变一下顺序
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    # 返回的格式：[label, [height, width, depth]]
    return result


# # 创建由n组batch data组成的样本数据
def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size= batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples
        )
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples +3 * batch_size
        )

    tf.image_summary('images', images)

    # 返回的是图像数据，标签
    return images, tf.reshape(label_batch, [batch_size])


def distorted_input(data_dir, batch_size):
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrang(1, 6)]

























