# -*- coding: utf-8 -*-

import argparse
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from time import *
import json


def parse_args():
    """
    :return:进行参数的解析
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/data.json', help="The path of data.json")
    parser.add_argument('--epochs', default=1, help="The epochs of train")
    parser.add_argument('--batch_size', default=16, help="The batch-size of train")
    parser.add_argument('--early_stop', default=0, help="early stop for train or not")
    opt = parser.parse_args()
    return opt

def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    """

    :param data_dir:
    :param test_data_dir:
    :param img_height:
    :param img_width:
    :param batch_size:
    :return:
    """

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    # 加载测试集
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        label_mode='categorical',
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    # 返回处理之后的训练集、验证集和类名
    return train_ds, val_ds, class_names


# 构建CNN模型
def model_load(class_num, img_shape=(224, 224, 3)):
    """

    :param class_num:
    :param img_shape:
    :return:
    """
    # 搭建模型
    model = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=img_shape),# 对模型做归一化的处理，将0-255之间的数字统一处理到0到1之间
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),# 卷积层，输出为32个通道，卷积核的大小是3*3，激活函数为relu
        tf.keras.layers.MaxPooling2D(2, 2),# 添加池化层，kernel大小是2*2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),# 卷积层，输出为64个通道，卷积核大小为3*3，激活函数为relu
        tf.keras.layers.MaxPooling2D(2, 2),# 池化层，最大池化，对2*2的区域进行池化操作
        tf.keras.layers.Flatten(),# 将二维的输出转化为一维
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    model.summary()
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])# 指明模型的训练参数，优化器为sgd优化器，损失函数为交叉熵损失函数

    return model


# 训练过程曲线图
def show_loss_acc(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('results/results_cnn.png', dpi=100)


def train(configs, epochs, batch_size, early_stop):
    begin_time = time()
    train_ds, val_ds, class_names = data_load(configs['train'], configs['val'], configs['height'], configs['width'],
                                              batch_size)

    model = model_load(class_num=len(class_names), img_shape=(configs['height'], configs['width'], 3))

    if early_stop:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor=configs['early_stop']['monitor'],
                                             min_delta=configs['early_stop']['min_delta'],
                                             patience=configs['early_stop']['patience'],
                                             mode=configs['early_stop']['mode'])
        ]
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks,
                            verbose=configs['training']['verbose'], shuffle=configs['training']['shuffle'],
                            validation_freq=configs['training']['validation_freq'])
    else:
        history = model.fit(train_ds, validation_data=val_ds, epochs=35)
    model.save("models/cats_cnn.h5")
    # 记录结束时间
    end_time = time()
    run_time = end_time - begin_time

    print('run_time：', run_time, "s")  # 循环程序运行时间

    # 绘制过程图
    if not os.path.exists('results'):
        os.makedirs('results')
    show_loss_acc(history)

if __name__ == '__main__':
    args = parse_args()
    train(json.load(open(args.data, 'r',encoding='utf-8')), int(args.epochs), int(args.batch_size), bool(args.early_stop))