# -*- coding: utf-8 -*- 

__author__ = 'Fan'
__institute__ = 'SYSU'

import datetime as dt
import os
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.utils.vis_utils import plot_model


class Model:
    """

    """
    def __init__(self):
        self.model = Sequential()

    # 加载模型
    def load_model(self, filepath):
        print "[Model] Loading model from file %s" % filepath
        self.model = load_model(filepath)

    # 生成模型
    def build_model(self, configs):
        print "[Model] Building model from configs"

        # 模型添加隐藏层
        for layer in configs['model']['layers']:
            type = layer['type'] if 'type' in layer else None
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['dropout_rate'] if 'dropout_rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None

            # 可添加
            if type == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if type == 'dropout':
                self.model.add(Dropout(dropout_rate))

        learning_rate = configs['optimizer']['learning_rate']
        loss = configs['optimizer']['loss']
        optimizer_type = configs['optimizer']['type']

        # 设置优化器
        # 可添加, 默认使用Adam
        if optimizer_type == 'sgd':
            optimizer = SGD(lr=learning_rate)
        else:
            optimizer = Adam(lr=learning_rate)

        # 模型编译
        self.model.compile(optimizer, loss=loss)

        # 模型可视化
        plot_model(self.model, to_file='model.png', show_shapes=True)

        print "[Model] Model compiled"

    # 生成孪生模型
    # def build_siamese_model(self, configs):

    # 训练模型
    def train_model(self, x, y, epochs, batch_size, save_dir):
        print "[Model] Training model"
        print "[Model] %s epochs, %s batch_size" % (epochs, batch_size)

        save_fname = str(os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs))))

        # 设置回调函数, 保存最佳模型
        callbacks = [
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]

        # 模型训练
        self.model.fit(
            x,
            y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )

        # 模型保存
        self.model.save(save_fname)

        print "[Model] Model trained"

    # 测试模型
    def test_model(self, x):
        print "[Model] Testing model"

        # 模型测试
        prediction = self.model.predict(x)
        return prediction
