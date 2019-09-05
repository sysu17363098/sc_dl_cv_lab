# -*- coding: utf-8 -*- 

__author__ = 'Fan'
__institute__ = 'SYSU'

import json
from core.model import Model


def main():
    configs = json.load(open('config.json', 'r'))

    # 加载数据
    X_train, y_train, X_test, y_test = None

    # 生成模型
    # load_model和build_model方法二选一
    model = Model()
    # model.load_model(filepath='')
    model.build_model(configs)

    # 训练模型
    model.train_model(
        x=X_train,
        y=y_train,
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        save_dir=configs['model']['save_dir'])

    # 测试模型
    prediction = model.test_model(
        x=X_test
    )

    print "Test data true label is : %s" % y_test
    print "Model output is : %s" % prediction


if __name__ == "__main__":
    main()
