# -*- coding: utf-8 -*-

"""
    作者:     Robin
    版本:     1.0
    日期:     2018/06
    文件名:    config.py
    功能：     配置文件

    声明：小象学院拥有完全知识产权的权利；只限于善意学习者在本课程使用，
         不得在课程范围外向任何第三方散播。任何其他人或机构不得盗版、复制、仿造其中的创意，
         我们将保留一切通过法律手段追究违反者的权利
"""
import os

# 指定数据集路径
dataset_path = './data'

# 训练集路径
train_data_file = os.path.join(dataset_path, 'fashion-mnist_train.csv')

# 测试集路径
test_data_file = os.path.join(dataset_path, 'fashion-mnist_test.csv')

# 结果保存路径
output_path = './output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 图像大小
img_rows, img_cols = 28, 28

