# -*- coding: utf-8 -*-

"""
    作者:     Robin
    版本:     1.0
    日期:     2018/05
    文件名:    utils.py
    功能：     工具文件

    声明：小象学院拥有完全知识产权的权利；只限于善意学习者在本课程使用，
         不得在课程范围外向任何第三方散播。任何其他人或机构不得盗版、复制、仿造其中的创意，
"""
import matplotlib.pyplot as plt
import seaborn as sns
import config
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import time
import numpy as np


def clean_data(raw_data):
    """
        清洗数据

        参数：
            - raw_data: 原始数据

        返回：
            - cln_data: 清洗后的数据
    """
    # 替换数据中的空数据为'moderate'
    # 数据中只有Saving accounts和Checking account存在空值
    cln_data = raw_data.fillna('moderate')
    return cln_data


def inspect_dataset(train_data, test_data):
    """
        查看数据集

        参数：
            - train_data:   训练数据集
            - test_data:    测试数据集
    """
    print('\n===================== 数据查看 =====================')
    print('训练集有{}条记录。'.format(len(train_data)))
    print('测试集有{}条记录。'.format(len(test_data)))

    # 可视化各类别的数量统计图
    plt.figure(figsize=(10, 5))

    # 训练集
    ax1 = plt.subplot(1, 2, 1)
    sns.countplot(x='Risk', data=train_data)

    plt.title('Training set')
    plt.xlabel('Risk')
    plt.ylabel('Count')

    # 测试集
    plt.subplot(1, 2, 2, sharey=ax1)
    sns.countplot(x='Risk', data=test_data)

    plt.title('Testing set')
    plt.xlabel('Risk')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()


def transform_data(data_df):
    """
        将类别型特征、预测标签转换为整型数字

        参数：
            - data_df: DataFrame数据

        返回：
            - X:    转换后数据特征
            - y:    转换后的标签
    """
    trans_data_df = data_df.copy()

    trans_data_df['Sex'] = data_df['Sex'].map(config.sex_dict)
    trans_data_df['Housing'] = data_df['Housing'].map(config.housing_dict)
    trans_data_df['Saving accounts'] = data_df['Saving accounts'].map(config.saving_dict)
    trans_data_df['Checking account'] = data_df['Checking account'].map(config.checking_dict)
    trans_data_df['Purpose'] = data_df['Purpose'].map(config.purpose_dict)
    trans_data_df['Risk'] = data_df['Risk'].map(config.risk_dict)

    X = trans_data_df[config.feat_cols].values
    y = trans_data_df[config.label_col].values

    return X, y


def train_test_model(X_train, y_train, X_test, y_test, param_range, model_name):
    """
        训练并测试模型
        model_name:
            knn, kNN模型，对应参数为 n_neighbors
            lr, 逻辑回归模型，对应参数为 C

        根据给定的参数训练模型，并返回
        1. 最优模型
        2. 平均训练耗时
        3. 准确率
    """
    models = []
    scores = []
    durations = []

    for param in param_range:

        if model_name == 'kNN':
            print('训练kNN（k={}）...'.format(param), end='')
            model = KNeighborsClassifier(n_neighbors=param)
        elif model_name == 'LR':
            print('训练Logistic Regression（C={}）...'.format(param), end='')
            model = LogisticRegression(C=param)

        start = time.time()
        # 训练模型
        model.fit(X_train, y_train)

        # 计时
        end = time.time()
        duration = end - start
        print('耗时{:.4f}s'.format(duration), end=', ')

        # 验证模型
        score = model.score(X_test, y_test)
        print('准确率：{:.2f}%'.format(score * 100))

        models.append(model)
        durations.append(duration)
        scores.append(score)

    mean_duration = np.mean(durations)
    print('训练模型平均耗时{:.4f}s'.format(mean_duration))
    print()

    # 记录最优模型
    best_idx = np.argmax(scores)
    best_acc = scores[best_idx]
    best_model = models[best_idx]

    return best_model, best_acc, mean_duration

