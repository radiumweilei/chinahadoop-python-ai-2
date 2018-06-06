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
import time
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV


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
    sns.countplot(x=config.label_col, data=train_data)

    plt.title('Training set')
    plt.xlabel('Class')
    plt.ylabel('Count')

    # 测试集
    plt.subplot(1, 2, 2, sharey=ax1)
    sns.countplot(x=config.label_col, data=test_data)

    plt.title('Testing set')
    plt.xlabel('Class')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()


def transform_data(train_data, test_data):
    """
        将类别型特征通过独热编码进行转换
        将数值型特征范围归一化到0-1
        使用PCA进行特征降维

        参数：
            - train_data:   DataFrame训练数据
            - test_data:    DataFrame测试数据

        返回：
            - X_train:  训练数据处理后的特征
            - X_test:   测试数据处理后的特征
    """
    # 独热编码处理类别特征
    encoder = OneHotEncoder(sparse=False)
    X_train_cat_feat = encoder.fit_transform(train_data[config.category_cols].values)
    X_test_cat_feat = encoder.transform(test_data[config.category_cols].values)

    # 范围归一化处理数值型特征
    scaler = MinMaxScaler()
    X_train_num_feat = scaler.fit_transform(train_data[config.num_cols].values)
    X_test_num_feat = scaler.transform(test_data[config.num_cols].values)

    # 合并所有特征
    X_train_raw = np.hstack((X_train_cat_feat, X_train_num_feat))
    X_test_raw = np.hstack((X_test_cat_feat, X_test_num_feat))

    print('特征处理后，特征维度为: {}（其中类别型特征维度为: {}，数值型特征维度为: {}）'.format(
        X_train_raw.shape[1], X_train_cat_feat.shape[1], X_train_num_feat.shape[1]))

    # 使用特征降维
    pca = PCA(n_components=0.99)
    X_train = pca.fit_transform(X_train_raw)
    X_test = pca.transform(X_test_raw)

    print('PCA特征降维后，特征维度为: {}'.format(X_train.shape[1]))

    return X_train, X_test


def train_test_model(X_train, y_train, X_test, y_test, model_name, model, param_range):
    """
        训练并测试模型
        model_name:
            kNN         kNN模型，对应参数为 n_neighbors
            LR          逻辑回归模型，对应参数为 C
            SVM         支持向量机，对应参数为 C
            DT          决策树，对应参数为 max_depth
            Stacking    将kNN, SVM, DT集成的Stacking模型， meta分类器为LR
            AdaBoost    AdaBoost模型，对应参数为 n_estimators
            GBDT        GBDT模型，对应参数为 learning_rate
            RF          随机森林模型，对应参数为 n_estimators

        根据给定的参数训练模型，并返回
        1. 最优模型
        2. 平均训练耗时
        3. 准确率
    """
    print('训练{}...'.format(model_name))
    clf = GridSearchCV(estimator=model,
                       param_grid=param_range,
                       cv=5,
                       scoring='accuracy',
                       refit=True)
    start = time.time()
    clf.fit(X_train, y_train)
    # 计时
    end = time.time()
    duration = end - start
    print('耗时{:.4f}s'.format(duration))

    # 验证模型
    train_score = clf.score(X_train, y_train)
    print('训练准确率：{:.3f}%'.format(train_score * 100))

    test_score = clf.score(X_test, y_test)
    print('测试准确率：{:.3f}%'.format(test_score * 100))
    print('训练模型耗时: {:.4f}s'.format(duration))
    print()

    return clf, test_score, duration

