# -*- coding: utf-8 -*-

"""
    作者:     Robin
    版本:     1.0
    日期:     2018/05
    文件名:    main.py
    功能：     主程序

    实战案例3-1：贷款违约预测 (1)
    任务：使用scikit-learn建立不同的机器学习模型进行贷款违约预测

    数据集来源： https://www.kaggle.com/kabure/german-credit-data-with-risk

    声明：小象学院拥有完全知识产权的权利；只限于善意学习者在本课程使用，
         不得在课程范围外向任何第三方散播。任何其他人或机构不得盗版、复制、仿造其中的创意，
         我们将保留一切通过法律手段追究违反者的权利
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import config
import utils


def main():
    """
        主函数
    """
    # 加载数据
    raw_data = pd.read_csv(os.path.join(config.dataset_path, 'german_credit_data.csv'))

    # 清洗数据
    cln_data = utils.clean_data(raw_data)

    # 分割数据集
    train_data, test_data = train_test_split(cln_data, test_size=1 / 4, random_state=10)

    # 数据查看
    utils.inspect_dataset(train_data, test_data)

    # 特征工程
    print('\n===================== 特征工程 =====================')
    X_train, y_train = utils.transform_data(train_data)
    X_test, y_test = utils.transform_data(test_data)

    # 构建训练测试数据
    # 数据建模及验证
    print('\n===================== 数据建模及验证 =====================')
    model_name_param_dict = {'kNN': [5, 11, 15],
                             'LR': [0.01, 1, 100]}

    # 比较结果的DataFrame
    results_df = pd.DataFrame(columns=['Accuracy (%)', 'Time (s)'],
                              index=list(model_name_param_dict.keys()))
    results_df.index.name = 'Model'
    for model_name, param_range in model_name_param_dict.items():
        _, best_acc, mean_duration = utils.train_test_model(X_train, y_train, X_test, y_test,
                                                            param_range, model_name)
        results_df.loc[model_name, 'Accuracy (%)'] = best_acc * 100
        results_df.loc[model_name, 'Time (s)'] = mean_duration

    results_df.to_csv(os.path.join(config.output_path, 'model_comparison.csv'))

    # 模型及结果比较
    print('\n===================== 模型及结果比较 =====================')

    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    results_df.plot(y=['Accuracy (%)'], kind='bar', ylim=[60, 100], ax=ax1, title='Accuracy(%)', legend=False)

    ax2 = plt.subplot(1, 2, 2)
    results_df.plot(y=['Time (s)'], kind='bar', ax=ax2, title='Time(s)', legend=False)
    plt.tight_layout()
    plt.savefig(os.path.join(config.output_path, 'pred_results.png'))
    plt.show()


if __name__ == '__main__':
    main()
