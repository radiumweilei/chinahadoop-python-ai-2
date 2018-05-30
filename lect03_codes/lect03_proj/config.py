# -*- coding: utf-8 -*-

"""
    作者:     Robin
    版本:     1.0
    日期:     2018/05
    文件名:    config.py
    功能：     配置文件

    声明：小象学院拥有完全知识产权的权利；只限于善意学习者在本课程使用，
         不得在课程范围外向任何第三方散播。任何其他人或机构不得盗版、复制、仿造其中的创意，
         我们将保留一切通过法律手段追究违反者的权利
"""
import os

# 指定数据集路径
dataset_path = './data'

# 结果保存路径
output_path = './output'
if not os.path.exists(output_path):
    os.makedirs(output_path)

feat_cols = ['Age',	'Sex', 'Job', 'Housing', 'Saving accounts',
             'Checking account', 'Credit amount', 'Duration', 'Purpose']

label_col = 'Risk'

# 类别数据列中，类别整型转换字典
sex_dict = {
    'male':     0,
    'female':   1
}

housing_dict = {
    'free': 0,
    'rent': 1,
    'own':  2
}

saving_dict = {
    'little':        0,
    'moderate':      1,
    'quite rich':    2,
    'rich':          3
}

checking_dict = {
    'little':        0,
    'moderate':      1,
    'rich':          2
}

purpose_dict = {
    'radio/TV':             0,
    'education':            1,
    'furniture/equipment':  2,
    'car':                  3,
    'business':             4,
    'domestic appliances':  5,
    'repairs':              6,
    'vacation/others':      7
}

risk_dict = {
    'bad':  0,
    'good': 1
}
