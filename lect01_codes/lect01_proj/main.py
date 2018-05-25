# -*- coding: utf-8 -*-

"""
    作者:     Robin
    版本:     1.0
    日期:     2018/05
    文件名:    main.py
    功能：     主程序

    实战案例1：利用NumPy实现梯度下降算法预测疾病

    任务：
        - 根据体重指数(BMI)和疾病发展的定量测量值(Y)使用梯度下降算法拟合出一条直线 y_hat = aX+b

    数据集来源：http://sklearn.apachecn.org/cn/0.19.0/sklearn/datasets/descr/diabetes.html

    案例文档：lect01_proj_readme.pdf

    声明：小象学院拥有完全知识产权的权利；只限于善意学习者在本课程使用，
         不得在课程范围外向任何第三方散播。任何其他人或机构不得盗版、复制、仿造其中的创意，
         我们将保留一切通过法律手段追究违反者的权利
"""
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = './data/diabetes.csv'


def load_data(data_file):
    """
        读取数据文件，加载数据
        参数：
            - data_file:    文件路径
        返回：
            - data_arr:     数据的多维数组表示
    """
    data_arr = np.loadtxt(data_file, delimiter=',', skiprows=1)
    return data_arr


def get_gradient(theta, x, y):
    m = x.shape[0]                                  # 样本个数 442
    y_estimate = x.dot(theta)                       # x: 442,2  theta: 1,2(2,1)  y: 442,1
    error = y_estimate - y                          # error: 442,1
    grad = 1.0 / m * error.dot(x)                   # error: 442,1(1,442) x: 442,2  grad: 1,2
    cost = 1.0 / (2 * m) * np.sum(error ** 2)       # cost 一个浮点数
    return grad, cost


def gradient_descent(x, y, max_iter=1500, alpha=0.01):
    theta = np.random.randn(2)                      # theta: 1,2

    # 收敛阈值
    tolerance = 1e-3

    # Perform Gradient Descent
    iterations = 1

    is_converged = False
    while not is_converged:
        grad, cost = get_gradient(theta, x, y)
        new_theta = theta - alpha * grad            # theta: 1,2  grad: 1,2  new_theta: 1,2

        # Stopping Condition
        if np.sum(abs(new_theta - theta)) < tolerance:
            is_converged = True
            print('参数收敛')

        # Print error every 50 iterations
        if iterations % 10 == 0:
            print('第{}次迭代，损失值 {:.4f}'.format(iterations, cost))

        iterations += 1
        theta = new_theta

        if iterations > max_iter:
            is_converged = True
            print('已至最大迭代次数{}'.format(max_iter))

    return theta


def main():
    """
        主函数
    """
    data_arr = load_data(DATA_PATH)
    x = data_arr[:, 0].reshape(-1, 1)    # diabetes.csv 中的 BMI 列
    # 添加一列全1的向量
    x = np.hstack((np.ones_like(x), x))  # 442行 2列
    y = data_arr[:, 1]                   # 442行 1列

    theta = gradient_descent(x, y, alpha=0.001, max_iter=200)
    print('线型模型参数', theta)

    # 绘制结果
    y_pred = theta[0] + theta[1] * x[:, 1]
    plt.figure()
    # 绘制样本点
    plt.scatter(x[:, 1], y)

    # 绘制拟合线
    plt.plot(x[:, 1], y_pred, c='red')
    plt.show()


if __name__ == '__main__':
    main()

