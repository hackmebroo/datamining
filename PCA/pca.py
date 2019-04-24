import numpy as np
import pandas as pd
import xlrd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def excel_to_matrix(path):
    table = xlrd.open_workbook(path).sheets()[0]  # 获取第一个sheet表
    row = table.nrows  # 行数
    col = table.ncols  # 列数
    datamatrix = np.zeros((row, col))  # 生成一个nrows行ncols列，且元素均为0的初始矩阵
    for x in range(col):
        cols = np.matrix(table.col_values(x))  # 把list转换为矩阵进行矩阵操作
        datamatrix[:, x] = cols  # 按列把数据存进矩阵中
    # print(datamatrix)
    return datamatrix


datafile = u'E:\\dataset\\dataset.xlsx'
X = excel_to_matrix(datafile)
pca = PCA(n_components=2)   #降到2维
pca.fit(X)                  #训练
newX = pca.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)
# print(pca.explained_variance_ratio_)  #输出贡献率
# print(newX)

# plt.scatter(newX[:, 0], newX[:, 1], marker='o')
# plt.show()
# Y = np.zeros((500, 1))
# plt.scatter(newX[:, 0], Y, marker='o')
# plt.savefig("PCA一维.png")
# plt.savefig("PCA二维.png")
# plt.show()

for i in range(500):
    if i < 100:
        plt.scatter(newX[i, 0], newX[i, 1], marker='o', color='red')
    elif i < 300:
        plt.scatter(newX[i, 0], newX[i, 1], marker='o', color='yellow')
    elif i < 500:
        plt.scatter(newX[i, 0], newX[i, 1], marker='o', color='blue')

# plt.scatter(Y2[:, 0], Y2[:, 1], marker='o')
plt.savefig("PCA二维.png")
plt.show()

# plt.scatter(Y1[:, 0], tmp, marker='o')
# plt.savefig("PCA一维.png")
# plt.show()
