# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:53:12 2018

@author: Administrator
"""

from Feature_scaling import *
from KNN_algr import classify_KNN

'''创建一个训练数据集'''


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def Test_accuray(split_ratio, test_set, test_label):
    """
    书上的测试函数没有参数，是自适应函数此处传入分割参数以及测试集，可以修改测试数值（使用书上的0.1作为分割率）
    :param split_ratio: 此处传入分割参数以及测试集，可以修改测试数值（使用书上的0.1作为分割率）
    :param test_set: 数据
    :param test_label: 数据 lavel
    :return:
    """
    norm_test, ranges, Min = Norm_feature(test_set)
    rows = norm_test.shape[0]
    rows_test = int(rows * split_ratio)

    error = 0

    for i in range(rows_test):
        Result = classify_KNN(norm_test[i, :], norm_test[rows_test:rows], test_label[rows_test:rows], 4)
        # 参数1表示从测试集（此处约会数据是随机的因此抽取前10%即可）中抽取一个实例
        # 参数2，3，4使用后90%作为训练数据，为输入的实例进行投票并分类，K=4
        # 测试可知，当K=4时，结果比较好

        # print("the classifier came with: {}, the real answer is :{}".format(Result, test_label[i]))
        if (Result != test_label[i]): error += 1
        # print(type(error)) #for test

    print("the accuracy is %f | the error_rate is %f " % \
          (1 - (float(error) / float(rows_test)), (float(error) / float(rows_test))))
