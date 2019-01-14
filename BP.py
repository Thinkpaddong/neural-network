#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:37:25 2018

@author:DZY
"""

from random import seed
from random import randrange
from random import random
from math import exp
from get_features import load_data
from cfg import home_root
from os.path import join
import numpy as np
import matplotlib.pyplot as plt


class BP_Network():
    # 初始化神经网络
    def __init__(self, n_inputs, n_hidden, n_outputs,load):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.number=0
        self.sumerror=[];
        if load==False:
            self.network = list()
            hidden_layer = [{'weights': [random() for i in range(self.n_inputs + 1)]} for i in range(self.n_hidden)]
            self.network.append(hidden_layer)
            output_layer = [{'weights': [random() for i in range(self.n_hidden + 1)]} for i in range(self.n_outputs)]
            self.network.append(output_layer)
        else:
            self.network =list(np.load(join(home_root,'data_network1.npy')))

    # 计算神经元的激活值（加权之和）
    def activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights) - 1):
            activation += weights[i] * inputs[i]                #标签都没动
        return activation

    # 定义激活函数
    def transfer(self, activation):
        return 1.0 / (1.0 + exp(-activation))

    # 计算神经网络的正向传播
    def forward_propagate(self, row):
        '''
        
        :param row: 每个样本
        :return: 
        '''
        inputs = row
        #print(inputs)     #数据正常
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = self.activate(neuron['weights'], inputs)
                neuron['output'] = self.transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs                 #数据改变
        return inputs

    # 计算激活函数的导数
    def transfer_derivative(self, output):
        return output * (1.0 - output)

    # 反向传播误差信息，并将纠偏责任存储在神经元中
    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            if i != len(self.network) - 1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['responsibility'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['responsibility'] = errors[j] * self.transfer_derivative(neuron['output'])

    # 根据误差，更新网络权重
    def _update_weights(self, row):
        for i in range(len(self.network)):
            inputs = row[:-1]                                                           #这里除去了标签
            #print(inputs)
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += self.l_rate * neuron['responsibility'] * inputs[j]
                neuron['weights'][-1] += self.l_rate * neuron['responsibility']         #权重更新

    # 根据指定的训练周期训练网络
    def train_network(self, train):
        for epoch in range(self.n_epoch):           #迭代次数
            sum_error = 0
            for row in train:
                outputs = self.forward_propagate(row)
                expected = [0 for i in range(self.n_outputs)]
                expected[row[-1]] = 1
                sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
                self.backward_propagate_error(expected)
                self._update_weights(row)
            print(' 第%s次的误差总和=%.3f' % (self.number+1,sum_error))
            self.sumerror.append(sum_error)
            self.number = self.number + 1
            #print('>周期=%d, 误差=%.3f' % (epoch, sum_error))

    # 利用训练好的网络，预测“新”数据
    def predict(self, row):
        outputs = self.forward_propagate(row)
        return outputs.index(max(outputs))

    # 利用随机梯度递减策略，训练网络
    def back_propagation(self, train, test):
        #print(len(train))
        #print(len(test))
        self.train_network(train)
        predictions = list()
        for row in test:
            prediction = self.predict(row)
            predictions.append(prediction)
        return (predictions)

    # 将数据库分割为 k等份
    def cross_validation_split(self, n_folds):
        dataset_split = list()
        dataset_copy = self.dataset
        fold_size = int(len(self.dataset) / n_folds)
                                                        #print(fold_size)

        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                                                        #print(index)
                fold.append(dataset_copy.pop(index))
                                                        #print(dataset_copy[index])     #用于确定一张图的情形
            dataset_split.append(fold)
        return dataset_split

    # 用预测正确百分比来衡量正确率
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    # 用每一个交叉分割的块（训练集合，试集合）来评估BP算法
    def evaluate_algorithm(self, dataset, n_folds, l_rate, n_epoch):
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.dataset = dataset
        folds = self.cross_validation_split(n_folds)
        scores = list()
        for fold in folds:                      #循环n_folds次
            train_set = list(folds)
            train_set.remove(fold)              #移出1个文件，含4800/n_folds给test
            train_set = sum(train_set, [])      #剩下的4800-4800/n_folds个图训练
            test_set = list()
            for row in fold:                    #无标签数据，做测试
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = self.back_propagation(train_set, test_set)      #给出预测值
            actual = [row[-1] for row in fold]                          #读出标签
            accuracy = self.accuracy_metric(actual, predicted)          #计算准确率
            print('训练集中测试算法性能得到的百分比: %s%%'% accuracy)
            scores.append(accuracy)
        return scores,self.sumerror

    def predict_image(self, dataset_test):
        predict_list=[]
        label_list=[]
        error_number = 0
        error_number_single = 0
        number = len(dataset_test)/4
        for row in dataset_test:
            predict_vaule = self.predict(row)
            predict_list.append(predict_vaule)
            label_list.append(row[-1])
        for k in range(len(dataset_test)//4):
            for i in range(4):
                if predict_list[4*k+i] != label_list[4*k+i]:
                    error_number_single = 1
            if error_number_single != 0:
                error_number = error_number + 1
            error_number_single=0
            accury=(number-error_number)/number*100
        print("验证码的准确率: %f%%" % accury )

    def draw_point(self,X_coordinate, Y_coordinate):
        x = X_coordinate
        y = Y_coordinate
        plt.xlim(xmax=len(x), xmin=0)
        plt.ylim(ymax=1, ymin=0)
        plt.figure(figsize=(8, 4))  # 整个现实图（框架）的大小
        plt.plot(x, y, 'r', linewidth=1)
        plt.xlabel('iteration number')
        plt.ylabel("accuracy")
        plt.show()




if __name__ == '__main__':
    seed(2)
    (dataset,dataset_test) = load_data(join(home_root,'train_data.npy'),join(home_root, 'test_data.npy'))
                                                     # 设置网络初始化参数
    n_inputs = len(dataset[0]) - 1
    n_hidden = 15
    n_outputs = len(set([row[-1] for row in dataset]))
    BP = BP_Network(n_inputs, n_hidden, n_outputs,False)
    l_rate = 1
    n_folds = 200                                    #每个文件有960个数据
    n_epoch = 1                                      #垫带次数在一个4800-4800/n_folds训练集上
    scores,sumerror = BP.evaluate_algorithm(dataset, n_folds, l_rate, n_epoch)
    print('评估算法正交验证得分: %s%%' % scores)       #5次的准确率
    print('得分: %s%%' % (sum(scores[189:199])/float(len(scores[189:199]))))
    BP.predict_image(dataset_test)
    np.save(join(home_root,'data_network1.npy'),BP.network)
    sumerror1 = [c/float(sumerror[0]) for c in sumerror]
    BP.draw_point(np.array(range(n_folds)),np.array(sumerror1))

