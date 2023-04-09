# @File : main.py.py
# -*- coding: utf-8 -*-
# @Time   : 2023/4/7 3:35 下午
# @Author : Shijie Zhang
# @Software: PyCharm

import numpy as np
import struct
import os
import argparse

# 构建命令行参数解析器
parser = argparse.ArgumentParser(description='Search for best hyperparameters for the neural network')

parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rates to search over')
parser.add_argument('--hidden_size', default=64, type=int, help='hidden sizes to search over')
parser.add_argument('--reg_strength', default=0.001, type=float, help='regularization strengths to search over')
parser.add_argument('--num_epochs', default=150, type=int, help='number of epochs to train each model')

args = parser.parse_args()

# 加载数据集
def load_mnist_train(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

def load_mnist_test(path, kind='t10k'):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels

path = '/Users/bao/Desktop/numpy_nn/'
X_train, y_train = load_mnist_train(path) # (10000, 784), (10000,)
X_test, y_test = load_mnist_test(path)    # (60000, 784), (60000,)

# 归一化
X_train = X_train/255.0
X_test = X_test/255.0

# 设置超参数
input_size = X_train.shape[1]
hidden_size = 30
output_size = np.max(y_train) + 1
learning_rate = 0.1
reg = 1e-3
num_epochs = 100
batch_size = 64

# 创建神经网络
# 定义神经网络类
class NeuralNet:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重参数
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        # 前向传播
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)  # ReLU 激活函数
        self.z2 = self.a1.dot(self.W2) + self.b2
        exp_scores = np.exp(self.z2)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def backward(self, X, y, learning_rate, reg):
        # 反向传播
        num_examples = X.shape[0]
        delta3 = self.probs
        delta3[range(num_examples), y] -= 1
        delta2 = delta3.dot(self.W2.T) * (self.a1 > 0)  # ReLU 激活函数的导数
        dW2 = (self.a1.T).dot(delta3) / num_examples + reg * self.W2
        db2 = np.sum(delta3, axis=0) / num_examples
        dW1 = np.dot(X.T, delta2) / num_examples + reg * self.W1
        db1 = np.sum(delta2, axis=0) / num_examples

        # 更新权重参数
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X_train, y_train, learning_rate, num_iterations, regularization_strength):
        # 将数据集随机分成若干个batch
        permutation = np.random.permutation(X_train.shape[0])
        X_batches = np.array_split(X_train[permutation], X_train.shape[0] // batch_size)
        y_batches = np.array_split(y_train[permutation], X_train.shape[0] // batch_size)

        for i in range(num_iterations):
            # 学习率下降策略
            learning_rate *= (1 - (i / float(num_iterations))) * 0.5 + 0.5

            # 在每个 batch 上进行训练
            for batch in range(len(X_batches)):
                #net.forward(X_batches[batch])
                #net.backward(X_batches[batch], y_batches[batch], regularization_strength)
                # 前向传播
                output = self.forward(X_batches[batch])
                # 反向传播
                self.backward(X_batches[batch], y_batches[batch], learning_rate, regularization_strength)

            # 每10次迭代打印一次损失函数值
            if i % 10 == 0:
                #loss = np.mean(-(y * np.log(output) + (1 - y) * np.log(1 - output)))
                loss = self.get_loss(X_train, y_train, reg)
                print("Loss after iteration {}: {}".format(i, loss))

        # 返回训练后的神经网络
        return self

    def predict(self, X):
        # 预测标签
        self.forward(X)
        return np.argmax(self.probs, axis=1)

    def get_loss(self, X, y, reg):
        # 计算损失函数值
        num_examples = X.shape[0]
        self.forward(X)
        data_loss = -np.sum(np.log(self.probs[range(num_examples), y])) / num_examples
        reg_loss = 0.5 * reg * (np.sum(np.square(self.W1)) + np.sum(np.square(self.W2)))
        return data_loss + reg_loss

    def get_accuracy(self, X, y):
        # 计算准确率
        return np.mean(self.predict(X) == y)

# 初始化network
nn = NeuralNet(input_size, args.hidden_size, output_size)
# 训练
nn.train(X_train, y_train, args.learning_rate, args.num_epochs, args.reg_strength)

# 在验证集上测试准确率
y_pred = np.round(nn.forward(X_test))
accuracy = nn.get_accuracy(X_test, y_test)
print('acc on test dataset:', accuracy)

# 保存模型参数
np.savez('./model_lr{}_hs_rs_.npz'.format(args.learning_rate,args.hidden_size, args.reg_strength), W1=nn.W1, b1=nn.b1, W2=nn.W2, b2=nn.b2)
