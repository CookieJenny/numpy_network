# @File : test.py.py 
# -*- coding: utf-8 -*-
# @Time   : 2023/4/9 6:31 下午 
# @Author : Shijie Zhang
# @Software: PyCharm

import numpy as np
import struct
import os
import argparse

batch_size = 64

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

path = './'
X_train, y_train = load_mnist_train(path) # (10000, 784), (10000,)
X_test, y_test = load_mnist_test(path)    # (60000, 784), (60000,)

# 归一化
X_train = X_train/255.0
X_test = X_test/255.0

class NeuralNet:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重参数
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    def load_para(self, para):
        self.W1 = para['W1']
        self.b1 = para['b1']
        self.W2 = para['W2']
        self.b2 = para['b2']

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
        # 将数据集随机分成若干个 batch
        permutation = np.random.permutation(X_train.shape[0])
        X_batches = np.array_split(X_train[permutation], X_train.shape[0] // batch_size)
        y_batches = np.array_split(y_train[permutation], X_train.shape[0] // batch_size)

        train_losses = []
        test_losses = []
        test_accs = []

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

            # 计算训练和测试集的损失函数值和准确率
            train_loss = self.get_loss(X_train, y_train, regularization_strength)
            test_loss = self.get_loss(X_test, y_test, regularization_strength)
            test_acc = self.get_accuracy(X_test, y_test)

            # 存储损失函数值和准确率
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

            # 每10次迭代打印一次损失函数值
            if i % 10 == 0:
                #loss = np.mean(-(y * np.log(output) + (1 - y) * np.log(1 - output)))
                loss = self.get_loss(X_train, y_train, regularization_strength)
                print("Loss after iteration {}: {}".format(i, loss))

        # 返回训练后的神经网络
        return self, train_losses, test_losses, test_accs

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


input_size = X_train.shape[1]
output_size = np.max(y_train) + 1

args = argparse.ArgumentParser()
args.add_argument('--checkpoint_path', type=str, default='./model.npz')
args.add_argument('--hidden_size', type=int, default=64)
opt = args.parse_args()

net = NeuralNet(input_size, opt.hidden_size, output_size)

# 载入参数
para = np.load(opt.checkpoint_path)
net.load_para(para)

# 测试
acc = net.get_accuracy(X_test, y_test)
print('acc from best model:', acc)
