import numpy as np
import struct
import os
import matplotlib.pyplot as plt
from tqdm import trange

import argparse
import itertools

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

# 设置超参数
input_size = X_train.shape[1]
output_size = np.max(y_train) + 1
num_epochs = 150
batch_size = 64

# 创建神经网络
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

# 定义参数搜索函数
def search_params(X, y, learning_rates, hidden_sizes, regularization_strengths):
    best_accuracy = -1
    best_nn = None
    best_train_losses = None
    best_test_losses = None
    best_test_acces = None
    results = {}

    for learning_rate in learning_rates:
        for hidden_size in hidden_sizes:
            for regularization_strength in regularization_strengths:
                print('--------start training with learning rate:{}, hidden size:{}, regularization strength:{}--------------'.format(learning_rate, hidden_size, regularization_strength))
                # 创建神经网络实例
                nn = NeuralNet(input_size, hidden_size, output_size)

                # 训练神经网络
                _, train_losses, test_losses, test_accs = nn.train(X, y, learning_rate, num_epochs, regularization_strength)

                # 在验证集上测试准确率
                y_pred = np.round(nn.forward(X_test))
                accuracy = nn.get_accuracy(X_test, y_test)
                print('test acc:', accuracy)

                # 记录当前参数下的准确率和神经网络实例
                results[(learning_rate, hidden_size, regularization_strength)] = accuracy

                # 更新最佳准确率和神经网络实例
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_nn = nn
                    best_train_losses = train_losses
                    best_test_losses = test_losses
                    best_test_acces = test_accs

            # 返回最佳神经网络和参数组合
        return best_nn, max(results, key=results.get), best_train_losses, best_test_losses, best_test_acces


# 构建命令行参数解析器
parser = argparse.ArgumentParser(description='Search for best hyperparameters for the neural network')

parser.add_argument('--learning_rates', default=[0.5, 0.1, 0.01], type=float, help='learning rates to search over')
parser.add_argument('--hidden_sizes', default=[16, 64, 128], type=int, help='hidden sizes to search over')
parser.add_argument('--reg_strengths', default=[0.01, 0.001, 0.1], type=float, help='regularization strengths to search over')
parser.add_argument('--num_epochs', default=150, type=int, help='number of epochs to train each model')

args = parser.parse_args()

# 定义超参数搜索空间
learning_rates = list(args.learning_rates)
hidden_sizes = list(args.hidden_sizes)
reg_strengths = list(args.reg_strengths)

# 进行参数搜索
best_nn, best_params, best_train_losses, best_test_losses, best_test_acces = search_params(X_train, y_train, learning_rates, hidden_sizes, reg_strengths)

# 输出最佳参数组合和对应的准确率
print("Best parameters: ", best_params)
print("Best accuracy: ", best_nn.get_accuracy(X_test, y_test))

# 保存模型
np.savez('./best_model.npz', W1=best_nn.W1, b1=best_nn.b1, W2=best_nn.W2, b2=best_nn.b2)

# 可视化训练和测试的损失函数值和准确率
fig, ax = plt.subplots(2, figsize=(10, 8))
ax[0].plot(best_train_losses, label='Train Loss')
ax[0].plot(best_test_losses, label='Test Loss')
ax[0].legend()
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[1].plot(best_test_acces, label='Test acc')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
plt.show()

# 可视化每层的权重参数
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs = axs.ravel()
for i in range(2):
    axs[i].imshow(best_nn.__dict__[f'W{i + 1}'], cmap='gray')
    axs[i].set_title(f'Layer {i + 1} Weights')
plt.show()
