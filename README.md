项目内容：用NumPy实现神经网络训练
====

简介
----
这个项目实现了一个两层的全连接神经网络，用于分类MNIST手写数字数据集。它使用relu激活函数和交叉熵损失及L2正则化函数，并使用梯度下降算法进行优化。

用法
---
该项目使用NumPy实现了两层的全连接神经网络。运行train.py以训练并测试模型。可以通过调整参数来优化模型，例如学习率，隐藏层大小和正则化强度。

# 0. 数据准备
请从MNIST官网http://yann.lecun.com/exdb/mnist/ 下载数据并解压，放置于代码同一目录下。


# 1. 训练代码
代码中包括激活函数，反向传播，loss以及梯度的计算，学习率下降策略，L2正则化，优化器SGD，保存模型几部分。
若要使用给定的超参数训练神经网络，请使用以下命令：
```
python train.py --learning_rate 0.01 --hidden_size 100 --reg_strength 0.01 --num_epochs 50
```
您可根据需要自定义超参数。

# 2. 参数查找代码
代码中对学习率，隐藏层大小，正则化强度三者进行了参数查找，固定epoch为150.
若想对超参数执行网格搜索并使用找到的最佳超参数训练神经网络，请使用以下命令：
```
python search.py --learning_rates 0.01 0.001 0.0001 --hidden_sizes 50 100 200 --reg_strengths 0.01 0.001 0.0001
```
这将使用指定的超参数执行网格搜索，并使用找到的最佳超参数训练神经网络。 您可以根据需要修改超参数和搜索设置。

# 3. 测试代码
要加载保存的模型checkpoint并评估其在测试集的性能，请使用以下命令：
```
python test.py --checkpoint_path ./best_model.npz --hidden_size 64
```
您可以根据需要修改文件路径。
