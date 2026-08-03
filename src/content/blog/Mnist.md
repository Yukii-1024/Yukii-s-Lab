---
# ===== 必填 =====
title: 'MNIST-深度学习入门探索'
description: '使用numpy完成深度学习入门项目MNIST(手写数字分类)'
pubDate: '2026-07-31'           # 发布日期，格式 YYYY-MM-DD
category: 'notes'               # 分区三选一：notes=技术笔记 / fixes=疑难处置 / essays=随笔

# ===== 可选 =====
tags: ['深度学习', 'numpy', 'MNIST']                 # 标签，可写多个，不影响分区
updatedDate: '2026-08-03'     # 有修改时再填
# heroImage:                    # 封面图，一般用不到可删
---

*众所周知，MNIST是深度学习入门不得不品的一环,而为了学习深度学习原理，暂时先不使用Pytorch来进行编写，这里使用到numpy包 （一个提供矩阵、矩阵运算等数学操作的python包）来完成MNIST。*

## 前期准备
### 1.准备数据集
&emsp;&emsp;下载MNIST数据集并放置于项目文件夹中
### 2.构建项目环境
&emsp;&emsp;这里用到uv来进行包管理，在项目的根目录下初始化uv环境并添加numpy包
### ~~3.准备GPU&安装conda（并不需要，因为numpy并不支持gpu加速）~~
<br><br>
## 概述

&emsp;&emsp;MNIST 是一个手写数字数据集，70000 张 28×28 的灰度图，每张图上是 0~9 中的一个数字，其中 60000 张用于训练、10000 张用于测试。我们的目标很朴素：写一个神经网络，输入一张图片的 784 个像素值，输出它"觉得"这张图是哪个数字的概率分布。<br><br>&emsp;&emsp;整个流程分四步：数据加载与预处理 → 前向传播 → 反向传播 → 训练与评估。前向传播和反向传播是灵魂，把这两个搞懂了，后面用 Pytorch 基本就是查 API 的事。<br><br>&emsp;&emsp;本文的网络结构定为三层全连接（784 → 128 → 64 → 10），激活函数用 ReLU，输出层接 softmax，损失函数用交叉熵，优化器是最朴素的 mini-batch 梯度下降。~~没有动量、没有 Adam、没有学习率调度，祖传配方，包教包会~~

## 数据加载与预处理

### 1. 解析 IDX 文件
&emsp;&emsp;前期准备里下载的那四个文件是 IDX 格式的二进制文件：开头是几个 32 位大端整数（魔数、样本数量、图像高宽），后面跟着原始字节。用 struct 把它们拆出来：<br><br>

```python
import os
import struct

import numpy as np


def load_mnist(data_dir: str, kind: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """读取 MNIST 的 IDX 格式文件，返回 (图像, 标签)"""
    images_path = os.path.join(data_dir, f"{kind}-images.idx3-ubyte")
    labels_path = os.path.join(data_dir, f"{kind}-labels.idx1-ubyte")

    with open(images_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))  # 魔数 + 数量 + 高 + 宽
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)

    with open(labels_path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images, labels
```

&emsp;&emsp;`frombuffer` 不拷贝数据，直接复用文件缓冲区，实测读取整个训练集（60000 张）只要 0.02 秒。~~比某语言手写逐字节解析快了不知多少~~<br><br>

### 2. 预处理
&emsp;&emsp;拿到原始数据后还要做两件事：归一化和 one-hot 编码。<br><br>&emsp;&emsp;像素值是 0~255 的整数，直接喂给网络的话数值太大，梯度更新容易起飞。除以 255 压到 0~1 区间。<br><br>&emsp;&emsp;标签是 0~9 的整数，但分类任务的"标准答案"一般用 one-hot 向量表示：数字 3 变成 [0,0,0,1,0,0,0,0,0,0]，第 3 位是 1。这样和网络的概率输出形状对齐，算损失时可以直接逐元素运算。

```python
def preprocess(images: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    images = images.astype(np.float32) / 255.0   # 归一化
    labels = np.eye(10)[labels]                  # one-hot
    return images, labels
```

## 网络结构与参数初始化

&emsp;&emsp;网络结构长这样：<br><br>

- 输入层：784 个神经元（28×28 像素展平）
- 隐藏层1：128 个神经元，ReLU
- 隐藏层2：64 个神经元，ReLU
- 输出层：10 个神经元，softmax

&emsp;&emsp;为什么是 128 和 64？~~因为 1024 会把我的小破本风扇干到起飞~~ 没什么深奥理由，这个规模对 MNIST 来说完全够用。参数量一共 784×128 + 128 + 128×64 + 64 + 64×10 + 10 = 109386 个，纯 numpy 一个 epoch 只要一秒左右。<br><br>&emsp;&emsp;权重初始化用 He 初始化：均值为 0、标准差为 √(2/n_in) 的正态分布（n_in 是上一层神经元数）。如果全初始化为 0，同一层的神经元输入输出完全一样、梯度也一样，整个网络退化成单个神经元，永远学不出差异化；如果初始值太大，数值经过层层放大，输出直接爆炸。<br><br>

```python
class Net:
    def __init__(self, sizes: list[int]):
        self.sizes = sizes
        self.W = []
        self.b = []
        for i in range(len(sizes) - 1):
            # He 初始化：标准差 sqrt(2/n_in)，配合 ReLU 防止梯度消失/爆炸
            self.W.append(np.random.randn(sizes[i], sizes[i + 1]) * np.sqrt(2 / sizes[i]))
            self.b.append(np.zeros(sizes[i + 1]))
```

## 前向传播

&emsp;&emsp;前向传播就是数据从输入流到输出的过程。每一层做两件事：线性变换 z = xW + b，然后过激活函数。如果没有激活函数，多层线性变换叠起来还是线性变换——堆多少层都等价于一层，网络就白设计了，所以激活函数是"让网络变深有意义"的关键。<br><br>&emsp;&emsp;隐藏层用 ReLU：max(0, z)。它计算便宜，且正区间梯度恒为 1，不会像 sigmoid 那样在两端饱和——sigmoid 的导数在远离 0 的地方趋近于 0，多层一叠，梯度连乘几次就消失得无影无踪，训练基本停滞，这就是著名的梯度消失问题。<br><br>&emsp;&emsp;输出层要输出 10 个类的概率，用 softmax：softmax(z)_i = e^{z_i} / Σ_j e^{z_j}，把任意实数分数压成一组和为 1 的非负概率。代码里先减掉每行最大值再算 exp 是数值技巧，防止 e^{100} 这种大数溢出（减最大值不改变结果，因为分子分母同时缩放了相同的 e^{max}）。

```python
    def forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """返回 (每层激活值 acts, 每层线性输出 zs)，反向传播要用"""
        acts = [x]
        zs = []
        for W, b in zip(self.W[:-1], self.b[:-1]):
            z = acts[-1] @ W + b
            zs.append(z)
            acts.append(np.maximum(z, 0))        # ReLU
        z = acts[-1] @ self.W[-1] + self.b[-1]   # 输出层不接 ReLU
        zs.append(z)
        e = np.exp(z - z.max(axis=1, keepdims=True))   # 减最大值防溢出
        acts.append(e / e.sum(axis=1, keepdims=True))  # softmax
        return acts, zs
```

&emsp;&emsp;注意这里一次性处理整个 batch（x 是 n×784 的矩阵，@ 是矩阵乘法），比逐条 for 循环快得多——numpy 的矩阵乘法底层调的是 BLAS 优化过的 C 代码，这也是全程不用 Pytorch 也跑得动的原因。

## 损失函数

&emsp;&emsp;训练需要一把尺子衡量预测有多差，这把尺子就是损失函数。这里用交叉熵：L = -Σ_i y_i·log(ŷ_i)。由于 y 是 one-hot 的，求和只剩一项：L = -log(ŷ_true)，也就是"模型给正确答案分配的概率越低，损失越大"。训练的目标就是最小化它。<br><br>&emsp;&emsp;为什么不直接用均方误差？分类问题里交叉熵和 softmax 是天作之合：两者求导时那一堆 e^x 会全部约掉，输出层的梯度化简成 δ = ŷ - y，干净利落。用 MSE 的话梯度里会残留 σ'(z) 因子，输出接近 0 或 1 时梯度趋近于 0、学不动——又是饱和问题。

## 反向传播

&emsp;&emsp;前向传播算出了预测，反向传播负责算损失对每个参数的梯度，然后按梯度的反方向更新参数。核心武器是链式法则：从输出层开始，把误差逐层往回传。<br><br>&emsp;&emsp;从输出层出发，定义 δ = ∂L/∂z。得益于 softmax + 交叉熵的化简，输出层的 δ 就是：<br><br>&emsp;&emsp;δ_out = (ŷ - y) / m　（m 是 batch 大小，来自求均值时的 1/m）<br><br>&emsp;&emsp;接下来每一层只有三个式子：<br><br>

1. ∂L/∂W_i = a_iᵀ δ_i（a_i 是第 i 层的输入激活）
2. ∂L/∂b_i = δ_i 按 batch 维度求和
3. 把 δ 穿过激活函数传回上一层：δ_{i-1} = (δ_i W_iᵀ) ⊙ (z_{i-1} > 0)（ReLU 的导数是 0/1 掩码，正好逐元素相乘）

&emsp;&emsp;代码和公式一一对应：<br><br>

```python
    def backward(self, x: np.ndarray, y: np.ndarray, acts: list, zs: list):
        grad_w, grad_b = [], []
        # delta = (y_hat - y) / m，softmax+交叉熵的化简结果；m 来自 batch 均值的 1/m
        delta = (acts[-1] - y) / y.shape[0]
        for i in range(len(self.W) - 1, -1, -1):
            grad_w.append(acts[i].T @ delta)      # dL/dW = a^T delta
            grad_b.append(delta.sum(axis=0))      # dL/db = delta 按 batch 求和
            if i > 0:
                delta = (delta @ self.W[i].T) * (zs[i - 1] > 0)   # 穿过 ReLU 的掩码传回上一层
        self.grad_w = grad_w[::-1]                # 恢复成 W[0]..W[-1] 的顺序
        self.grad_b = grad_b[::-1]

    def update(self, lr: float):
        for i in range(len(self.W)):
            self.W[i] -= lr * self.grad_w[i]      # 梯度下降：往反方向走一步
            self.b[i] -= lr * self.grad_b[i]
```

&emsp;&emsp;写完反向传播先别急着训练——强烈建议先用数值梯度检查验证一遍：把某个参数扰动一个极小量 ε，用 (L(w+ε) - L(w-ε)) / 2ε 近似梯度，和反向传播算出来的解析梯度对比。我的实现实测最大误差 1.7e-10。~~别问我为什么知道要先检查，问就是被 1e+00 的误差毒打过~~ 这也是手写神经网络独有的乐趣：每一步都能拆开验证。

## 训练

&emsp;&emsp;训练循环就是反复执行四件事：取一小批数据（mini-batch）→ 前向传播 → 算损失 → 反向传播 → 更新参数。完整遍历一遍训练集叫一个 epoch，每个 epoch 开始前把数据打乱。<br><br>&emsp;&emsp;为什么用 mini-batch 而不是一次算全部数据？60000 张全算一次梯度太慢，而一小批样本的"平均梯度方向"已经足够接近真实梯度，算得快还能白嫖一点随机性帮助跳出局部最优。~~当然主要是快~~<br><br>

```python
def train(net: Net, X: np.ndarray, y: np.ndarray, X_val, y_val,
          epochs: int = 20, batch_size: int = 64, lr: float = 0.1):
    n = X.shape[0]
    for epoch in range(epochs):
        idx = np.random.permutation(n)            # 每个 epoch 打乱顺序
        total_loss = 0.0
        for start in range(0, n, batch_size):
            batch = idx[start:start + batch_size]
            xb, yb = X[batch], y[batch]
            acts, zs = net.forward(xb)
            loss = -np.log(acts[-1][np.arange(len(yb)), yb.argmax(axis=1)]).mean()
            net.backward(xb, yb, acts, zs)
            net.update(lr)
            total_loss += loss * len(yb)
        val_acc = accuracy(net, X_val, y_val)
        print(f"epoch {epoch + 1:>2d} | loss {total_loss / n:.4f} | val acc {val_acc:.4f}")
```

&emsp;&emsp;再加一个准确率函数：<br><br>

```python
def accuracy(net: Net, X: np.ndarray, y: np.ndarray) -> float:
    acts, _ = net.forward(X)
    return float((acts[-1].argmax(axis=1) == y.argmax(axis=1)).mean())
```

## 运行与结果

&emsp;&emsp;把前面的代码拼起来：加载数据 → 预处理 → 从训练集末尾切 5000 张当验证集（观察训练过程中的泛化情况，不参与训练）→ 初始化网络 → 开训：<br><br>

```python
X_train, y_train = load_mnist("data", "train")
X_test, y_test = load_mnist("data", "t10k")

X_train, y_train = preprocess(X_train, y_train)
X_test, y_test = preprocess(X_test, y_test)

# 从训练集切 5000 张当验证集
X_val, y_val = X_train[-5000:], y_train[-5000:]
X_train, y_train = X_train[:-5000], y_train[:-5000]

net = Net([784, 128, 64, 10])
train(net, X_train, y_train, X_val, y_val, epochs=20, batch_size=64, lr=0.1)
print("test acc:", accuracy(net, X_test, y_test))
```

&emsp;&emsp;跑 20 个 epoch，我的笔记本上总共 20.5 秒，输出长这样（随机初始化不同，数字会有细微浮动）：<br><br>

```
epoch  1 | loss 0.3406 | val acc 0.9626
epoch  2 | loss 0.1559 | val acc 0.9668
epoch  3 | loss 0.1109 | val acc 0.9702
...
epoch 19 | loss 0.0054 | val acc 0.9820
epoch 20 | loss 0.0044 | val acc 0.9836
test acc: 0.9792
```

&emsp;&emsp;测试集准确率 97.9%。~~虽然被各种框架的 99%+ 吊打，但这可是纯手搓的，意义不一样~~<br><br>&emsp;&emsp;还有个值得注意的细节：训练损失一路降到 0.004，但验证集准确率在 98% 附近就开始原地踏步甚至回退。这说明模型开始"背答案"而不是"学规律"——训练集上的损失还在降，泛化能力却没再涨，这就是过拟合的早期信号。想再往上提，可以加 Dropout、L2 正则，或者调小网络。

## 看看它学到了什么

&emsp;&emsp;光看数字没意思，把预测错的图画出来看看（matplotlib 是额外装的，不影响本文主角 numpy；Windows 上记得先配一下中文字体，不然标题全是方框）：<br><br>

```python
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

acts, _ = net.forward(X_test)
pred = acts[-1].argmax(axis=1)
wrong = np.where(pred != y_test.argmax(axis=1))[0][:8]

fig, axes = plt.subplots(2, 4)
for ax, i in zip(axes.flat, wrong):
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"真值 {y_test[i].argmax()} 预测 {pred[i]}")
    ax.axis("off")
plt.show()
```

&emsp;&emsp;我实测跑了一遍，测试集 10000 张里错了 202 张，混淆最多的组合是：4→9（11 张）、5→3（10 张）、2↔7（各 8 张）。全是人类自己不看半天也容易认错的连笔写法，所以也别太苛责它，~~毕竟它连手都没有~~。

## 总结

&emsp;&emsp;到这里，一个从零开始的神经网络就算完整跑通了。回头看，核心其实只有三件事：<br><br>

1. 前向传播：数据流过网络得到预测；
2. 损失函数：量化预测有多差；
3. 反向传播：用链式法则算出每个参数的梯度，然后梯度下降更新。

&emsp;&emsp;所谓"深度学习"，剥开来看就是矩阵乘法和链式法则。~~当然这句话仅限于入门阶段~~ 搞懂这些之后再去碰 Pytorch，会发现 nn.Linear、F.relu、loss.backward() 全是眼熟的老朋友——你已经在 numpy 里手写过它们了。<br><br>&emsp;&emsp;下一步可以折腾的方向：换激活函数（tanh / LeakyReLU / GELU）、加深网络、加正则化（Dropout / L2）、调学习率，或者直接上卷积神经网络——MNIST 的经典操作是 CNN 打天下。~~然后你就会发现，是时候上 Pytorch 了~~
