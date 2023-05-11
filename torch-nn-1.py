#nn和funcitonal的区别，就是nn功能更加强大， 而functional说白了就是一个函数库
import torch.nn as nn    # torch神经网络库
import torch.nn.functional as F
import torch as t

#1、torch.nn.Linear 线性
# para in_features，out_features,bias
#这里就看出了，我们的nn是用类进行封装的；我们调用linear，实际是返回的是一个实例
module_linear = nn.Linear(6,3,bias=True)
#我们也可以去打印它的参数
#print(module_linear.parameters())
#比如我们输入一个一维的6个的序列，然后输出就是一个一维的3个的序列
lista = t.ones(6,dtype=t.float)
listb = module_linear(lista)
#print(listb)
#我们去打印参数，但是参数的话，用这样的方式可以打印参数的具体的值，可以看到参数就是一个3行6列的一个数组
#print(module_linear._parameters)



# 2、torch.nn.Conv1d、Conv2d、Conv3d
# 他们的参数基本都是如下的参数
# in_channels(int) 输入信号的通道
# out_channels(int) 输出信号的通道
# kerner_size(int or tuple) - 卷积核的尺寸
# stride(int or tuple) 步长 默认为1
# padding(int or tuple) 即为padding 默认为0
# dilation(int or tuple) 详细描述在这里 卷积核元素之间的间距 默认为1
# groups(int or tuple) 从输入通道到输出通道的阻塞连接数 默认为1,控制输入和输出之间的连接，group=1，输出是所有的输入的卷积；
# group=2，此时相当于有并排的两个卷积层，每个卷积层计算输入通道的一半，并且产生的输出是输出通道的一半，随后将这两个输出连接起来
#bias(bool, optional) - 如果bias=True，添加偏置
#需要注意的是其中的dilation是个什么东西
#https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

#这里面就是有一个感受野的的东西；之前的话，做卷积的话，我们输入都只能是，一个卷积后的特征值，对应的是之前的3x3，或者5x5的像素块；
#这个像素块的像素是紧密相连的；但是如果dilation是2的话，那么这个就不再是紧密相连了，比如是2的话，就隔一个像素这种；


module_conv1d = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=1,stride=1,padding=0,dilation=1,groups=1,bias=False)
#为什么1维的卷积，我们需要输入是一个三维呢
#这三维的第一个维度，代表的是batch_size；第二个维度代表的是通道数，也可以理解为输入的特征数；
#我们可以想象，就是图像卷积的那个通道数
#自然第三维就是我们代表的是长度，也就是具体的长度
listc = module_conv1d(lista.view(1,1,-1))
#print(listc)

#同样，二维卷积，输入就是一个4维度 = batch_size x channels x width x height

module_conv2d = nn.Conv2d(1,1,1,1,0,1,1,bias=False)
listd = module_conv2d(lista.view(1,1,1,-1))
#print(listd)

#同样，三维卷积，输入就必须是一个5维度的

module_conv3d = nn.Conv3d(1,1,1,1,0,1,1,bias=False)
liste = module_conv3d(lista.view(1,1,1,1,-1))
#print(liste)


#3池化层
#无论是最大池化，还是平均池化，首先输入的维度，和前面的卷积是一样的；比如1d的话，就是输入3维，2d的输入4维，3d的话输入5维
MaxPool1d = nn.MaxPool1d(1,1)
MaxPool2d = nn.MaxPool2d(1,1)
MaxPool3d = nn.MaxPool3d(1,1)
AvgPool1d = nn.AvgPool1d(1,1)
AvgPool2d = nn.AvgPool2d(1,1)
AvgPool3d = nn.AvgPool3d(1,1)

#print(MaxPool1d(lista.view(1,1,-1)))
#print(MaxPool2d(lista.view(1,1,1,-1)))
#print(MaxPool3d(lista.view(1,1,1,1,-1)))

#print(AvgPool1d(lista.view(1,1,-1)))
#print(AvgPool2d(lista.view(1,1,1,-1)))
#print(AvgPool3d(lista.view(1,1,1,1,-1)))

#4、非线性
m = nn.ReLU()
m = nn.Sigmoid()
m = nn.Tanh()

#标准化
#torch.nn.BatchNorm1d
#torch.nn.BatchNorm2d
#torch.nn.BatchNorm3d
#这四个都是有一个参数就是特征数，也就是输入的特征数，在类实例化的时候，需要传入输入的特征数，也就理解为通道数
#在调用类方法的时候，和上面卷积池化是一致的
BatchNorm1d = nn.BatchNorm1d(1)
BatchNorm2d = nn.BatchNorm2d(1)
BatchNorm3d = nn.BatchNorm3d(1)
lista = t.arange(0,6,dtype=t.float)
# print(BatchNorm1d(lista.view(1,1,-1)))
# print(BatchNorm2d(lista.view(1,1,1,-1)))
# print(BatchNorm3d(lista.view(1,1,1,1,-1)))


#=======================================================================================================================================

#优化器
#首先要了解什么是优化器；所谓的优化器，就是指我们损失函数在反向传播的时候，是如何下降的，也是不是可以理解为是属于梯度下降算法的具体选择
#看一下torch的梯度下降算法都包含哪些
#说几个主要的优化器
#SGD  随机梯度下降法
#Adam 自适应学习率
#这两种的区别是，adam更加适合大数据量的，容易过拟合，收敛速度快；sgd更加适合小数据量的，计算开销小


import torch.optim as optim

#首先，初始化一个优化器
optimizer = optim.SGD(module_conv2d.parameters(), lr = 0.01)

#第二步，就是梯度清零，防止梯度的累加
optimizer.zero_grad() # 网络参数的导数变化为0

#第三步，前向计算
#output = net(input) # 网络前向计算

#第四步，进行损失计算
#loss = criterion(output, target) # 计算损失

#第五步，反向传播
#loss.backward() #　得到模型中参数对当前输入的梯度，使用损失的值后面加上.backward()反向传播

#第六步，更新参数
optimizer.step() # 保存更新的参数


#==========================================================================================================================

#损失函数
from torch.nn import CrossEntropyLoss
from torch.nn import L1Loss
from torch.nn import MSELoss
from torch.nn import NLLLoss
from torch.nn import BCELoss

#常用的集中损失函数，比如
#1、交叉熵损失函数 CrossEntropyLoss
# 交叉熵损失函数，说白了就是softmax；用作更多的是分类任务；输入的概率标签+预测的概率标签
#如真实概率分布p(xi)的向量为[1, 0, 0]，预测的概率分布q(xi)的向量为：[0.67, 0.24, 0.09]
#那么算出来的损失值其实就是0.4

#2、L1Loss