#对于tensor格式的学习，以及我们可以把tensor一些场景的操作，还有和numpy的常见操作

import torch as t
import numpy as np

#第一种初始化的方法，但是记得Tensor首字母大写
x = t.Tensor(5,3)
#如果是小写的话，就和np.array是同样的用法了，就是可以直接复制
x = t.tensor([1,2,3,4,5])
#print(x)

#第二种初始化方法，rand，或者randn
#randn是归一化的情况
x = t.rand(5,3)
y = t.randn(5,3)
#print(x,y)

#打印一下他们的大小，size和shape都是同样的意义
#print(x.size(),x.shape)

#矩阵的加法，有三种方法，1、可以直接加；2、可以t.add；3、可以x.add(y)
x = y =  t.ones(2,2)
# print(x+y)
# print(x.add(y))
# print(t.add(x,y))
#矩阵切片，比如我们要取最左边一列
x = t.Tensor(5,3)
#print(x)
#因为x是5行三列的东西，所以我们取的x所有的行，第一个代表的是行，然后是取第0列，也就是最左边的列
print(x[:,0])


#numpy和tensor格式的互相转换

a = t.randn(5,3)
b = np.random.random((5,3))
#直接就可以tensor转np
c = a.numpy()
#numpy转tensor
d = t.from_numpy(b)

#print(c)
#print(d)

#关于cuda
#这是判断是否cuda可以用的情况
print(t.cuda.is_available())
#如果cuda可以用的话，那么可以把tensor张量的值放到cuda里面，也就是可以进行gpu计算
#a = a.cuda()
#d = d.cuda()
#print(a+d)

#tensor初始化向量
#arange的话，是从0开始到5，因为要小于6；所以共有6个元素,注意不是range而是arange
A = t.arange(0,6)

a = A.view(2,3)
b = A.reshape(2,3)


#print(a,b)

#关于深拷贝和浅拷贝的问题的验证，我们假设我们通过切片的方式，修改是否会改变元素

a = t.tensor([1,2,3,4,5,6])
b = a.view(2,3)
b[0][0] = 7
#print(b)

import copy
d = copy.deepcopy(b[0][0])
#d虽然是一个数字，但是d也是一个tensor格式的；
print(d)
print(b)
print(a)

e = copy.deepcopy(b[:,0])
#虽然e我们取的是一列，但是我们一列是一个一维的；我们不能认为是一个2维，所以e[0][0]是错误的
print(e[0])


#矩阵的点乘
a = t.tensor([1,2,3,4,5,6])
b = t.tensor([2,2,2,2,2,2])
c = a*b
#就是对应的元素相乘
print(c)

h = b.view(-1,1)
print(h.shape)
#首先这个mm是一个二维的相乘；所以我们为什么a要进行view；是因为开始的时候，a是一个一维的；必须a是一个2维的才可以；
#另外的话，就是b这个我们view称为了一个6行一列的内容；
d = t.mm(a.view(1,-1),h)
print(d)


#bmm就是顾名思义，b代表的是batch；所以我们的b就是代表第三个围堵

e = t.bmm(a.view(1,1,-1),h.reshape(1,-1,1))
print(e)


#矩阵的除法
#如果元素都是float类型的话，那么可以是/来代表除法；如果是int类型，那么就是//,//除了以后，仍然还是整数

f = a//b
print(f)

#矩阵的切割和拼接也是非常的重要；主要是当我们遇到比如就是那种需要把整个图像进行分区间的情况，就是需要用到拼接或者切割的函数

#拼接函数cat
#矩阵拼接操作，将符合要求的不同矩阵在某一维度上进行拼接。
#cat要求进行拼接的矩阵在非拼接维度上完全相同。
import torch

a1 = torch.rand(4, 3, 32, 32)
a2 = torch.rand(5, 3, 32, 32)
a3 = torch.rand(4, 1, 32, 32)
a4 = torch.rand(4, 3, 16, 32)

# 要求其他维度必须相同
print("torch.cat([a1, a2], dim=0): ",
      torch.cat([a1, a2], dim=0).shape)
print("torch.cat([a1, a3], dim=1): ",
      torch.cat([a1, a3], dim=1).shape)
print("torch.cat([a1, a4], dim=2): ",
      torch.cat([a1, a4], dim=2).shape)

#拼接函数stack
#矩阵堆叠，将若干维度完全相同的句子在某一维度上进行堆叠。

#stack操作会在指定维度前增加一个维度，然后让后面的维度堆叠起来。

#要求堆叠的矩阵维度完全相同。

import torch

a1 = torch.rand(4, 3, 16, 32)
a2 = torch.rand(4, 3, 16, 32)
print("torch.cat([a1, a2], dim=2): ",
      torch.cat([a1, a2], dim=2).shape)

# stack会新插入一个维度，使其他维度保持不变
# stack要求两个矩阵维度完全一致
print("torch.stack([a1, a2], dim=2): ",
      torch.stack([a1, a2], dim=2).shape)

a = torch.rand(32, 8)
b = torch.rand(32, 8)
print("torch.stack([a, b], dim=0): ",
      torch.stack([a, b], dim=0).shape)
# 可用于将两个相同的表整合起来，且会保持两张表的独立性

#切割函数split
#split可对矩阵按长度进行分割。

#split有两种分割模式：

#输入列表：会根据列表中的值进行分割，要求列表值之和等于被分割维度元素个数。
#输入数值：会根据数值进行分割，会分割出若干个等长的矩阵，当剩余长度小于指定长度时，也会将加入返回列表。要求输入数值小于等于被分割维度元素个数。
import torch

a = torch.rand(32, 8)
c = torch.stack([a]*6, dim=0)

print("c.shape: ", c.shape)

def showSplits(ms):
    for index, m in enumerate(ms):
        print("第{}个tensor的shape为 {}".format(
            index + 1, m.shape
        ))
    print()

# 自定义拆分，列表中每个数表示拆分后目标维度的长度
# 要求列表元素和等于目标维度拆分前的长度
# 本来c的维度是6，32，8；现在从维度0拆分，这个列表代表份数；1个1份，1个2份，一个1份。。。
ms = c.split([1, 2, 1, 1, 1], dim=0)
print("c.split([1, 2, 1, 1, 1], dim=0):")
showSplits(ms)

# 按长度拆分，输入的值表示每个拆分后的tensor的目标维度的长度
# 最后一个tensor如果长度不够也会返回

# 这里突出的是分割后的每个tensor的维度是4，突出的是每个；
ms = c.split(4, dim=0)
print("c.split(4, dim=0):")
showSplits(ms)

ms = c.split(8, dim=1)
print("c.split(8, dim=1):")
showSplits(ms)

ms = c.split(32, dim=1)
print("c.split(32, dim=1):")
showSplits(ms)

#切割函数 chunk
# chunck可按照数量对矩阵进行分割，将矩阵分割为指定个数。
#
# 若矩阵可被均匀分割，则返回指定个数的相同维度矩阵。
#
# 若不可被均匀分割，则返回的最后一个矩阵维度会较低。

import torch

a = torch.rand(32, 8)
c = torch.stack([a, a, a, a, a, a], dim=0)

print("c.shape: ", c.shape)

def showSplits(ms):
    for index, m in enumerate(ms):
        print("第{}个tensor的shape为 {}".format(
            index + 1, m.shape
        ))
    print()

# 按个数拆分，输入拆分的个数和维度
ms = c.chunk(2, dim=0)
print("c.chunk(2, dim=0):")
showSplits(ms)

ms = c.chunk(3, dim=0)
print("c.chunk(3, dim=0):")
showSplits(ms)

ms = c.chunk(5, dim=1)
print("c.chunk(5, dim=0):")
showSplits(ms)

