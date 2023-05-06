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

e = copy.deepcopy(b[:,0])
#虽然e我们取的是一列，但是我们一列是一个一维的；我们不能认为是一个2维，所以e[0][0]是错误的
print(e[0])


