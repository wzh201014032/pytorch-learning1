import torch as t
import torch
import numpy as np


#第一需要注意的是，我们如何设定参数是否是可导的

input = t.randn(5,3)
para1 = t.ones(5,3)
#第一种设置可以求导的情况，就是先赋值后，然后指定requires_grad是否可以求导
para1.requires_grad  = True
#第二种可以求导的方式，就是在赋值的过程当中，作为参数来决定
parab = t.ones(5,3,requires_grad=True)

out = input*para1+parab
#注意，求导的必须是一个标量，不能是一个张量；我们只需要想，标量就是求和，对每个参数求导，实际上因为是加法，就很容易求导
loss = out.sum()
#如果我们不设置retain_graph的话，那么就只能做一次的backward；
loss.backward(retain_graph=True)
print(para1.grad)
print(parab.grad)

#此时，因为我们没有清零；所以我们现在只是出现了
loss.backward(retain_graph=True)
print(para1.grad)
print(parab.grad)


#我们对梯度进行清零，就是不让它累加
para1.grad.zero_()
parab.grad.zero_()

loss.backward(retain_graph=True)
print(para1.grad)
print(parab.grad)

#如果使用了优化器的话，则可以直接用model.zero_grad()来进行参数消除