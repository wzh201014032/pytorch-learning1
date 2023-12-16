import torch as torch
import torch.nn as nn
import numpy as np

#几种初始化
a = torch.rand(1,1,3)
b = torch.randn(1,3)
c = torch.empty(1,3)
d = torch.zeros(1,3)
e = torch.ones(1,3)
#从numpy导入
f = np.random.randn(1,2)
g = torch.from_numpy(f)

#初始化按自己的数据，进行tensor格式的初始化
#这样的话，必须是float类型，才能反向传播，是int貌似是不行的
h = [1,2,3]
I = torch.tensor(h,dtype=torch.float64)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#两种放到gpu的手段
#print(a.cuda())
print(b.to(device))

print(c)
print(d)
print(e)
print(g)
print(I)


#常见的几种torch的函数，这里我们都说的是基础的函数
#aaaaa、nn.conv2d
#必须是一个四维，才可以，batch_size*input_channels*width*height
print("*****************conv2d********************")

a = torch.ones(1,1,3,3)
print(a)
model = nn.Conv2d(in_channels=1,out_channels=2,stride=1,kernel_size=2,padding=0,bias=1)
b = model(a)
print(b)

#2、nn.avgpool2d nn maxpool2d
#维度至少是三维，或者四维才可以
#必须是float类型的元素，才可以进行池化
a = torch.tensor([[[[1,3],[2,4]]]],dtype=torch.float)
modelavg = nn.AvgPool2d(2,1,0)
modelmax = nn.MaxPool2d(2,1,0)

b = torch.tensor([[[1,3],[2,4]],[[5,6],[7,8]]],dtype=torch.float)
print("*****************modelpool********************")
print(modelavg(a))
print(modelmax(a))

print(modelavg(b))
print(modelmax(b))

print("*****************relu********************")
#3、Relu函数
#小于0的都变成了0，大于0的和原来的数值相等
a = torch.tensor([[[[100,-0.2],[0.12,41]]]],dtype=torch.float)
modelrelu = nn.ReLU()
print(modelrelu(a))

#4、参数归一化batchnormalization
#batch函数有一个参数就是特征图的个数，这个是需要设置的
#必须是四维， 且传入的参数是第三维的那个参数，输入的是通道数，可以看出这个通道数是2，所以输入的通道数作为batchnorm2d的参数
print("*****************batchNorm2d********************")
modelBatch = nn.BatchNorm2d(2)
a = torch.tensor([[[[100,-0.2],[0.12,41]],[[100,-0.2],[0.12,41]]]],dtype=torch.float)
modelrelu = nn.ReLU()
b = modelrelu(a)
print(b)
print(modelBatch(b))

#5、nn.Linear
#必须float类型的才可以
print("*****************nn.Linear********************")

a= torch.tensor([1,2,3,4,5],dtype=torch.float)
modelLinear = nn.Linear(5,3)
print(modelLinear(a))

#6、nn.Sequnetial
#把上面的进行合并

a = torch.tensor([[[[1,-2],[2,4]],[[1,-2],[2,4]]]],dtype=torch.float)
#第一种链接
model = nn.Sequential(modelavg,modelrelu)
print(model(a))


#第二种链接
b = [modelavg,modelrelu]
modelB = nn.Sequential(*b)
print(modelB(a))
print("*********F.conv2d*********")
#functional的卷积
#https://www.baidu.com/link?url=CyRAn8HWXqD1m0QlTNhxORfJrDSKHTKDX8nrFtbUXdxSm0wnu3bOWA5cOgsEubWfZwVN5nPPva-r_xp-cRXZqbGoLr0MMbcIGU98OIVvCcm&wd=&eqid=a5de14f7000aa8c20000000262cc4058
#a是输入，b是参数
#b的第三个通道xgroups参数，需要和a的第三个通道数相同
#这里用了分组卷积，说白了就是把batchxcxwxh中的c分成了两组，也就是c/2，然后分别去卷积
#同样输出通道数也是，分成了c/2，所以b的第一个参数必须是偶数才行
from torch.nn import functional as F

a  = torch.randn(1,4,3,3)
b  = torch.randn(6,2,2,2)
#print(b)
#F的和nn不一样，nn看成是类，返回的是model类型的类，但是F是函数，返回的是result
#参数是输入数据和卷积核，就是因为我们能确定参数，所以采用F的东西进行卷积
res= F.conv2d(a,b,padding=0,stride=1,groups=2)
print(res)



print("*********F.linear*********")
#2、F.linear()
#这是linear的问题，必须weight里面的第一个维度，是output，第二个维度是inut才可以
#相当于linear做了一个转置的操作
a = torch.tensor([1,2,3,4,5],dtype=torch.float)
b = torch.ones(5,3).reshape(3,5)
#print(b)
res = F.linear(a,b,bias=torch.ones(1,3).reshape(3))
print(a.shape)
print(b.shape)
print(a==a.reshape([1,5]))

def func1(para):
    print(para)
    def func2(func):
        print("middle")
        def func3(*args,**kwargs):
            func(*args,**kwargs)
        return  func3
    return func2

@func1("装饰器")
def func(para):
    print("first!")
    print(para)

func("aaa")

a = torch.ones(1,1,3,3)
print(a)
model = nn.Conv2d(in_channels=1,out_channels=2,stride=1,kernel_size=2,padding=0,bias=1)
b = model(a)
for param in model.parameters():
    print(param.requires_grad)


