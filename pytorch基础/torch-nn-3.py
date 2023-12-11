#第一个问题 enumerate 的作用
import torch.nn as nn
lista = ["aa","bb","cc"]

#这里的0代表的是指定的序列号
#也就是说enumerate的作用，就是可以打印出序列号，还有具体的值

para = enumerate(lista,0)
for index,paratmp in para:
    print("%s:%s"%(index,paratmp))

#第二个问题，我们就是关于交叉熵损失函数，logsoftmax 和nlllos的关系

#交叉熵损失函数，其实就是  nn.CrossEntropyLoss,他是包含的是logsoftmax + nllloss的

"""
nllloss就是负对数损失函数，而logsoftmax的作用就是，把值进行归一化，归一化到所有的值的和为0的一个过程；

什么意思呢？就是说，我们输入一张图片，经过分类过程以后，就会得到一堆概率的值；这堆概率代表的就是，当前这张图片，是每一个分类的概率；

然后呢？这些概率现在加在一起不是1；我们经过logsoftmax以后，概率和就是1了

然后的话，再去算损失；

要注意的是，我们输入有两个值，一个是output，一个是label，这个label是分类的索引，这个output是概率值；一定要注意，这两个不是同一类，不是都是概率或者都是标签的索引；

这里说的索引，举例比如有5个分类，猫狗鸡鸭鹅；那么索引就是0，aaaaa，2，3，4
"""

#require_grad的使用；

#比如我们想做一个很简单的回归模型，他的就是一个线性的回归，y = 1.6x+3

#定义输入
import torch as t
import torch.optim as optim
import numpy as np

#这里必须要做reshape的操作，否则这些所有的数都是同一个样本；
#我们想想的话，就是全连接都是竖着的多行，就只到要做这个reshape
inputs = t.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],dtype=t.float).reshape(-1,1)
outputs = t.tensor([4.6,6.2,7.8,9.4,11,12.6,14.2,15.8,17.4,19,20.6,22.2,23.8,25.4,27,28.6,30.2,31.8,33.4,35],dtype=t.float).reshape(-1,1)

models= t.nn.Linear(1,1,bias=True)

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

inputs = inputs.to(device)
outputs = outputs.to(device)
models = models.to(device)

# 优化器设置
optimizer_ft = optim.Adam(models.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)#学习率每7个epoch衰减成原来的1/10
criterion = nn.MSELoss()

epoch = 500
batch_size = 8


#如果改成Flase也还是无法求导；还是那句话，就是set_grad_enabled是无法进行求导的；
for paramtmp in models.parameters():
    paramtmp.requires_grad = True
for i in range(0,epoch):
    start = 0
    batch_loss = []
    #set_grad_enabled并不能改变reqieres_grad的值，但是设定为false的时候，反向传播就无法更新梯度；
    #如果设成true，就可以反向传播，但是哪些requires_grad为false的，依然是false，不会改变

    with t.set_grad_enabled(True):
        while (start+batch_size)<len(inputs):

            optimizer_ft.zero_grad()

            xx = t.tensor(inputs[start:start+batch_size], dtype=t.float, requires_grad=False)
            yy = t.tensor(outputs[start:start+batch_size], dtype=t.float, requires_grad=False)
            print(xx.requires_grad)
            print(yy.requires_grad)
            prediction = models(xx)
            loss = criterion(prediction, yy)
            optimizer_ft.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_ft.step()
            batch_loss.append(loss.data.numpy())
            start+=1
    print(i, np.mean(batch_loss))

test = t.tensor([88.])
print("==========")
print(models(test))

