import os
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
#pip install torchvision
from torchvision import transforms, models, datasets
#https://pytorch.org/docs/stable/torchvision/index.html
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image

#这一小节，通过一个例子，把前面的全都串起来，重点看一下如何搭建一个网络


#第一步，首先我们需要组织一下数据结构，目录的结构一个是train，一个是valid；都是代表的是标注之后的数据
data_dir = 'flower_data/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

#第二步，开始做数据预处理，这里只是进行组织，还没有真正的进行预处理
#这里面需要了解一下比较主要的几点函数
#aaaaa、transforms.Compose;说白了就是连接，连接这些预处理操作
#2、Totensor，图片需要读到tensor格式
#3、Normalize，归一化，标准化，前面是均值，后面是标准差；也就是说，我们要按照这样的均值标准差，对图像进行归一化；归一化以后图片，他们就可以做到这样的均值和标准差

data_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(45),#随机旋转，-45到45度之间随机选
        transforms.CenterCrop(224),#从中心开始裁剪
        transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转 选择一个概率概率
        transforms.RandomVerticalFlip(p=0.5),#随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        transforms.RandomGrayscale(p=0.025),#概率转换成灰度率，3通道就是R=G=B
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差
    ]),
    'valid': transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#第三步 batch数据的制作
#batch数据的制作，绝对不是那么简单的就是一组图片，我们拼接一下说是一个batch就可以了；它实际上是一个batch数据，用torch的方式

batch_size = 8

#image_datasets是一个字典；其中imagefolder代表的是要对图片做预处理，形成一个新的数据集
#我们形成的就是数据集
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}

#对新的数据集，我们去做batch，shuffle代表是打乱数据集，也就是说dataloaders是为了做batch
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes


# 是否用GPU训练
# 第三个就是我们需要看设备问题，是否含有gpu的机器；如果有的话，我们选择cuda为0的设备，如果没有的话，我们选择使用cpu
# 另外的话，就是torch.device
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = 'resnet'  #可选的比较多 ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
#是否用人家训练好的特征来做
feature_extract = True

#这里说的，其实就是迁移学习；也就说说，我们可以选择冻住多少层的参数；这里可以看，我们是需要把model所有的参数都冻住；
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


#加载模型
def initialize_model(model_name, feature_extract, use_pretrained=True):
    # 选择合适的模型，不同模型的初始化方法稍微有点区别
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet152
        """
        #这里的模型，我们直接用的就是models里面的resnet，并且使用了预训练的参数
        model_ft = models.resnet152(pretrained=use_pretrained)
        #冻结所有层的参数
        set_parameter_requires_grad(model_ft, feature_extract)
        #我们专门看的是，在进行了一堆卷积，池化，然后最后进入到全连接层的特征数是多少
        num_ftrs = model_ft.fc.in_features
        #我们这里是需要修改他的特征数的；需要自己指定全连接层；

        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 102),
                                   nn.LogSoftmax(dim=1))
        input_size = 224
    #返回输入的大小，还有模型
    return model_ft, input_size


model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

#GPU计算
model_ft = model_ft.to(device)

# 模型保存
#这里我们可以学习一个非常好的知识点，主要就是关于python 序列化的一个东西，我们经常看到就是我们保存的模型文件后缀，有的时候是pth文件，有的时候是pt文件
#有的时候还是pkl文件
#其实pth文件和pt文件是一样的；我们可以认为是一致的；但是pkl不是的，这里面有一个叫做序列化的东西
#也就是说，我们比如一个字典，或者说一个结构，我们要序列化后，存到文件里面，不是字符串存，那样的话，就等于写文件了
#我们pkl说白了就是叫持久化存储，东西存进去了以后，拿出来还可以用

#可以举个例子，不过这个例子，和我们这一节讲的关系不大


import pickle
"""
#下面是序列化存到文件的内容
my_dict = {'name':'xuebi','age':24}
with open('text.txt','wb') as f:
    pickle.dump(my_dict,f)
"""


#下面是反序列化从文件当中读取的内容
"""
file_path = './text.txt'
with open(file_path,'rb') as f:
    my_dict=pickle.load(f)
    print(my_dict)
    print(type(my_dict))
"""
#需要保存最后的模型
filename='checkpoint.pth'

# 这里有一个逻辑，就是我们的params_to_update就是本身是代表了model的所有参数的；但是由于我们的迁移学习，我们是需要冻住一些层的；
# 此时，我们需要把可以做梯度计算的层参数拿出来，后面可以去更新；
# 如果不做迁移学习的话，那么params_to_updata就是代表的是所有的层

params_to_update = model_ft.parameters()
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)



# 优化器设置
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)#学习率每7个epoch衰减成原来的1/10
#最后一层已经LogSoftmax()了，所以不能nn.CrossEntropyLoss()来计算了，nn.CrossEntropyLoss()相当于logSoftmax()和nn.NLLLoss()整合
criterion = nn.NLLLoss()






#这里是比较重要的训练模块
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False,filename=filename):
    """
    :param model: 就是我们的训练模型
    :param dataloaders: 做好batch 的数据
    :param criterion: 损失函数
    :param optimizer: 训练器
    :param num_epochs: epoch个数
    :param is_inception:
    :param filename:
    :return:
    """
    since = time.time()

    #最好的一次准确率
    best_acc = 0
    #模型也得放到gpu中，如果没有gpu的话，那么就得放到cpu当中
    model.to(device)

    #记录两个准确率的历史，我们是为了，当所有的epoch学习完的时候，得到最好的那一次准确率
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    # 这里下面的两行代码，需要注释讲解一下
    # 首先，我们先看state_dict和parameters()的区别；我们前面不是就是可以获取每一层的参数嘛，为什么现在出现了一个state_dict呢；其实这个state_dict是一个字典，包含的不只是参数，或者说
    # 参数只是一个value，我们除了value还是有key的；
    # 而且，不仅仅是model有statedict，其实optimizer也是有statedict的；
    # model的statedict就是key是层级的名字，然后value就是具体的参数；optim的statedict，key可能就是学习率，动量什么的，值就是value
    # 我们前面说，存模型文件，存的是字典，肯定不能是只存一些参数，所以我们就是这里要保留的是statedict
    # 可以看一下就是正常的state_dict都是存的啥

    # Model.state_dict:
    # conv1.weight
    # torch.Size([6, 3, 5, 5])
    # conv1.bias
    # torch.Size([6])
    # conv2.weight
    # torch.Size([16, 6, 5, 5])
    # conv2.bias
    # torch.Size([16])
    # fc1.weight
    # torch.Size([120, 400])
    # fc1.bias
    # torch.Size([120])
    # fc2.weight
    # torch.Size([84, 120])
    # fc2.bias
    # torch.Size([84])
    # fc3.weight
    # torch.Size([10, 84])
    # fc3.bias
    # torch.Size([10])



    # Optimizer, s
    # state_dict:
    # state
    # {}
    # param_groups 	[{'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False
    #               , 'params': [367949288, 367949432, 376459056, 381121808, 381121952, 381122024, 381121880, 3811221
    #                          8, 381122096, 381122312]}]


    LRs = [optimizer.param_groups[0]['lr']]

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'valid']:
            #model.train()和model.eval()的作用
            #这里我们说train就是为了让它开启batchnormalaztion，以及dropout；eval的话，就是验证集不让它开启；因为eval不需要更新参数
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()   # 验证

            running_loss = 0.0
            running_corrects = 0

            # 把数据都取个遍
            # 当时我们的标签，好像没有标，那怎么获取到的呢？实际目录名字就当做了是类别名字
            #但是请注意，这个labels，并不是分类的名字，而是索引！！！！！！！！！！是分类的索引，一定要注意这个问题
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        #做一次前向传播
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:#resnet执行的是这里
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # 训练阶段更新权重
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 计算损失
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)


            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))


            # 得到最好那次的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                #可以自己去组织，形成要存入的state格式
                state = {
                  'state_dict': model.state_dict(),
                  'best_acc': best_acc,
                  'optimizer' : optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次当做模型最终的结果
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs



### 开始训练！

model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs  = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=20, is_inception=(model_name=="inception"))

### 再继续训练所有层

for param in model_ft.parameters():
    param.requires_grad = True

# 再继续训练所有的参数，学习率调小一点
optimizer = optim.Adam(params_to_update, lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 损失函数
criterion = nn.NLLLoss()


# Load the checkpoint
#我们可以看到，我们如果加在模型的话，进行继续训练的话，是非常关键的；我们需要好好梳理一下这一块的逻辑
#这里我们不是只是load文件，对于model和optim都需要进行load
#这里有个疑问，model_ft不是已经init了么。还需要load干啥
#load的是参数，而且这个参数和init的层数，层的设定是一一对应的
#可以看到，我们在load了以后，就意味着在这个基础之上，我们接着去训练，一开始存的是微调的参数，因为当时参数只跑了全连接层；
# 现在的话我们在这个基础之上，接着放开所有的参数，继续训练
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])


model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs  = train_model(model_ft, dataloaders, criterion, optimizer, num_epochs=10, is_inception=(model_name=="inception"))

#%% md

### 测试网络效果

#输入一张测试图像，看看网络的返回结果：

"""
probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
"""

#注意预处理方法需相同

### 加载训练好的模型

#这里我们在加在模型的时候，是非常关键的；这里我们加在了模型，但是要先把模型初始化好，然后再进行load参数

model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

# GPU模式
model_ft = model_ft.to(device)

# 保存文件的名字
filename='seriouscheckpoint.pth'

# 加载模型
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])

