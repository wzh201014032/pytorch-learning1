import torch as t
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from PIL import Image
path='flower_data/test/3/image_06612.jpg'


from torchvision import transforms, models, datasets

import resnet_mineclass as mineresnet
##第一步，首先我们需要组织一下数据结构，目录的结构一个是train，一个是valid；都是代表的是标注之后的数据
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
#可以认为dataloader是吧image_datesets处理好的数据，加载到了dataloader里面；dataloader里面的dateset就是所有的图片
#他的其中一个属性是batch_size=8

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes


train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model_name = 'resnet'  #可选的比较多 ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
#是否用人家训练好的特征来做
feature_extract = True

#这里说的，其实就是迁移学习；也就说说，我们可以选择冻住多少层的参数；这里可以看，我们是需要把model所有的参数都冻住；
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


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

#model_ft, input_size = initialize_model(model_name, feature_extract, use_pretrained=True)


#用我们自己的resnet152
model_ft = mineresnet.resnet152()

#GPU计算
model_ft = model_ft.to(device)


params_to_update = model_ft.parameters()

# 优化器设置
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)#学习率每7个epoch衰减成原来的1/10
#最后一层已经LogSoftmax()了，所以不能nn.CrossEntropyLoss()来计算了，nn.CrossEntropyLoss()相当于logSoftmax()和nn.NLLLoss()整合
criterion = nn.NLLLoss()

filename='checkpoint.pth'
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False,filename=filename):
    for epoch in range(100):
        for phase in ['train', 'valid']:
            #model.train()和model.eval()的作用
            #这里我们说train就是为了让它开启batchnormalaztion，以及dropout；eval的话，就是验证集不让它开启；因为eval不需要更新参数
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()   # 验证
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                #这里的inputs和labels都是一个8维的数组，因为是一个batch，这个for循环会遍历完所有的batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                #print(labels) #tensor([1, 1, 2, 0, 2, 2, 2, 1])
                #梯度清0
                optimizer.zero_grad()
                #前向传播
                outputs = model(inputs)
                #计算损失
                loss = criterion(outputs, labels)
                preddata, preds = torch.max(outputs, 1)
                #print(preddata,preds)
                #反向传播
                if phase == 'train':
                    loss.backward()
                    #更新参数
                    optimizer.step()
                # 计算损失
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                print(running_loss,running_corrects)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print("the epoch_loss is %s and the epoch_acc is %s"%(epoch_loss,epoch_acc))
            print("+++++++++++++++++++++++++++")
    return model


model_ft = train_model(model_ft,dataloaders,criterion,optimizer_ft)


img=Image.open(path)

img = transforms.CenterCrop(224)(img)
img = transforms.ToTensor()(img)
img = torch.unsqueeze(img, 0)
res = model_ft(img)
preddata, preds = torch.max(res, 0)
print(preddata,preds)

