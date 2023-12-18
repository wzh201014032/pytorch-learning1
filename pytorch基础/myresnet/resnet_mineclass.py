import torch as torch
import torchvision
import torch.nn as nn

class bottleblock(nn.Module):
    expansion = 4
    base_width = 64
    def __init__(self,inplanes,planes,isindentify):
        self.isidentify = isindentify
        super(bottleblock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes,planes,stride=1,kernel_size=1,padding=0,dilation=1,groups=1)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, stride=1, kernel_size=3,groups=1, dilation=1,padding=1)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes,planes*self.expansion,stride=1,kernel_size=1,padding=0,dilation=1,groups=1)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = nn.Sequential(nn.Conv2d(planes,planes*self.expansion,stride=1,kernel_size=1,padding=0,dilation=1,groups=1),norm_layer(planes*self.expansion))

    def forward(self,x):

        out = self.conv1(x)
        indentify = out
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        #在relu前就要先进行残差
        if self.isidentify == True:
            indentify =self.downsample(indentify)
            out+=indentify
        out = self.relu(out)
        return out


class resnet152(nn.Module):
    def __init__(self):
        super(resnet152,self).__init__()
        norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(256, 128, 4)
        self.layer3 = self._make_layer(512, 256, 6)
        self.layer4 = self._make_layer(1024, 512, 3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(2048, 3),nn.LogSoftmax(dim=1))

    def _make_layer(self,inplanes,planes,inum):
        layerlist = []
        layerlist.append(bottleblock(inplanes,planes,True))
        inplanes = bottleblock.expansion*planes
        for i in range(1,inum):
            layerlist.append(bottleblock(inplanes,planes,True))
        return nn.Sequential(*layerlist)


    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        #第一维是一个batch，所以我们要从第二维开始拉成1维
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x