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
        self.conv2 = nn.Conv2d(planes, planes, stride=1, kernel_size=3,groups=1, dilation=1)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes*self.expansion,planes*self.expansion,stride=1,kernel_size=1,padding=0,dilation=1,groups=1)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = nn.Sequential(nn.Conv2d(planes,planes*self.expansion,stride=1,kernel_size=1,padding=0,dilation=1,groups=1),norm_layer(planes*self.expansion))

    def forward(self,x):
        indentify = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(x)
        out = self.bn3(out)

        #在relu前就要先进行残差
        if self.isidentify == True:
            indentify =self.downsample(indentify)
            out+=indentify
        out = self.relu(out)


class resnet152(nn.Module):
    def __init__(self):
        super(resnet152,self).__init__()

    def _make_layer(self,inplanes,inum):

