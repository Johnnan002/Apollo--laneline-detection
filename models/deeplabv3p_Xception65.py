#coding:utf-8
#@author: Jiangnan He
#@date:2019.12.17 15:03
''
'''
本实现为deeplabv3+，backbone为Xception_65 
'''

import torch.nn as nn
import torch
from torchsummary import summary
from torch.nn import functional as F
#实现深度分离卷积 将卷积分程in_ch 组进行卷积 接着用1x1 卷积进行融合
class depthwiseconv(nn.Module):
    def __init__(self, in_ch, out_ch,ksize=3,stride=1,padding=1):
        super(depthwiseconv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=ksize,
            stride=stride,
            padding=padding,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
    def forward(self, x):
        return self.point_conv(self.depth_conv(x))

#块
class Block(nn.Module):
    def __init__(self,in_ch,out_ch,ksize=3,stride=1,padding=1,sep=True):
        super(Block,self).__init__()
        #是否为深度可分离卷积模块标志位
        self.issepconv = sep
        self.conv=nn.Conv2d(in_ch,out_ch,kernel_size=ksize,stride=stride,padding=padding,bias=False)
        self.sepconv = depthwiseconv(in_ch, out_ch, ksize, stride, padding)
        self.Bn=nn.BatchNorm2d(out_ch)
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        if self.issepconv==True:
            return self.relu(self.Bn(self.sepconv(x)))
        else:
            return self.relu(self.Bn(self.conv(x)))


#层
class Layer(nn.Module):
    def __init__(self,in_ch,out_ch,ksize,stride,padding=1,upsample=True):# 64 128 3 1 1
        super(Layer,self).__init__()
        self.upsample=upsample#是否进行下采样
        self.block1=Block(in_ch,out_ch,ksize,stride,padding)
        self.block2 = Block(out_ch, out_ch, ksize, stride, padding)
        self.block3 = Block(out_ch, out_ch, ksize, stride, padding)#不进行下采样时ksize=3 padding=1 same
        #下采样时
        self.block4=Block(out_ch, out_ch, ksize, stride=2, padding=1)#进行下采样时
        self.skipconv=nn.Conv2d(in_ch,out_ch,kernel_size=1,stride=2,padding=0)#padding设置注意
        self.skipconv1=nn.Conv2d(in_ch,out_ch,kernel_size=1,stride=1,padding=0)#padding设置注意
        if self.upsample==True:#进行下采样时
            layer = [self.block1, self.block2, self.block4]
        else:#不进行下采样时的组合
            layer=[self.block1,self.block2,self.block3]
        self.layer = self.layer=nn.Sequential(*layer)

    def forward(self,x):
        if self.upsample==True:
            out=self.layer(x)
            skipout=self.skipconv(x)
            return torch.cat([out,skipout],dim=1)
        else:
            out=self.layer(x)
            skipout=self.skipconv1(x)
            return torch.cat([out,skipout],dim=1)



#模型 Xception 作为backbone
class DCNN(nn.Module):
    def __init__(self):
        super(DCNN,self).__init__()
        self.conv1=Block(3,32,3,2,1,False)#conv  1/2
        self.conv2=Block(32,64,3,1,1,False)#conv
        self.layer1=Layer(64,128,3,1,1,True)#128  1/4
        self.layer2=Layer(256,256,3,1,1,True) #512 /8
        self.layer3 =Layer( 512,728, 3, 1, 1, True)# 1456 /16
        self.BlockLayer=Layer(1456,728,3,1,1,False)#repeat 16times
        self.layer4 = self.make_layer(16)
        self.layer5=Layer(1456,1024,3,1,1,True) #1/32
        self.conv3=Block(2048,1536,3,1,1,True)#sepconv
        self.conv4 = Block(1536, 1536, 3, 1, 1, True)  # sepconv
        self.conv5=Block(1536, 2048, 3, 1, 1, True)  # sepconv

    def make_layer(self,numlayer):
        layer=[]
        for i in range(numlayer):
           layer+=[self.BlockLayer]
        return nn.Sequential(*layer)

    def forward(self, x):
        #entry flow
        conv1=self.conv1(x)
        conv2=self.conv2(conv1)
        layer1=self.layer1(conv2)#
        layer2=self.layer2(layer1)
        layer3 = self.layer3(layer2)
        #middlle flow
        layer4 = self.layer4(layer3)
        #exit flow
        layer5 = self.layer5(layer4)
        conv3=self.conv3(layer5)
        conv4=self.conv4(conv3)#
        conv5=self.conv5(conv4)
       # print("DCNN ",conv1.size(), conv2.size(),layer1.size(),layer2.size(),layer3.size() ,layer4.size(), layer5.size(), conv3.size(),conv4.size(),conv5.size())
        return layer2,conv5

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rate1, rate2, rate3 = (6,12,18)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class deeplabv3p(nn.Module):
    def __init__(self):
        super(deeplabv3p,self).__init__()
        self.DCNN=DCNN()
        self.ASPP=ASPP(2048)
        self.conv1=Block(512,256,ksize=1,padding=0,sep=False)
        self.conv2=Block(2*256,8,ksize=3,padding=1,sep=False)# 原本为 3x3  这里测试模型 输入很小 所以改为1x1
        self.conv3 = Block(256, 256, ksize=1, padding=0, sep=False)


        #初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')#mode="fan_in" weight方差在前向传播中保持不变 mode="fan_out" weight后向传播方差不变
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        lowfeature,out=self.DCNN(x)
        lowfeature=self.conv1(lowfeature)#
        #print(lowfeature.size(),'lowfeaturesize1')
        f1=self.conv3(self.ASPP(out))
        #print(f1.size(),'f1')
        hightfeature=F.interpolate(f1,scale_factor=4, mode='bilinear', align_corners=False)
        #print(hightfeature.size(),"hightfeatsize")
        out1=torch.cat([lowfeature,hightfeature],dim=1)
        return F.interpolate(self.conv2(out1),scale_factor=8, mode='bilinear', align_corners=False)

if __name__=="__main__":
    if torch.cuda.is_available():
        x=torch.randn(1,3,128, 384).cuda()
        model=deeplabv3p().cuda()
    else:
        x=torch.randn(1,3,128, 384)
        model=deeplabv3p()
    model.eval()
    y=model(x)
    print(y.size(),"y.size")



    '''
    summary(model, (3, 128, 384))
    '''







