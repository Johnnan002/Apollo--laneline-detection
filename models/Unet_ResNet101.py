#coding:utf-8
#@author: Jiangnan He
#@date:2019.12.10 18:21

''''''


'''
resnet101-unet
1.上采样  还是采用 转置卷积
2.进行高低层特征融合时使用了torch.cat() 替换了 FCN 中的 "+"
3.resnet101 下采样1/32 所以进行了 5次下采样 本实现进行了 3次高低层特征融合  最后得到heatmap为原图的1/4
4.与label做loss 时直接将输出 heatmap 上采样到image尺寸

  layer0         input                                                           
                  ↓                                                                 
  layer1     conv3x3  ch=64          (1/2)                                       
                  ↓                                                                 
  layer2     reslayer(maxpool) ch=256  (1/4)   ---------------------------   256+256---conv3 ch=256
                  ↓                                                                 ↑upconv2
  layer3     reslayer   ch=512       (1/8) ------------------------------ 512+512---conv3 ch=512
                  ↓                                                                 ↑upconv2
  layer4     reslayer   ch=1024     (1/16)---------------------------------- 1024+1024--conv3 ch=1024
                  ↓                                                                 ↑upconv2
  layer5     reslaeyr  ch=2048      (1/32)-----------------------------------------
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchsummary import summary
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self,in_ch,out_ch,ksize=3,stride=1,padding=1):
        super(Block,self).__init__()
        self.conv1=nn.Conv2d(in_ch,out_ch,kernel_size=ksize,stride=stride,padding=padding)
        self.bn=nn.BatchNorm2d(out_ch)
        self.relu=nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv1(x)))

def make_layers(in_channels, layer_list):
    layers = []
    for v in layer_list:
        layers += [Block(in_channels, v)]
        in_channels = v
    return nn.Sequential(*layers)

#层
class Layer(nn.Module):
    def __init__(self, in_channels, layer_list):
        super(Layer, self).__init__()
        self.layer = make_layers(in_channels, layer_list)
    def forward(self, x):
        out = self.layer(x)
        return out

#残差块
class ResBlock(nn.Module):
    def __init__(self,ch_list,downsample,Res):# ch_list=[in_ch,ch,out_ch]
        super(ResBlock,self).__init__()

        self.res=Res# 残差块 还是 瓶颈块的标志位
        self.ds = downsample  # 残差块时  是否下采样
        #第一个1x1 卷积
        self.firconv1x1=Block(ch_list[0],ch_list[1],1,1,0)         #第一个1x1卷积   不下采样
        self.firconv1x1d = Block(ch_list[0],ch_list[1], 1,2,0)   #第一个1x1卷积   下采样
        # 3x3卷积
        self.conv3x3=Block(ch_list[1],ch_list[1],3,1,1)
        #第二个1x1卷积
        self.secconv1x1=Block(ch_list[1],ch_list[2],1,1,0)
        #skip 卷积操作
        self.resconv1x1d=Block(ch_list[0],ch_list[2],1,2,0)        #skip下采样的1x1卷积
        self.resconv1x1=Block(ch_list[0],ch_list[2],1,1,0)         #skip下采样的1x1卷积

    def forward(self,x):
        if self.res==True:#此时为残差块
            if self.ds==True:#skip有卷积操作需要下采样
                residual=self.resconv1x1d(x)
                f1=self.firconv1x1d(x)
            else:#skip有卷积操作不需要下采样
                residual = self.resconv1x1(x)
                f1 = self.firconv1x1(x)

        else:# 瓶颈块sikp无卷积操作
            residual=x
            f1 = self.firconv1x1(x)
        f2=self.conv3x3(f1)
        f3=self.secconv1x1(f2)
        f3+=residual
        return f3

#Res层
class ResLayer(nn.Module):
    def __init__(self,ch_list1,ch_list2,downsample,numBotBlock):
        super(ResLayer,self).__init__()
        self.num = numBotBlock
        self.resb=ResBlock(ch_list1,downsample,True)
        self.botb=ResBlock(ch_list2,downsample,False)
        self.BoB=self.make_layers()
    def make_layers(self):
        layers=[]
        for i in range(self.num):
            layers+=[self.botb]
        return nn.Sequential(*layers)
    def forward(self,x):
        res=self.resb(x)
        return self.BoB(res)

#resnet101
class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101,self).__init__()
        self.conv0=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1)
        self.conv1=nn.Conv2d(64,64,kernel_size=3,stride=2,padding=1)    #1/2
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)                 #1/4
        self.layer1=ResLayer([64,64,256],[256,64,256],False,4)
        self.layer2=ResLayer([256,128,512],[512,128,512],True,3)       #1/8   ch= 512
        self.layer3=ResLayer([512,256,1024],[1024,256,1024],True,23)   #1/16   ch= 1024
        self.layer4=ResLayer([1024,512,2048],[2048,512,2048],True,6)   #1/32   ch=2048
        #权重初始化 conv 以及BN
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')#mode="fan_in" weight方差在前向传播中保持不变 mode="fan_out" weight后向传播方差不变
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        layer0=self.conv0(x)                    #1
        layer1=self.conv1(layer0)               #1/2
        layer2 =self.layer1(self.pool(layer1) ) #1/4
        layer3=self.layer2(layer2)                         #1/8
        layer4 = self.layer3(layer3)                       #1/16
        layer5 = self.layer4(layer4)                       #1/32
        #print(layer0.size(),layer1.size(),layer2.size(),layer3.size(),layer4.size(),layer5.size())
        return [layer2,layer3,layer4,layer5]



#上采样模块
class Upsample(nn.Module):

    def __init__(self,in_ch1,in_ch2,upsampleratio=2):#  layer42048 layer3 1024
        super(Upsample,self).__init__()
        self.conv1=Block(in_ch1,in_ch2)# layer4 2048-->1024
        self.conv2=Block(in_ch2,in_ch2)#layer3 1024-->1024
        self.conv3=Block(in_ch2*2,in_ch2) # cat layer3  2048--> 1024
        self.trans_conv=nn.ConvTranspose2d(in_ch2,in_ch2,upsampleratio,stride=upsampleratio)
    def forward(self,featrue1,featrue2):#layer4 ,layer3
        #
        return self.conv3( torch.cat([ self.trans_conv(self.conv1(featrue1)), self.conv2(featrue2) ],dim=1)   )

class UnetDecode(nn.Module):
    def __init__(self,n,in_ch,out_ch,upsratio,trans_conv=False):     # in_ch  VGG 后通道数  out_ch 为上采样之后的通道数
        super(UnetDecode,self).__init__()
        self.conv1=Layer(in_ch,[out_ch]*n)         #这里加n层Block
        self.trans_conv=nn.ConvTranspose2d(out_ch,out_ch,upsratio,stride=upsratio)#        upsratio 上采样倍数

    def forward(self,x):
        return self.trans_conv(self.conv1(x))

class Unet_resnet101(nn.Module):
    def __init__(self,n,in_ch,out_ch,upsratio):
        super(Unet_resnet101,self).__init__()
        self.encode=ResNet101()
        self.catup1=Upsample(2048,1024)
        self.catup2=Upsample(1024,512)
        self.catup3=Upsample(512,256)
        self.decode=UnetDecode(n,in_ch,out_ch,upsratio)
        self.classfier=nn.Conv2d(256,10,3,padding=1)

    def forward(self,x):
        size=x.shape[-2:]
        features=self.encode(x)
        [layer2,layer3,layer4,layer5]=features#
        #layer1 256*16   layer2 :512*8*8   layer3 :  1024*4*4   layer4 : 2048*2*2
        layer4=self.catup1(layer5,layer4)
        layer3=self.catup2(layer4,layer3)
        layer2=self.catup3(layer3,layer2)
        #return self.classfier(layer2)
        return F.interpolate(self.classfier(layer2), size=size, mode='bilinear', align_corners=False)

if __name__=="__main__":
    if  torch.cuda.is_available():
        model=Unet_resnet101(4,2048,64,4).cuda()
        x = Variable(torch.randn(1, 3, 64, 64)).cuda()
    else:
        model=Unet_resnet101(4,2048,64,4)
        x = Variable(torch.randn(1, 3, 64, 64))
    model.eval()
    y=model(x)
    print("y.size",y.size())
    summary(model,(3,64,64))

