#coding=utf-8
#@author Jiangnan He
#@date:  2019.12.29 20:10

'''
实现用来训练模型
本实现有3个模型 deeplabv3+  heatmap为原图的1/2   unet-resnet101  heatmap为原图的1/4   解决方案通过双线性插值到原图大小再和label 做loss
训练模型  1. metrics,
         2.loss,
         3.选择选了网络,
         4.lr,  save model, 多卡训练 , earlystop
'''

import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.deeplabv3p_Xception65 import deeplabv3p
from models.Unet_ResNet101 import Unet_resnet101
from utils.create_dataset import DatasetFromCSV
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from time import time
from utils.img_process import Process_dataset
from utils.img_process import ToTensor
from tqdm import tqdm


#==========================================
#loss:  bce, dice,  focal_loss , lovasz_loss,
#==========================================

#bce_loss
class MySoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, nbclasses):
        super(MySoftmaxCrossEntropyLoss, self).__init__()
        self.nbclasses = nbclasses

    def forward(self, inputs, target):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, self.nbclasses)  # N,H*W,C => N*H*W,C
        target = target.view(-1)
        return nn.CrossEntropyLoss(reduction="mean")(inputs, target)


#dice_loss
class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        num = 2*torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth
        loss = 1 - num / den
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss
        return total_loss/target.shape[1]


#focal_loss =  -(1-pt)^gamma*log(pt)
class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, class_weight=None, type='softmax'):
        target = target.view(-1, 1).long()
        if type=='sigmoid':
            if class_weight is None:
                class_weight = [1]*2 #[0.5, 0.5]
            prob   = F.sigmoid(logit)
            prob   = prob.view(-1, 1)
            prob   = torch.cat((1-prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)
        elif  type=='softmax':
            B,C,H,W = logit.size()          #one-hot编码
            if class_weight is None:
                class_weight =[1]*C #[1/C]*C
            logit   = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob    = F.softmax(logit,1)
            select  = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)
        class_weight = torch.FloatTensor(class_weight).cuda().view(-1,1)
        class_weight = torch.gather(class_weight, 0, target)
        prob = (prob*select).sum(1).view(-1,1)
        prob = torch.clamp(prob,1e-8,1-1e-8)
        batch_loss = - class_weight *(torch.pow((1-prob), self.gamma))*prob.log()
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss
        return loss




def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    return result


#==========================================================
#metric
#==========================================================

def compute_iou(pred, gt, result):
    """
    pred : [N, H, W]
    gt: [N, H, W]
    """
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()
    for i in range(8):
        single_gt=gt==i
        single_pred=pred==i
        temp_tp = np.sum(single_gt * single_pred)
        temp_ta = np.sum(single_pred) + np.sum(single_gt) - temp_tp
        result["TP"][i] += temp_tp
        result["TA"][i] += temp_ta
    return result                                               #计算iou


#EarlyStopping
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:#best_score 为空 保存一次模型
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
            if self.counter >= self.patience:#计数大于忍耐次数，早停
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        ''' Saves model when validation loss decrease.  '''
        if self.verbose:
            print('Validation loss decreased ({} --> {}).  Saving model ...'.format(self.val_loss_min,val_loss))
        torch.save(model.state_dict(), os.path.join(os.getcwd(),'checkpoint.pt'))#保存模型到磁盘 path='checkpoint.pt'
        self.val_loss_min = val_loss

#Network
class network():
      def __init__(self,net):
         self.net=net
         self.model=None
      def getnet(self):
         if self.net=="deeplabv3p":
            self.model= deeplabv3p()
         elif self.net  =="Unet-ResNet101":
            self.model=Unet_resnet101(4,2048,64,1)
         return self.model


#train lr save_model
#cycle lr  cosine lr

def train():
    #==============================================================
    #  check if gpu is available
    #==============================================================
    train_on_gpu=torch.cuda.is_available()
    if train_on_gpu:
       print("cuda is available,train on gpu")
       device_list = [0, 6]  #
       print(device_list)
    else:
       device_list="cpu"
       print("cuda is not available,train on cpu")
    #===============================================================
    # Config
    #===============================================================
    trained_network_name="deeplabv3p"   #  "deeplabv3p",    "Unet-ResNet101"
    batch_size=2
    print("batch_size is:"+str(batch_size))
    epoch=10
    print("epoch is:"+str(epoch))
    #trained_network_name="deeplabv3p"
    lr = 3e-4# Adam  最佳初始化3e-4或者5e-4
    numclasses = 8
    #===============================================================
    #data
    #===============================================================

    train_data_list="./data_list/train.csv"
    val_data_list="./data_list/val.csv"
    transform = transforms.Compose([Process_dataset(), ToTensor()])
    train_data=DatasetFromCSV(train_data_list,transform)
    transform = transforms.Compose([ ToTensor()])
    val_data=DatasetFromCSV(val_data_list,transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    #=================================================================
    # Adam
    #=================================================================
    model = network(trained_network_name).getnet().cuda(device=device_list[0])
    # 并行处理
    #model = torch.nn.DataParallel(model, device_ids=device_list)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lam = 0.5
    MAX_STEP = int(1e10)
    #cycle lr 设置  学习率的设置  周期衰减 或者周期变化挑出局部最优cosine lr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, MAX_STEP, eta_min=1e-5)#1e-6推荐

    #Warming up
    #=================================================================
    #Train loop
    #=================================================================
    early_stopping = EarlyStopping(patience=20, verbose=True)#
    since = time()
    for e in range(epoch):
        print(e)
        train_loss=0.0
        val_loss=0.0
        lr=scheduler.get_lr()
        print('epoch:{},lr:{}'.format(e,lr))
        #===============================================================
        #train
        #===============================================================
        model.train()
        for x,y in tqdm(train_loader):#训练数据
            y = torch.unsqueeze(y, 1).long()
            y_=make_one_hot(y,numclasses).cuda(device=device_list[0])
            x=x.cuda(device=device_list[0])
            y=y.cuda(device=device_list[0])
            opt.zero_grad()#梯度置为0
            y_pre=model(x)
            DLoss=DiceLoss()
            BLoss=MySoftmaxCrossEntropyLoss(numclasses)
            loss=lam*DLoss(y_pre,y_)+(1-lam)*BLoss(y_pre,y)#计算loss
            #loss=BLoss(y_pre,y)
            train_loss+=loss.item()
            loss.backward()#求导
            opt.step()#更新
            scheduler.step(e)
        #========================================================================
        #  validata
        #========================================================================
        model.eval()# changes the forward() behaviour of the module it is called upon. eg, it disables dropout and has batch norm use the entire population statistics
        torch.no_grad() # disables tracking of gradients in autograd.
        result = {"TP": {i: 0 for i in range(8)}, "TA": {i: 0 for i in range(8)}}
        for x1, y1 in tqdm(val_loader):
            y1_ = make_one_hot(y1, numclasses).cuda(device=device_list[0])
            y1 = torch.unsqueeze(y1, 1).long()
            x1, y1 = x1.cuda(device=device_list[0]), y1.cuda(device=device_list[0])
            y_pred1 = model(x1)
            #计算 dice_loss ce_loss
            DLoss1=DiceLoss()
            BLoss1 = MySoftmaxCrossEntropyLoss(numclasses)
            loss1=lam*DLoss1(y_pred1,y1_)+(1-lam)*BLoss1(y_pred1,y1)#计算loss
            val_loss+=loss1.item()

            #计算metric
            pred = torch.argmax(F.softmax(y_pred1, dim=1), dim=1)
            result = compute_iou(pred, y1, result)  # 计算iou
            print("Epoch:{}, test loss is {:.4f} \n".format(epoch, val_loss / len(val_loader)))
            for i in range(8):
                result_string = "{}: {:.4f} \n".format(i, result["TP"][i] / result["TA"][i])
                print(result_string)
        #===========================================================
        # Print training  information
        #===========================================================

        print_msg = ('[{epoch:>{}}/{n_epochs:>{}}] ' .format(e,epoch)+
                     'train_loss: {} '.format(train_loss) +
                     'valid_loss: {}'.format(val_loss))
        print(print_msg)

        #==========================================================
        # Early Stopping
        #==========================================================

        early_stopping(val_loss, model)  # 相当于 earlystopping.__call__(valid_loss, model)
        if early_stopping.early_stop:
                print("Early stopping")
                break

    #===============================================
    # load the last checkpoint with the best model
    #===============================================

    model.load_state_dict(torch.load('checkpoint.pt'))
    time_elapsed = time()- since
    print('training_time:{}'.format(time_elapsed/60))

if __name__=="__main__":
    train()




