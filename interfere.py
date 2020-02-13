#coding=utf-8
#@author Jiangnan He
#@date:  2020.01.11 20:10

import os
import cv2
import torch
import numpy as np
from models.deeplabv3p_Xception65 import deeplabv3p
from models.Unet_ResNet101 import Unet_resnet101
from utils.img_process import crop_resize_data
from utils.lab_process import decode_color_labels
from train import network




if torch.cuda.is_available():
    device_id=0
predict_net = 'deeplabv3p'
nets = {'deeplabv3p': deeplabv3p, 'unet': Unet_resnet101}

def load_model():
    if torch.cuda.is_available():

        net=network(predict_net).getnet().cuda(device_id)
    else:
        net=network(predict_net).getnet()
    net.eval()
    # 加载网络参数
    net.load_state_dict(torch.load(os.path.join(os.getcwd(),'checkpoint.pt')))
    return net

def img_transform(img):
    img = crop_resize_data(img)
    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis, ...].astype(np.float32)
    img = torch.from_numpy(img.copy())
    if torch.cuda.is_available():
        img = img.cuda(device=device_id)
    return img

def get_color_mask(pred):
    pred = torch.softmax(pred, dim=1)
    pred_heatmap = torch.max(pred, dim=1)
    # 1,H,W,C
    pred = torch.argmax(pred, dim=1)
    pred = torch.squeeze(pred)
    pred = pred.detach().cpu().numpy()
    pred = decode_color_labels(pred)
    pred = np.transpose(pred, (1, 2, 0))
    return pred

def main():
    test_dir = 'test_example'
    net = load_model()
    img_path = os.path.join(test_dir, 'test.jpg')
    img = cv2.imread(img_path)
    img = img_transform(img)
    pred = net(img)
    color_mask = get_color_mask(pred)
    cv2.imwrite(os.path.join(test_dir, 'color_mask.jpg'), color_mask)

if __name__ == '__main__':
    main()
