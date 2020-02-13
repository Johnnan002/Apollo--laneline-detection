#coding: utf-8
#@author jiangnan He
#@date: 15:00 2019.11.13
'''
APollo laneline dataset
'''

###############################  config ##########################

#原图尺寸3384x1710 wh
#这里采用三种尺寸进行训练 768x256,1024x384,1536x512（wh）
imgH=128
imgW=384
offset=690
###############################################################

import numpy as np
import cv2
import torch

def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)

def rotate(xb,yb, angle):
    M_rotate = cv2.getRotationMatrix2D((imgH/2, imgW/2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (imgH , imgW ))
    yb = cv2.warpAffine(yb, M_rotate, (imgH , imgW ))
    return xb, yb

def blur(img):
    img = cv2.blur(img, (3, 3))
    return img

def add_noise(img):
    for i in range(200):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img

def data_augment(xb, yb):

    if np.random.random() < 0.25:#只对原图操作
        xb = random_gamma_transform(xb, 1.0)
    if np.random.random() < 0.25:#只对原图操作
        xb = blur(xb)

    if np.random.random() < 0.2:#只对原图操作
        xb = add_noise(xb)
    return xb, yb

class Process_dataset(object):
    def __call__(self,sample):
        image, mask=sample
        src_img=image[offset:,:]#hw
        gt_img =mask[offset:,:]  #先将 原图上部分的填空部分 3384x690 剪裁掉
        #1 图像尺度处理 将图像缩放到 训练尺寸  这里采用三种尺寸进行训练 768x256,1024x384,1536x512（wh）
        src_img = cv2.resize(src_img,
                               dsize=(imgW, imgH),
                               interpolation=cv2.INTER_LINEAR)
        gt_img = cv2.resize(gt_img,
                                      dsize=(imgW, imgH),
                                      interpolation=cv2.INTER_NEAREST)#  这里用最近邻处理
        #样本增强
        src_img,gt_img=data_augment(src_img,gt_img)
        return src_img,gt_img

class ToTensor(object):
    def __call__(self,sample ):
        image, mask=sample
        image = np.transpose(image,(2,0,1))
        image = image.astype(np.float32)
        mask = mask.astype(np.uint8)
        return torch.from_numpy(image.copy()),  torch.from_numpy(mask.copy())



# crop the image to discard useless parts
def crop_resize_data(image, label=None, image_size=(1024, 384), offset=690):
    """
    Attention:
    h,w, c = image.shape
    cv2.resize(image,(w,h))
    """
    roi_image = image[offset:, :]
    if label is not None:
        roi_label = label[offset:, :]
        train_image = cv2.resize(roi_image, image_size, interpolation=cv2.INTER_LINEAR)
        train_label = cv2.resize(roi_label, image_size, interpolation=cv2.INTER_NEAREST)
        return train_image, train_label
    else:
        train_image = cv2.resize(roi_image, image_size, interpolation=cv2.INTER_LINEAR)
        return train_image
