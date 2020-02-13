# coding=utf-8
# @author：Jiangnan He
# @date： 2019.12.25 18:54
# 本实现用于生成 pytorch 的数据集

import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from utils.lab_process import encode_labels
from utils.img_process import Process_dataset
from utils.img_process import ToTensor
import cv2


class DatasetFromCSV(Dataset):
    def __init__(self, csv_path, transforms=None):
        self.data = pd.read_csv(csv_path,header=None, names=["image","label"])
        self.images = self.data["image"].values[1:]
        self.labels = self.data["label"].values[1:]
        self.transforms = transforms

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        image=self.images[index]
        label=self.labels[index]
        image=cv2.imread(image,cv2.IMREAD_COLOR)
        lab_as_np=cv2.imread(label, cv2.IMREAD_GRAYSCALE)
        label=encode_labels(lab_as_np)#将不同灰度值标记为不同类 # 后续需要将其转为one-hot编码
        sample=[image,label]
        img_as_tensor ,lab_as_tensor= self.transforms(sample)
        return img_as_tensor, lab_as_tensor

#测试
if __name__=="__main__":
    batch_size =2
    transform = transforms.Compose([Process_dataset(),ToTensor()] )
    train_data = DatasetFromCSV('../data_list/train.csv',transform)
    val_data = DatasetFromCSV('../data_list/val.csv', transform)
    test_data = DatasetFromCSV("../data_list/test.csv",transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)


    for batch in train_loader:
         img, lab = batch
         lab=torch.unsqueeze(lab,1)
         print(img.shape,lab.shape)


