#coding=utf-8
#@author : Jiangnan He
#@date : 2019.12.25 15:47

"""
img_process 生成的文件结构如下：
--train_set :D:\Dataset\Apollolaneline\train_set   - gt_image
                                                   - src_image

-- test_set :D:\Dataset\Apollolaneline\test_set     - src_image
"""
import pandas as pd
import os
from sklearn.utils import shuffle
import glob
# train_set 共有21914 个样本 我们按照 train：val：test=3:1：1 的比例划分 训练集和 验证集 测试集
datasetpath1=datasetpath='D:/Dataset/Apollolaneline/'
#================================================
# make train & validation & test lists
#================================================
def get_train_val_test():
    img_list=[]
    lab_list=[]
    for file in glob.glob(str(datasetpath1)+"/Road*/"):
        if file.find("zip")!=-1:
            continue
        for i in glob.glob(str(file)+"/ColorImage_road*"):
                file1=os.path.join(datasetpath1,i,"ColorImage")
                for j in os.listdir(file1):
                    path=os.path.join(file1,j)
                    for k in os.listdir(path):
                        path1=os.path.join(path,k)
                        print(path1,len(os.listdir(path1)))
                        for z in os.listdir(path1):
                            image_path=os.path.join(path1,z)
                            # D:\Dataset\Apollolaneline\Road02\ColorImage_road02\ColorImage\Record001\Camera 5\170927_063811892_Camera_5.jpg
                            #D:\Dataset\Apollolaneline\Gray_Label\Gray_Label\Label_road02\Label\Record001\Camera 5\170927_063811892_Camera_5_bin.png
                            print(image_path,"image_path")
                            image_name=str(image_path.split()[1]).split('\\')[1].split(".")[0]+'_bin.png'
                            print("image_name",image_name)
                            #获取src_img 文件地址
                            src_path=os.path.join(path1,z)
                            # 获取gt_img 文件地址
                            gt_path=os.path.join("D:/Dataset/Apollolaneline/Gray_Label/Gray_Label/","Label_"+str(image_path.split("\\")[2]).split("_")[1],"Label","/".join(image_path.split("\\")[4:6]),image_name)
                            img_list.append(src_path)
                            lab_list.append(gt_path)
    assert len(img_list)==len(lab_list)
    total_len=len(img_list)
    six_part=int(total_len*0.6)
    eight_part=int(total_len*0.8)
    all=pd.DataFrame({'image': img_list, 'label': lab_list})
    all_shuffle = shuffle(all)  # 打乱存储
    train_list=all_shuffle[:six_part]
    val_list=all_shuffle[six_part:eight_part]
    test_list=all_shuffle[eight_part:]
    #生成val.csv
    train_list.to_csv('../data_list/train.csv', index=False)#生成val的数据集csv文件
    # 生成train.csv
    val_list.to_csv('../data_list/val.csv', index=False)#生成train的数据集csv文件
    #生成test.csv
    test_list.to_csv('../data_list/test.csv', index=False)  # 生成test的数据集csv文件

if __name__=="__main__":
    get_train_val_test()


