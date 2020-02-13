#coding：utf-8
#@author： Jiangnan He
# date： 2020.02.13
#模型融合的策略是三个模型预测值取平均作为最终的输出
# 预测的图片名为    test_dir + network + color_mask.jpg

'''
import os
import numpy as np
import cv2
test_dir="test_example"
out1=cv2.imread(os.path.join(test_dir,"deeplabv3p","color_mask"))
out2 = cv2.imread(os.path.join(test_dir, "unet", "color_mask"))
img1 = out1.astype(np.float32)
img2 = out2.astype(np.float32)
img = (img1+img2)/2.0
img = img.astype(np.uint8)
cv2.imwrite('final.jpg',img)
'''


#用投票的思想，少数服从多数
import numpy as np
import cv2
import os
test_dir="test_example"
RESULT_PREFIXX = ['deeplabv3p', 'unet']
# each mask has 8 classes: 0~7
def vote_per_image():
    result_list = []
    for j in range(len(RESULT_PREFIXX)):
        im = cv2.imread( os.path.join(test_dir,RESULT_PREFIXX[j] )+ '.jpg', 0)
        result_list.append(im)
    # each pixel
    height, width = result_list[0].shape
    vote_mask = np.zeros((height, width))

    for h in range(height):
        for w in range(width):
            record = np.zeros((1, 8))
            for n in range(len(result_list)):
                mask = result_list[n]
                pixel = mask[h, w]
                # print('pix:',pixel)
                record[0, pixel] += 1
            label = record.argmax()
            # print(label)
            vote_mask[h, w] = label
    cv2.imwrite('vote_mask'  + '.jpg', vote_mask)
vote_per_image()
