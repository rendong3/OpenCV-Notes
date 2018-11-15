# -*- coding:utf-8 -*-
import cv2
import numpy as np

'''
       created on  08:10:27 2018-11-15
       @author:ren_dong

            Grabcut 图像分割  
                
            直接给定矩形区域作为ROI
                
            GrabCut算法的实现步骤：
            
                1 在图片中定义(一个或者多个)包含物体的矩形。
                2 矩形外的区域被自动认为是背景。
                3 对于用户定义的矩形区域，可用背景中的数据来区分它里面的前景和背景区域。
                4 用高斯混合模型(GMM)来对背景和前景建模，并将未定义的像素标记为可能的前景或者背景。
                5 图像中的每一个像素都被看做通过虚拟边与周围像素相连接，而每条边都有一个属于前景或者背景的概率，这是基于它与周边像素颜色上的相似性。
                6 每一个像素(即算法中的节点)会与一个前景或背景节点连接。
                7 在节点完成连接后(可能与背景或前景连接)，若节点之间的边属于不同终端(即一个节点属于前景，另一个节点属于背景)，则会切断他们之间的边，这就能将图像各部分分割出来。

                
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount[, mode]) → None


'''
#载入图像img
fname = 'test.jpg'
img = cv2.imread(fname)

#设定矩形区域  作为ROI         矩形区域外作为背景
rect = (275, 120, 170, 320)

#img.shape[:2]得到img的row 和 col ,
# 得到和img尺寸一样的掩模即mask ,然后用0填充
mask = np.zeros(img.shape[:2], np.uint8)

#创建以0填充的前景和背景模型,  输入必须是单通道的浮点型图像, 1行, 13x5 = 65的列 即(1,65)
bgModel = np.zeros((1,65), np.float64)
fgModel = np.zeros((1,65), np.float64)

#调用grabcut函数进行分割,输入图像img, mask,  mode为 cv2.GC_INIT_WITH-RECT
cv2.grabCut(img, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)

##调用grabcut得到rect[0,1,2,3],将0,2合并为0,   1,3合并为1  存放于mask2中
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)

#得到输出图像
out = img * mask2[:, :, np.newaxis]

cv2.imshow('origin', img)
cv2.imshow('grabcut', out)
cv2.waitKey()
cv2.destroyAllWindows()

