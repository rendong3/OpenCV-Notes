# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os

'''

       created on  10:24:27 2018-11-18

       @author:ren_dong

       人脸识别  
       利用数据进行数据集制作

'''


# 2、读取ORL人脸数据库 准备训练数据
def LoadImages(data):
    '''
    加载数据集
    params:
        data:训练集数据所在的目录，要求数据尺寸大小一样
    ret:
        images:[m,height,width]  m为样本数,height为高,width为宽
        names：名字的集合
        labels：标签
    '''
    images = []
    labels = []
    names = []

    label = 0
    # 过滤所有的文件夹
    for subDirname in os.listdir(data):

        subjectPath = os.path.join(data, subDirname)
        #subjectpath即每一类所在的文件夹
        if os.path.isdir(subjectPath):
            # 每一个文件夹下存放着一个人的照片
            names.append(subDirname)
            ##得到图片名字filename

            for fileName in os.listdir(subjectPath):
                #路径拼接,得到图片路径
                imgPath = os.path.join(subjectPath, fileName)
                #遍历路径按照文件名 读取图片
                img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
                #添加images
                images.append(img)
                #添加label
                labels.append(label)
            label += 1
    images = np.asarray(images)
    labels = np.asarray(labels)
    return images, labels, names


if __name__ == '__main__':
    data  = './data'
    LoadImages(data)