# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os
import shutil

'''

       created on  10:24:27 2018-11-18

       @author:ren_dong

       人脸识别  调用摄像头进行个人头像数据采集,作为数据库

       只是在原来检测人脸目标框的基础上,添加了resize()函数进行图像的resize,
       然后调用imwrite()函数进行保存到指定路径下

'''

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 19:41:19 2018

@author: lenovo
"""

'''
调用opencv库实现人脸识别
'''

# 读取pgm图像，并显示
def ShowPgm(filepath):
    cv2.namedWindow('pgm')
    img = cv2.imread(filepath)
    cv2.imshow('pgm', img)
    print(img.shape)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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


def FaceRec(data):
    # 加载训练数据
    X, y, names = LoadImages('./data')


    model = cv2.face.EigenFaceRecognizer_create()
    model.train(X, y)

    # 创建一个级联分类器 加载一个 .xml 分类器文件. 它既可以是Haar特征也可以是LBP特征的分类器.
    face_cascade = cv2.CascadeClassifier('./cascade/haarcascade_frontalface_default.xml')

    # 打开摄像头
    camera = cv2.VideoCapture(0)
    cv2.namedWindow('Dynamic')

    while (True):
        # 读取一帧图像
        ret, frame = camera.read()
        # 判断图片读取成功？
        if ret:
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 人脸检测

            faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
            for (x, y, w, h) in faces:
                # 在原图像上绘制矩形
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray_img[y:y + h, x:x + w]

                try:
                    # 宽92 高112
                    roi_gray = cv2.resize(roi_gray, (92, 112), interpolation=cv2.INTER_LINEAR)
                    params = model.predict(roi_gray)
                    print('Label:%s,confidence:%.2f' % (params[0], params[1]))
                    cv2.putText(frame, names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                except:
                    continue

            cv2.imshow('Dynamic', frame)
            # 如果按下q键则退出
            if cv2.waitKey(100) & 0xff == ord('q'):
                break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # ShowPgm('./face/s1/1.pgm')
    data = './data'
    # 生成自己的人脸数据
    # generator(data)
    FaceRec(data)