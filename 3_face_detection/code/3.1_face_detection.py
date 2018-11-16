# -*- coding:utf-8 -*-
import cv2
import numpy as np

'''
       
       created on  08:29:27 2018-11-16
       
       @author:ren_dong
       
       haar级联分类器实现静态图像人脸检测
       
        1、准备人脸、非人脸样本集；

        2、计算特征值和积分图；

        3、筛选出T个优秀的特征值（即最优弱分类器）；

        4、把这个T个最优弱分类器传给AdaBoost进行训练。

        5、级联，也就是强分类器的强强联手。
       
        cv2.CascadeClassifier([filename]) → <CascadeClassifier object>
        
        cv2.CascadeClassifier.detectMultiScale(image[, scaleFactor[, minNeighbors[, flags[, minSize[, maxSize]]]]]) → objects

'''
filename = './images/face1.jpg'

def detect(filename):

    #声明face_cascade对象,该变量为cascadeclassifier对象,它负责人脸检测
    face_cascade = cv2.CascadeClassifier('./cascade/haarcascade_frontalface_default.xml')

    img= cv2.imread(filename)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # 进行人脸检测，传入scaleFactor，minNegihbors，分别表示人脸检测过程中每次迭代时图像的压缩率以及
    # 每个人脸矩形保留近似数目的最小值

    # 返回人脸矩形数组
    face = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in face:

        #绘制矩形
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2 )

    #创建窗口
    cv2.namedWindow('face_detection')

    cv2.imshow('face_detection', img)

    cv2.waitKey()

    cv2.destroyAllWindows()


if __name__  == '__main__':
    detect(filename)