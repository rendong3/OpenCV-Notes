# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 09:17:37 2018

@author: ren_dong
"""

'''
HOG检测人
'''
import cv2
import numpy as np


def is_inside(o, i):
    '''
    判断矩形o是不是在i矩形中

    args:
        o：矩形o  (x,y,w,h)
        i：矩形i  (x,y,w,h)
    '''
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih


def draw_person(img, person):
    '''
    在img图像上绘制矩形框person

    args:
        img：图像img
        person：人所在的边框位置 (x,y,w,h)
    '''
    x, y, w, h = person
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)


def detect_test():
    '''
    检测人
    '''
    img = cv2.imread('./images/person.jpg')
    rows, cols = img.shape[:2]
    sacle = 0.5
    print('img',img.shape)
    img = cv2.resize(img, dsize=(int(cols * sacle), int(rows * sacle)))
    # print('img',img.shape)
    # img = cv2.resize(img, (128,64))

    # 创建HOG描述符对象
    # 计算一个检测窗口特征向量维度：(64/8 - 1)*(128/8 - 1)*4*9 = 3780
    '''
    winSize = (64,128)
    blockSize = (16,16)    
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9    
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)  
    '''
    hog = cv2.HOGDescriptor()
    # hist = hog.compute(img[0:128,0:64])   计算一个检测窗口的维度
    # print(hist.shape)
    detector = cv2.HOGDescriptor_getDefaultPeopleDetector()
    print('detector', type(detector), detector.shape)
    hog.setSVMDetector(detector)

    # 多尺度检测，found是一个数组，每一个元素都是对应一个矩形，即检测到的目标框
    found, w = hog.detectMultiScale(img)
    print('found', type(found), found.shape)

    # 过滤一些矩形，如果矩形o在矩形i中，则过滤掉o
    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            # r在q内？
            if ri != qi and is_inside(r, q):
                break
        else:
            found_filtered.append(r)

    for person in found_filtered:
        draw_person(img, person)

    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_test()