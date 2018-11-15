# -*- coding: utf-8 -*-

from numpy import *
from scipy import *
import numpy as np
import cv2
import os

''''
        对指定文件夹进行图片遍历,并添加椒盐和高斯噪声,可以进行数据增强

'''

#定义添加椒盐噪声的函数
def SaltAndPepper(src,percetage):
    SP_NoiseImg=src
    SP_NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(SP_NoiseNum):
        randX=random.random_integers(0,src.shape[0]-1)
        randY=random.random_integers(0,src.shape[1]-1)
        if random.random_integers(0,1)==0:
            SP_NoiseImg[randX,randY]=0
        else:
            SP_NoiseImg[randX,randY]=255
    return SP_NoiseImg
#定义添加高斯噪声的函数
def addGaussianNoise(image,percetage):
    G_Noiseimg = image
    G_NoiseNum=int(percetage*image.shape[0]*image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(20,40)
        temp_y = np.random.randint(20,40)
        G_Noiseimg[temp_x][temp_y] = 255
    return G_Noiseimg

'''
            单张图片测试
'''
# img = cv2.imread("1.jpg")
# cv2.imshow('org',img)
# salt = SaltAndPepper(img, 0.01)
# cv2.imshow('salt',salt)
# gauss = addGaussianNoise(img, 0.01)
# cv2.imshow('gauss',gauss)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
        指定文件夹批量测试

'''
class Batch():
    def __init__(self):
        self.path = '/home/ren_dong/PycharmProjects/OpenCv/img'
        self.savepath = '/home/ren_dong/PycharmProjects/OpenCv/save/'

    def add_noise(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 0
        n = 1
        while(n<=3):
            for item  in filelist:
                if item.endswith('.jpg'):

                    item = self.path + '/'  + str(item)
                    img = cv2.imread(item)
                    cv2.imshow('org', img)
                    percentage = 0.01
                    percentage = percentage*(n+1)

                    salt = SaltAndPepper(img, percentage)

                    gauss = addGaussianNoise(salt, percentage)

                    cv2.imshow('salt',salt)
                    cv2.imshow('gauss',gauss)

                    k = cv2.waitKey(0)
                    if k == 27:
                        cv2.destroyAllWindows()
                    elif k == ord('s'):
                        for count in range(1):
                            cv2.imwrite(self.savepath + str(i) + '.jpg', salt)
                            cv2.imwrite(self.savepath + str(i) + '0' + '.jpg', gauss)
                            count += 1
                    else:
                        for count in range(1):
                            cv2.imwrite(self.savepath +  str(i) + '0' + '.jpg', gauss)
                            count += 1

                i += 1

            n += 1



if __name__ == '__main__':
    demo = Batch()
    demo.add_noise()
