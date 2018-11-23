# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np

'''
        created on 09:05:10 2018-11-22
        @author ren_dong

        使用ORB特征匹配

'''
def orb_test():
    # 加载图片  灰色
    img1 = cv2.imread('orb1.jpg')
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread('orb2.jpg')
    img2 = cv2.resize(img2, dsize=(450, 300))
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    image1 = gray1.copy()
    image2 = gray2.copy()

    '''
    1.使用ORB算法检测特征点、描述符
    '''
    orb = cv2.ORB_create(128)
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
    # 在图像上绘制关键点
    image1 = cv2.drawKeypoints(image=image1, keypoints=keypoints1, outImage=image1, color=(255, 0, 255),
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image2 = cv2.drawKeypoints(image=image2, keypoints=keypoints2, outImage=image2, color=(255, 0, 255),
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # 显示图像
    cv2.imshow('orb_keypoints1', image1)
    cv2.imshow('orb_keypoints2', image2)
    cv2.waitKey(20)

    '''
    2、匹配
    '''
    matcher = cv2.BFMatcher_create(cv2.HAMMING_NORM_TYPE, crossCheck=True)
    matchePoints = matcher.match(descriptors1, descriptors2)
    print(type(matchePoints), len(matchePoints), matchePoints[0])
    # 按照距离从小到大排序，选取最优匹配的
    sorted(matchePoints, key=lambda x: x.distance)
    # 绘制最优匹配点
    outImg = None
    outImg = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matchePoints[:40], outImg, matchColor=(0, 255, 0),
                             flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    cv2.imshow('matche', outImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    orb_test()