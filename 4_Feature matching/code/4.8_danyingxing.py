# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 13:22:10 2018

@author: zy
"""

'''
单应性匹配
'''

import numpy as np
import cv2


def flann_hom_test():
    # 加载图像
    img1 = cv2.imread('orb1.jpg')  # queryImage
    img2 = cv2.imread('orb2.jpg')  # trainImage
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # img2 = cv2.resize(img2,dsize=(450,600))

    MIN_MATCH_COUNT = 10

    '''
    1.使用SIFT算法检测特征点、描述符
    '''
    sift = cv2.xfeatures2d.SIFT_create(100)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    # 在图像上绘制关键点
    # img1 = cv2.drawKeypoints(image=img1,keypoints = kp1,outImage=img1,color=(255,0,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # img2 = cv2.drawKeypoints(image=img2,keypoints = kp2,outImage=img2,color=(255,0,255),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # 显示图像
    # cv2.imshow('sift_keypoints1',img1)
    # cv2.imshow('sift_keypoints2',img2)
    # cv2.waitKey(20)

    '''
    2、FLANN匹配 
    '''
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(des1, des2, k=2)

    # 将不满足的最近邻的匹配之间距离比率大于设定的阈值匹配剔除。
    goodMatches = []
    minRatio = 0.7
    for m, n in matches:
        if m.distance / n.distance < minRatio:
            goodMatches.append(m)  # 注意 如果使用drawMatches 则不用写成List类型[m]

    '''
    3、单应性变换
    '''
    # 确保至少有一定数目的良好匹配(理论上，计算单应性至少需要4对匹配点，实际上会使用10对以上的匹配点)
    if len(goodMatches) > MIN_MATCH_COUNT:
        # 获取匹配点坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 2)

        print('src_pts：', len(src_pts), src_pts[0])
        print('dst_pts：', len(dst_pts), dst_pts[0])

        # 获取单应性：即一个平面到另一个平面的映射矩阵
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # print('M：',M,type(M))   #<class 'numpy.ndarray'> [3,3]
        matchesMask = mask.ravel().tolist()  # 用来配置匹配图，只绘制单应性图片中关键点的匹配线
        # 由于使用的是drawMatches绘制匹配线，这里list
        # 每个元素也是一个标量，并不是一个list
        print('matchesMask：', len(matchesMask), matchesMask[0])

        # 计算原始图像img1中书的四个顶点的投影畸变，并在目标图像img2上绘制边框
        h, w = img1.shape[:2]
        # 源图片中书的的四个角点
        pts = np.float32([[55, 74], [695, 45], [727, 464], [102, 548]]).reshape(-1, 1, 2)
        print('pts：', pts.shape)
        dst = cv2.perspectiveTransform(pts, M)
        print('dst:', dst.shape)
        # 在img2上绘制边框
        img2 = cv2.polylines(img2, [np.int32(dst)], True, (0, 255, 0), 2, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" % (len(goodMatches), MIN_MATCH_COUNT))
        matchesMask = None

    '''
    绘制显示效果
    '''
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, goodMatches, None, **draw_params)
    cv2.imshow('matche', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    flann_hom_test()