# -*- coding:utf-8 -*-
import cv2
import numpy as np

'''
       created on  08:10:27 2018-11-15
       @author:ren_dong

                Grabcut 图像分割  
                
                加入鼠标回调函数,可以进行交互式操作,鼠标选择矩形区域作为ROI
                
               cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount[, mode]) → None
               on_mouse()


'''

# 鼠标事件的回调函数
def on_mouse(event, x, y, flag, param):
    global rect
    global leftButtonDown
    global leftButtonUp

    # 鼠标左键按下
    if event == cv2.EVENT_LBUTTONDOWN:
        rect[0] = x
        rect[2] = x
        rect[1] = y
        rect[3] = y
        leftButtonDown = True
        leftButtonUp = False

    # 移动鼠标事件
    if event == cv2.EVENT_MOUSEMOVE:
        if leftButtonDown and not leftButtonUp:
            rect[2] = x
            rect[3] = y

            # 鼠标左键松开
    if event == cv2.EVENT_LBUTTONUP:
        if leftButtonDown and not leftButtonUp:
            x_min = min(rect[0], rect[2])
            y_min = min(rect[1], rect[3])

            x_max = max(rect[0], rect[2])
            y_max = max(rect[1], rect[3])

            rect[0] = x_min
            rect[1] = y_min
            rect[2] = x_max
            rect[3] = y_max
            leftButtonDown = False
            leftButtonUp = True


# 读入图片
img = cv2.imread('ouc.jpg')
'''
    
     掩码图像，如果使用掩码进行初始化，那么mask保存初始化掩码信息；
     在执行分割的时候，也可以将用户交互所设定的前景与背景保存到mask中，
     然后再传入grabCut函数；
     在处理结束之后，mask中会保存结果

'''
#img.shape[:2]=img.shape[0:2] 代表取彩色图像的长和宽  img.shape[:3] 代表取长+宽+通道
mask = np.zeros(img.shape[:2], np.uint8)

# 背景模型，如果为None，函数内部会自动创建一个bgdModel；bgdModel必须是单通道浮点型图像，且行数只能为1，列数只能为13x5；
bgdModel = np.zeros((1, 65), np.float64)
# fgdModel——前景模型，如果为None，函数内部会自动创建一个fgdModel；fgdModel必须是单通道浮点型图像，且行数只能为1，列数只能为13x5；
fgdModel = np.zeros((1, 65), np.float64)

# 用于限定需要进行分割的图像范围，只有该矩形窗口内的图像部分才被处理；
# rect 初始化
rect = [0, 0, 0, 0]

# 鼠标左键按下
leftButtonDown = False
# 鼠标左键松开
leftButtonUp = True

# 指定窗口名来创建窗口
cv2.namedWindow('img')
# 设置鼠标事件回调函数 来获取鼠标输入
cv2.setMouseCallback('img', on_mouse)

# 显示图片
cv2.imshow('img', img)


##设定循环,进行交互式操作
while cv2.waitKey(2) == -1:
    # 左键按下，画矩阵
    if leftButtonDown and not leftButtonUp:

        img_copy = img.copy()
        # 在img图像上，绘制矩形  线条颜色为green 线宽为2
        cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
        # 显示图片
        cv2.imshow('img', img_copy)

    # 左键松开，矩形画好
    elif not leftButtonDown and leftButtonUp and rect[2] - rect[0] != 0 and rect[3] - rect[1] != 0:
        # 转换为宽度高度
        rect[2] = rect[2] - rect[0]
        rect[3] = rect[3] - rect[1]
        # rect_copy = tuple(rect.copy())
        rect_copy = tuple(rect)
        rect = [0, 0, 0, 0]
        # 物体分割
        cv2.grabCut(img, mask, rect_copy, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img_show = img * mask2[:, :, np.newaxis]
        # 显示图片分割后结果
        cv2.imshow('grabcut', img_show)
        # 显示原图
        cv2.imshow('img', img)

cv2.waitKey()
cv2.destroyAllWindows()