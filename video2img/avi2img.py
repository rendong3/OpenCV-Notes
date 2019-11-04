#-*- coding: utf-8 -*-
import os
import cv2


video_path = '/home/ren_dong/PycharmProjects/OpenCv/blog/video/1.avi'
save_path = '/home/ren_dong/PycharmProjects/OpenCv/blog/img/'

try:
    if not os.path.exists(save_path):
        os.mkdir(save_path)
except:
    print('path is exist')


def video2img(video_path, save_path):

    cap = cv2.VideoCapture(video_path)
    isopened = cap.isOpened()

    i = 1
    if cap.isOpened():
        flag, frame = cap.read()
        
    else:
        flag = False
        print('video is not open..')
        

    while(flag):
            
        flag, frame = cap.read()
        cv2.imwrite(save_path + str(i) + '.jpg', frame)
        i += 1


    cap.release()
if __name__ == '__main__':

    video2img(video_path, save_path)
