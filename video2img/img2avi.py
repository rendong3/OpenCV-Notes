import os
import cv2


def img2avi():

    file_path = '/home/ren_dong/PycharmProjects/OpenCv/blog/imgs'
    save_path = file_path + '/' + '%s.avi' % str('result')

    file_list = os.listdir(file_path)
#视频的帧速率，秒为单位，一秒播放多少张图片
    fps = 2
    img = cv2.imread('/home/ren_dong/PycharmProjects/OpenCv/blog/imgs/000001.jpg')
    size = (img.shape[1],img.shape[0])
#size视频图片的尺寸
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter(save_path, fourcc, float(fps), size)
#cv2.VideoWriter(1,2,3,4)
#参数1 文件名称  参数2：选择编译器 参数3:设置帧率  参数4：设置视频尺寸


    for item in file_list:
        if item.endswith('.jpg'):
            item  = file_path + '/' + item
            img = cv2.imread(item)
            img = cv2.resize(img, size)
            videoWriter.write(img)


    videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img2avi()
    

    

