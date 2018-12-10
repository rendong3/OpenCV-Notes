#coding=utf-8
#绘制面部轮廓
import cv2
import face_recognition
from PIL import Image, ImageDraw

# 将图片文件加载到numpy 数组中
img = cv2.imread('images/001.jpg')
img1 = cv2.resize(img, (400,600), interpolation=cv2.INTER_CUBIC)
cv2.imwrite('images/002.jpg',img1)
image = face_recognition.load_image_file('images/002.jpg')

#查找图像中所有面部的所有面部特征
face_landmarks_list = face_recognition.face_landmarks(image)
print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

for face_landmarks in face_landmarks_list:
    facial_features = [
        'chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge', 'nose_tip',
        'left_eye', 'right_eye', 'top_lip', 'bottom_lip'
    ]
    pil_image = Image.fromarray(image)

    d = ImageDraw.Draw(pil_image)

    for facial_feature in facial_features:
        d.line(face_landmarks[facial_feature], fill=(0, 255, 0), width=2)
        print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

    pil_image.show()

