# coding:utf-8
import cv2
from scipy.misc import imresize
import numpy as np
import os
from keras.engine.saving import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = load_model('./saveModel/realFakeDiscern.h5')



label = ['fake', 'real']
print(label)
# 导入模型分类器
cascadePath = "./faces_detector_model/haarcascade_frontalface_default.xml"

# 从预构建的模型中创建分类器
faceCascade = cv2.CascadeClassifier(cascadePath)

# 设定字体样式
font = cv2.FONT_HERSHEY_SIMPLEX

# 初始化并启动视频帧捕获
cam = cv2.VideoCapture(0)

# Loop
while True:
    # 读取视频帧
    ret, im =cam.read()

    # 将捕获的帧转换为灰度
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # 从视频框架得到所有的脸
    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    # 每一张脸
    for(x,y,w,h) in faces:

        # 在面部周围创建矩形
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        # 自己训练的模型
        face_data = imresize(np.array(im[y:y + h, x:x + w]),(32,32)).reshape((1,32,32,3))

        ID = np.argmax(model.predict(face_data),axis=1)

        for i in ID:
            name = label[i]
            cv2.rectangle(im, (x - 22, y - 90), (x + w + 22, y - 22), (0, 255, 0), -1)
            cv2.putText(im, str(name), (x, y - 40), font, 1, (255, 255, 255), 3)

    # 用有界矩形显示视频帧
    cv2.imshow('im',im)

    # 如果按下“q”键，则关闭程序
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# 停止相机
cam.release()

# 关闭所有窗口
cv2.destroyAllWindows()
