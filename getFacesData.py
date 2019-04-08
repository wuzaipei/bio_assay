# coding:utf-8
import cv2
import os

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


# 开始捕获视频
realPath = "./data/fakeFaces.mp4"
vid_cam = cv2.VideoCapture(0)

# 利用Haarcascade正面检测视频流中的目标
face_detector = cv2.CascadeClassifier('./faces_detector_model/haarcascade_frontalface_default.xml')

# 每录入一张人脸的时候在这里写一个id，记住一点就是每个人的ID都不能相同。
face_id = 'real'

# 初始化样本人脸图像
count = 0
path = "./data/trainingData/"
assure_path_exists(path)

# 开始循环
while (True):

    # 捕获的视频帧
    _, image_frame = vid_cam.read()

    if _ == False:
        break
    # 帧转换为灰度图
    # gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # 检测不同大小的帧，人脸矩形列表，返回四个值就是人脸位置的坐标
    faces = face_detector.detectMultiScale(image_frame, 1.3, 5)

    # Loops for each faces
    for (x, y, w, h) in faces:
        # 将图像帧裁剪成矩形
        cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 增量样本人脸图像
        count += 1

        # 将捕获的图像保存到数据集文件夹中
        cv2.imwrite(path + str(face_id) + '_' + str(count) + ".jpg", image_frame[y:y + h, x:x + w])

        # 显示视频帧，在人的脸上有一个有边界的矩形
    cv2.imshow('frame', image_frame)

    # 停止录像，按“q”键至少100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # 如果拍摄的图像达到100，停止拍摄视频
    elif count == 200:
        break

# 停止视频
vid_cam.release()

# 关闭所有已启动的窗口
cv2.destroyAllWindows()
