# coding=utf-8
# 人脸识别类 - 使用face_recognition模块
import cv2
import face_recognition
import os
from PIL import Image, ImageDraw, ImageFont
import numpy
import asyncio
from websocket import create_connection
import threading
import time
from PIL import Image
import os

facename = "未知"
isCapLoop = True


def imgsFlip():

    dir_img = "preimg/"
    # 待处理的图片地址
    dir_save = "img/"
    # 水平镜像翻转后保存的地址

    list_img = os.listdir(dir_img)

    for img_name in list_img:
        if img_name.endswith(".png") or img_name.endswith(".jpg"):
            pri_image = Image.open(dir_img+img_name)
            tmppath = dir_save + img_name
            pri_image.transpose(Image.FLIP_LEFT_RIGHT).save(tmppath)


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "font/Hei.ttf", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)


def faceloop():
    path = "img"  # 模型数据图片目录
    cap = cv2.VideoCapture(0)
    Width, Height = 640, 480
    cap.set(3, Width)
    cap.set(4, Height)

    # 提取图片并编码写入到编码数组对象，同时提取图片名字，写入另一数组用于提取图片名。
    total_image_name = []
    total_face_encoding = []
    for fn in os.listdir(path):  # fn 表示的是文件名q
        print(path + "/" + fn)
        if fn.split(".")[-1] in ["png", "jpg"]:
            total_face_encoding.append(
                face_recognition.face_encodings(
                    face_recognition.load_image_file(path + "/" + fn))[0])
            total_image_name.append(fn.split("."))  # 图片名字列表
    # 提取图片结束
    name = "未知"
    process_this_frame = True
    global isCapLoop
    while isCapLoop:

        _, frame = cap.read()

        frame = cv2.flip(frame, 1)  # 左右翻转，符合镜像视觉

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # 发现在视频帧所有的脸和face_enqcodings
        face_locations = face_recognition.face_locations(
            rgb_small_frame)  # 获取图片中的所有人脸的位置信息
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        # 在这个视频帧中循环遍历每个人脸
        for (top, right, bottom, left), face_encoding in zip(
                face_locations, face_encodings):

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            if process_this_frame:
                # 看看面部是否与已知人脸相匹配。
                for i, v in enumerate(total_face_encoding):
                    match = face_recognition.compare_faces(
                        [v], face_encoding, tolerance=0.4)  # 容错率0.4比较符合亚洲人
                    name = "未知"
                    if match[0]:
                        name = "%s" % (total_image_name[i][0])
                        break
            # 画出一个框，框住脸
            # 如果在中间部位，则框框为蓝色
            global facename
            facename = "无"
            mainColor = (53, 67, 203)
            if abs(abs(left-right)/2+left - Width/2) < Width/6 and abs(abs(top-bottom)/2+top - Height/2) < Height/4:
                if name != "未知":
                    mainColor = (99, 180, 40)
                facename = name

            cv2.rectangle(frame, (left, top),
                          (right, bottom), mainColor, 2)
            # 画出一个带名字的标签，放在框下
            cv2.rectangle(frame, (left-2, bottom + 35), (right+2, bottom), mainColor,
                          cv2.FILLED)
            frame = cv2ImgAddText(frame, name, left+6,
                                  bottom+10, (255, 255, 255), 20)

        process_this_frame = not process_this_frame
        # 显示结果图像
        cv2.imshow('face recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            isCapLoop = False
            break

    cap.release()
    try:
        cv2.destroyAllWindows()
    except:
        return


def wscli():
    uri = "ws://localhost:6789"
    global isCapLoop
    global facename
    try:
        ws = create_connection(uri)
        while isCapLoop:
            time.sleep(1)
            message = '{"from":"face","msg":"%s"}' % facename
            ws.send(message)
            print(f"> {message}")
            reply = ws.recv()
            print(f"< {reply}")
        ws.close()
    except:
        # isCapLoop = False
        print("websocket disconnect")
        if isCapLoop:
            print("Retry after 3 seconds")
            time.sleep(3)
            wscli()


if __name__ == '__main__':

    imgsFlip()
    t = threading.Thread(target=wscli, name='wscli')
    t.start()
    faceloop()
    t.join()
