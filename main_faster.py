# coding=utf-8
# 人脸识别类 - 使用face_recognition模块
import cv2
import face_recognition
import os
from PIL import Image, ImageDraw, ImageFont
import numpy
from websocket import create_connection
import threading
import time
from PIL import Image
import os
import getopt
import sys
import pickle

facename = "未知"
isCapLoop = True
total_image_name = []
total_face_encoding = []


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    # cv2 不支持中文，因此讲中文文字做成图片写入到图片中
    if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "font/Hei.ttf", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)


def patchImages(staticdata=False):
     # 提取图片并编码写入到编码数组对象，同时提取图片名字，写入另一数组用于提取图片名。
     # staticdata如果为真，则读取静态数据
     # staticdata如果为假，则从img目录读取图片，并且生成静态数据
    path = "img"  # 模型数据图片目录
    global total_image_name
    global total_face_encoding

    if not staticdata:
        for fn in os.listdir(path):  # fn 表示的是文件名q
            print(path + "/" + fn)
            if fn.split(".")[-1] in ["png", "jpg"]:
                total_face_encoding.append(
                    face_recognition.face_encodings(
                        face_recognition.load_image_file(path + "/" + fn))[0])
                total_image_name.append(fn.split("."))  # 图片名字列表
        # 提取图片结束
        f = open('data_image_names.dat', 'wb')
        pickle.dump(total_image_name, f)
        f.close()

        f = open('data_face_encoding.dat', 'wb')
        pickle.dump(total_face_encoding, f)
        f.close()

    else:
        f = open('data_image_names.dat', 'rb')
        total_image_name = pickle.load(f)
        f.close()

        f = open('data_face_encoding.dat', 'rb')
        total_face_encoding = pickle.load(f)
        f.close()


def faceloop(capIndex=0, Width=640, Height=480, model="hog", tolerance=0.5, number_of_times_to_upsample=1):

    cap = cv2.VideoCapture(capIndex)
    cap.set(3, Width)
    cap.set(4, Height)

    process_frame_index = 5

    global isCapLoop
    names = []

    while isCapLoop:

        _, frame = cap.read()

        frame = cv2.flip(frame, 1)  # 左右翻转，符合镜像视觉
        w, h = len(frame[0]), len(frame)  # 获得图片的高宽像素

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # 缩小图片
        rgb_small_frame = small_frame[:, :, ::-1]  # 只保留rgb颜色

        # 发现在视频帧所有的脸和face_enqcodings
        face_locations = face_recognition.face_locations(
            rgb_small_frame, model=model, number_of_times_to_upsample=number_of_times_to_upsample)  # 获取图片中的所有人脸的位置信息

        # 左右顺序排列，避免每次解析得到的数组顺序不同，影响之后标记名字
        face_locations = sorted(
            face_locations, key=(lambda x: x[1]), reverse=True)

        # 进行人脸识别比对的部分，每隔5帧执行1次
        if process_frame_index % 5 == 0:
            process_frame_index = 0

            # 得到所有识别出来的脸的编码数组
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations)

            names = []  # 创建一个名字数组

            # 从得到的人脸数据中循环匹配预设的人脸，并且找到近似度最高的一个，然后从totle名字数组中得到名字放在names里
            for face_encoding in face_encodings:

                # 得到所有预设人脸匹配度
                face_distances = face_recognition.face_distance(
                    total_face_encoding, face_encoding)

                # 得到匹配度最高的序号
                best_match_index = numpy.argmin(face_distances)

                # （重要）得有可能匹配的人脸
                matches = face_recognition.compare_faces(
                    total_face_encoding, face_encoding, tolerance=tolerance)

                # （重要）虽然之前得到匹配度最高的，但是这样总会有一个被认为匹配，
                # 因此，用compare_faces再一次得到匹配结果，两者共同判断才能得到
                # 被认为是匹配到的人脸中的最接近的。
                if matches[best_match_index]:
                    name = total_image_name[best_match_index][0]
                else:
                    name = "未知"

                names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, names):
            # 之前将摄像头抓取的图片缩小到1/4，因此计算坐标的时候，需要放大4倍
            # 之前抓取到人脸的部分names被用来保存名字，因此这里也需要把names每一项取出来
            # 因为face_locations每次抓出来的人脸的顺序是不确定的，而没5次才匹配一次人脸，因此很可能导致位置和名字数组顺序不对
            # 因此之前对face_locations排序，以保证了names的顺序和face_locations的顺序保持一致
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # 获得比较靠中间的人脸的名称，并且用绿色标记
            global facename
            facename = "无"
            mainColor = (53, 67, 203)
            if abs(abs(left-right)/2+left - w/2) < abs(left-right)/2 and abs(abs(top-bottom)/2+top - h/2) < abs(top-bottom)/2:
                if name != "未知":
                    mainColor = (99, 180, 40)
                facename = name

            cv2.rectangle(frame, (left, top),
                          (right, bottom), mainColor, 2)
            # 画出一个带名字的标签，放在框下
            cv2.rectangle(frame, (left-2, bottom + 35), (right+2, bottom), mainColor,
                          cv2.FILLED)
            frame = cv2ImgAddText(frame, "%s" % (name), left+6,
                                  bottom+10, (255, 255, 255), 20)

        process_frame_index += 1
        # 显示结果图像
        cv2.imshow('face recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            isCapLoop = False
            break

    try:
        cap.release()
        cv2.destroyAllWindows()
    except:
        return


def wscli(port=6789):
    uri = "ws://localhost:%d" % port
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


def main(argv):

    port = 0
    width = 640
    height = 480
    model = "hog"
    staticdata = False
    capIndex = 0
    tolerance = 0.5
    number_of_times_to_upsample = 1

    try:
        opts, args = getopt.getopt(argv, "hW:H:m:sp:c:t:n:", [])
    except getopt.GetoptError:
        print('arg error')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('-W width, default 640')
            print('-H height, default 480')
            print('-m use cnn, default false')
            print('-s from static face data, default from img folder')
            print('-p port to start websocket, default not start ws')
            print('-c camera index, default 0')
            print('-t tolerance, default 0.5')
            print('-n number_of_times_to_upsample, default 1')
            sys.exit()
        elif opt in ("-W"):
            width = int(arg)
        elif opt in ("-H"):
            height = int(arg)
        elif opt in ("-m"):
            model = arg
        elif opt in ("-s"):
            staticdata = True
        elif opt in ("-p"):
            port = int(arg)
        elif opt in ("-c"):
            capIndex = int(arg)
        elif opt in ("-t"):
            tolerance = float(arg)
        elif opt in ("-n"):
            number_of_times_to_upsample = int(arg)

    if port > 0:
        t = threading.Thread(target=wscli, args=[port], name='wscli')
        t.start()

    patchImages(staticdata)

    faceloop(capIndex=capIndex, Width=width, Height=height,
             model=model, tolerance=tolerance, number_of_times_to_upsample=number_of_times_to_upsample)


if __name__ == '__main__':

    main(sys.argv[1:])
