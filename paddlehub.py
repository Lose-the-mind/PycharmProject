import paddlehub as hub
import cv2
import math
import random
import string
from PIL import Image, ImageDraw, ImageFont
import copy
import numpy as np

mask_detector = hub.Module(name="pyramidbox_lite_server_mask")
temp_mark = 0
length = 0
temp_sum = 0
glob_score = 0
self_score = 0
# # 定义字符串长度和数量
# str_len = 3
# str_num = 55
#
# # 生成随机字符串列表
# str_list = [''.join(random.sample(string.ascii_letters + string.digits, str_len)) for i in range(str_num)]

list_name = ["林劲松", '韦思琪', '官远航', '曾宏睿', '肖梓宏', '陈棋濠', '陈沛琳', '袁子杰', '李佳泽', '吴雨墨',
             '林嘉宇', '李艺嘉', '赖弘煊', '左晟宇', '刘飞', '许孟辰', '吴瑞晨', '刘雅涵', '邹玉婷', '廖子涵',
             '章子怡', '彭予棠', '芦芸', '陈彦泽', '张文成', '谭玉婷', '康悦函', '李懿轩', '龚垲锋', '李佳鑫',
             '彭志煊', '施美奂', '薛雅雯', '胡俊宇', '夏桂清', '叶璟琰', '郑晨曦', '郭凯泽', '施美轮', '邓浩宇',
             '郑阳洋', '吴少尧', '黄靖然', '苏奕凡', '赵钰彤', '杜林东', '韦思琪', '王珅瑜', '吴佳豪', '许睿浩'
                                                                                                       '许梓晨',
             '詹泽云', '苏奕凡', '官远航', '吴瑞晨', '官远航', '李诺依', '官远航', '郭铠泽', '芦芸',
             '许家诚', '薛雅雯', '桑统治', '范梓雯', '刘雅涵', '刘锦龙', '官远航', '刘飞', '官远航', '官远航',
             '官远航',
             '官远航', '袁子杰', '陈棋濠', '袁子杰', '官远航', '廖凌姗', '官远航', '官远航', '官远航', '官远航',
             '林嘉宇''蒋 艳', '陈世玲', '官远航', '颜资琪', '官远航', '施美奂', '许孟辰', '黄增鑫', '卢 芸', '李艺嘉',
             '官远航', '官远航', '官远航', '官远航', '官远航', '兰雨嘉', '李艺嘉', '陈子毅', '官远航', '官远航',
             '官远航', '陈怀刚', '官远航', '吴瑞晨', '沈佩圻', '李诺依', '王若熙', '郭铠泽', '芦芸', '许家诚',
             '薛雅雯', '桑统治', '官远航', '刘雅涵', '刘锦龙', '官远航', '康悦涵', '范俊文', '官远航', '官远航',
             '官远航', '廖子涵', '陈棋濠', '袁子杰', '邹 蒙', '廖凌姗', '黄玄烨', '蔡金余', '王静娴', '官远航',
             '林嘉宇''蒋 艳', '陈世玲', '章子怡', '颜资琪', '官远航', '施美奂', '吴宛儒', '黄增鑫', '卢 芸', '李艺嘉',
             '官远航', '官远航', '官远航', '官远航', '官远航', '兰雨嘉', '官远航', '陈子毅', '官远航', '林嘉宇',
             '詹泽云', '陈怀刚', '林亿森', '吴瑞晨', '沈佩圻', '李诺依', '王若熙', '郭铠泽', '赖韦铭', '官远航',
             '薛雅雯', '桑统治', '官远航', '刘雅涵', '刘锦龙', '陈棋濠', '陈棋濠', '范俊文', '陈棋濠', '官远航',
             '官远航', '官远航', '陈棋濠', '蔡麒乐', '邹 蒙', '廖凌姗', '官远航', '蔡金余', '王静娴', '林嘉宇',
             '兰昊益''蒋 艳', '陈世玲', '章子怡', '颜资琪', '官远航', '苏鸿圣', '吴宛儒', '黄增鑫', '陈彦泽', '许孟辰',
             '康致轩', '康悦涵', '官远航', '官远航', '官远航', '兰雨嘉', '陈柳丹', '陈子毅', '黄宗灿', '林嘉宇',
             '詹泽云', '陈怀刚', '林亿森', '吴瑞晨', '沈佩圻', '李诺依', '王若熙', '郭铠泽', '官远航', '许家诚',
             '薛雅雯', '桑统治', '辛添淇', '刘雅涵', '刘锦龙', '官远航', '潘栩若', '范俊文', '林嘉宇', '官远航',
             '袁子杰', '陈柏霖', '周 婷', '蔡麒乐', '邹 蒙', '廖凌姗', '黄玄烨', '蔡金余', '王静娴', '王媛媛',
             '兰昊益' '蒋 艳', '陈世玲', '章子怡', '颜资琪', '林泽楷', '苏鸿圣', '吴宛儒', '黄增鑫', '卢 芸', '楼雨馨',
             '康致轩', '康悦涵', '官远航', '连欣怡', '官远航', '兰雨嘉', '陈柳丹', '陈子毅', '黄宗灿', '官远航',
             '詹泽云', '陈怀刚', '林亿森', '吴瑞晨', '沈佩圻', '李诺依', '王若熙', '郭铠泽', '许孟辰', '许家诚',
             '薛雅雯', '桑统治', '辛添淇', '刘雅涵', '刘锦龙', '黄馨怡', '潘栩若', '范俊文', '丁诗诗', '荣伟宏',
             '王亿宸', '陈柏霖', '袁子杰', '蔡麒乐', '邹 蒙', '廖凌姗', '黄玄烨', '蔡金余', '王静娴', '王媛媛',
             '兰昊益' '蒋 艳', '陈世玲', '章子怡', '颜资琪', '林泽楷', '苏鸿圣', '吴宛儒', '黄增鑫', '卢 芸', '楼雨馨',
             '康致轩', '王芷筠', '王可晗', '连欣怡', '官远航', '兰雨嘉', '陈柳丹', '陈子毅', '黄宗灿', '李仁杰',
             '詹泽云', '陈怀刚', '林亿森', '吴瑞晨', '沈佩圻', '李诺依', '王若熙', '郭铠泽', '赖韦铭', '许家诚',
             '薛雅雯', '桑统治', '辛添淇', '刘雅涵', '刘锦龙', '黄馨怡', '潘栩若', '范俊文', '丁诗诗', '荣伟宏',
             '王亿宸', '陈柏霖', '周 婷', '蔡麒乐', '邹 蒙', '廖凌姗', '黄玄烨', '蔡金余', '王静娴', '王媛媛',
             '兰昊益' '蒋 艳', '陈世玲', '章子怡', '颜资琪', '林泽楷', '苏鸿圣', '吴宛儒', '黄增鑫', '卢 芸', '楼雨馨',
             '康致轩', '王芷筠', '王可晗', '连欣怡', '官远航', '兰雨嘉', '陈柳丹', '陈子毅', '黄宗灿', '李仁杰',
             '詹泽云', '陈怀刚', '林亿森', '吴瑞晨', '沈佩圻', '李诺依', '王若熙', '郭铠泽', '赖韦铭', '许家诚',
             '薛雅雯', '桑统治', '辛添淇', '刘雅涵', '刘锦龙', '黄馨怡', '潘栩若', '范俊文', '丁诗诗', '荣伟宏',
             '王亿宸', '陈柏霖', '周 婷', '蔡麒乐', '邹 蒙', '廖凌姗', '黄玄烨', '蔡金余', '王静娴', '王媛媛',
             '兰昊益' '蒋 艳', '陈世玲', '章子怡', '颜资琪', '林泽楷', '苏鸿圣', '吴宛儒', '黄增鑫', '卢 芸', '楼雨馨',
             '康致轩', '王芷筠', '王可晗', '连欣怡', '官远航', '兰雨嘉', '陈柳丹', '陈子毅', '黄宗灿', '李仁杰',
             '金 鑫']

name_ndarray = np.array(list_name)


# str_list = [chr(i) for i in range(97, 123)]
class Point1:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.score = 0


lis1 = []  # 结构体数组


class Point2:
    def __init__(self2):
        self2.xx = 0
        self2.yy = 0
        self2.zz = 0


lis2 = []  # 结构体数组


# 封装函数
def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=3):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        print("openCV success")
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def mask_detecion(img):
    global temp_sum
    temp_sum = temp_sum + 1
    input_dict = {"data": [img]}
    result = mask_detector.face_detection(data=input_dict)
    count = len(result[0]['data'])
    if count < 1:
        # print('There is no face detected!')
        pass
    else:
        # labels = []  # 存储每个人脸标签
        global glob_score
        global self_score
        global temp_mark
        global temp_count
        global length
        global Point1
        global Point2
        temp_conf = 0
        if temp_mark == 0:
            faces_coordinates = []
            for i in range(0, count):
                lis1.append(Point1())  # 添加一个结构体
                # print(result[0]['data'][i])
                # label = result[0]['data'][i].get('label')
                score = float(result[0]['data'][i].get('confidence'))
                print("置信率")
                print(score)
                if score > 0.8:
                    temp_conf = temp_conf + 1
                # if temp_sum % 60 == 0:
                x1 = int(result[0]['data'][i].get('left'))
                y1 = int(result[0]['data'][i].get('top'))
                x2 = int(result[0]['data'][i].get('right'))
                y2 = int(result[0]['data'][i].get('bottom'))
                # faces_coordinates.append([x1, y1, x2, y2])
                print("标记点的坐标\n")
                print(x1)
                print(y1)
                lis1[i].x = x1
                lis1[i].y = y1
                lis1[i].z = temp_count
                lis1[i].score = score
                label = label = name_ndarray[lis1[i].z]
                print(label)
                self_score = round(lis1[i].score * 100, 2)
                selfscore = str(self_score)
                print("每个人脸的置信率")
                print(selfscore)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
                cv2.putText(img, selfscore, (x1 + 45, y1), 0, 0.7, (0, 255, 200), 1)
                # img1 = cv2AddChineseText(img, label, (x1, y1), (0, 255, 0), 3)
                img = cv2AddChineseText(img, label, (x1, y1 - 15), (0, 255, 200), 10)
                temp_count = temp_count + 1
                temp_mark = temp_mark + 1
            # if temp_sum % 60 == 0:
            k = temp_conf / count
            glob_score = round(k * 100, 2)
            globscore = str(glob_score)
            cv2.putText(img, globscore, (30, 30), 0, 0.8, (0, 255, 200), 3)
            img = cv2AddChineseText(img, '课堂人数：51', (10, 60), (0, 255, 200), 25)
            img = cv2AddChineseText(img, '课堂气氛：', (10, 30), (0, 255, 200), 25)
            length = temp_count
            print("每一帧有几张人脸图像\n")
            print(count)
            print(temp_count)
        else:
            print("每一帧有几张人脸图像\n")
            print(count)
            list_len = length

            for i in range(0, count):
                # if i > length-1:
                #     lis1.append(Point1())
                # label = result[0]['data'][i].get('label')
                score = float(result[0]['data'][i].get('confidence'))
                print("置信率")
                print(score)
                if score > 0.8:
                    temp_conf = temp_conf + 1
                x1 = int(result[0]['data'][i].get('left'))
                y1 = int(result[0]['data'][i].get('top'))
                x2 = int(result[0]['data'][i].get('right'))
                y2 = int(result[0]['data'][i].get('bottom'))
                # faces_coordinates.append([x1, y1, x2, y2])
                print("标记点的坐标")
                print(x1)
                print(y1)
                min_dis = float('inf')
                num = 0
                for j in range(0, list_len - 1):
                    distance = math.sqrt((lis1[j].x - x1) ** 2 + (lis1[j].y - y1) ** 2)
                    if distance < min_dis:
                        min_dis = distance
                    if distance < 20:
                        # lis1[i].z = lis1[j].z
                        lis1[j].x = x1
                        lis1[j].y = y1
                        num = 1
                        b = lis1[j].z
                        if temp_sum % 60 == 0:
                            lis1[j].score = score
                        label = name_ndarray[b]
                        self_score = round(lis1[j].score * 100, 2)
                        print(label)
                        break
                print("本次检测坐标与上一帧坐标之间的距离最小为：")
                print(distance)
                print("\n")
                if num == 0:
                    lis1.append(Point1())  # 添加一个结构体
                    temp_count = temp_count + 1
                    lis1[length - 1].x = x1
                    lis1[length - 1].y = y1
                    lis1[length - 1].z = temp_count
                    lis1[length - 1].score = score
                    # label = str(lis1[length - 1].z)
                    label = name_ndarray[lis1[length - 1].z]
                    self_score = round(lis1[length - 1].score * 100, 2)
                    print("label failed")
                    print(label)
                # label = str(lis1[i].z)
                print("输出中文")
                # percentscore = '{:.2%}'.format(score)
                # sumscore = str(percentscore)
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
                selfscore = str(self_score)
                # self_score = round()
                cv2.putText(img, selfscore, (x1 + 45, y1), 0, 0.5, (0, 255, 0), 1)
                img = cv2AddChineseText(img, label, (x1, y1 - 15), (0, 255, 0), 15)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
                # cv2.putText(img, label, (x1, y1), 0, 0.6, (0, 255, 0), 2)
                print("输出边框")
            length = temp_count
            print("一共用了多少个名字")
            print(length)
            # k = round(temp_conf / count, 4)
            # percent = '{:.2%}'.format(k)
            # sum = str(percent)
            if temp_sum % 60 == 0:
                k = temp_conf / count
                glob_score = round(k * 100, 2)
            globscore = str(glob_score)
            # focus = 'Student concentration'
            # cv2.putText(img, focus, (10, 20), 0, 1, (0, 255, 200), 4)
            cv2.putText(img, globscore, (120, 50), 0, 0.8, (0, 255, 200), 1)
            img = cv2AddChineseText(img, '课堂人数：51 人', (10, 60), (0, 255, 200), 25)
            img = cv2AddChineseText(img, '课堂气氛：', (10, 30), (0, 255, 200), 25)
    return img


if __name__ == '__main__':
    cap = cv2.VideoCapture('D:\openCVProject\student30fps10min.mp4')  # 视频文件检测
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('outputstudent10mintest3.mp4', 0x7634706d, 30.0, (1920, 1080))
    temp_count = 1  # 出现的第几张图片
    length = 0
    if (cap.isOpened()):  # 视频打开成功
        while (True):
            ret, frame = cap.read()  # 读取一帧
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = mask_detecion(frame)
            print("***\n")
            cv2.imshow('mask_detection', result)
            out.write(result)
            if cv2.waitKey(1) & 0xFF == 27:  # 按下Esc键退出
                break
    else:
        print('open video failed!')

    cap.release()
    cv2.destroyAllWindows()
