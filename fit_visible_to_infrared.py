import argparse

import cv2
import torch
import numpy as np
from scipy import signal

from siamfc import TrackerSiamFC

# https://github.com/NklausMikealson/SiameseFC-pytorch

# #  呼吸率对应的频率:0.15-0.40Hz，https://www.nature.com/articles/s41598-019-53808-9
RR_Min_HZ = 0.15
RR_Max_HZ = 0.70
# 采样频率
FPS = 25

def parse_args():
    """
    args for testing.
    """
    parser = argparse.ArgumentParser(
        description='PyTorch SiamFC Tracking Demo')
    parser.add_argument('--video', dest='video',
                        default='./test.mp4', help='video path')
    parser.add_argument('--model', dest='model',
                        default='pretrained/siamfc/model.pth', help='pretrained model')
    args = parser.parse_args()

    return args


def _x1y1wh_to_xyxy(bbox_x1y1wh):
    x1, y1, w, h = bbox_x1y1wh
    x2 = int(x1+w)
    y2 = int(y1+h)
    return x1, y1, x2, y2

def readvideo_infrared(datapath_infrared):
    vc = cv2.VideoCapture(datapath_infrared)  # 读取视频文件
    c = 1
    count_imgs_num = 0
    videonpy = []
    timeF_infrared = 1
    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        rval, frame = vc.read()
        if (c % timeF_infrared == 0):  # 每隔timeF帧加到数组内;  # timeF 视频帧计数间隔频率
            if frame is not None:
                videonpy.append(frame)
                count_imgs_num = count_imgs_num + 1
        c = c + 1
        if c >= 500:
            break
        cv2.waitKey(1)
    vc.release()
    videonpy = np.array(videonpy)
    return videonpy

# 计算img_ROI转化成灰度图之后的平均像素值
def get_avg_gray_pixel(img_ROI):
    gray_img = cv2.cvtColor(img_ROI, cv2.COLOR_BGR2GRAY)
    avg_pixel = np.mean(gray_img)
    # cv2.imshow('gray_img', gray_img)
    return avg_pixel

def infrared_preprocessing(signals):
    # ppg = np.array(signals)
    # df = pd.DataFrame(ppg)
    # signals_pre = df.interpolate()
    # 去趋势
    detrend_signals = detrend(signals, 100)
    # detrend_signals = detrend(signals_pre, 100)
    detrend_signals = detrend_signals.flatten()
    # 标准化
    normalized_signals = normalize(detrend_signals)
    # 滤波
    filtered_signals = filter_signal_infrared(normalized_signals)
    return filtered_signals
def normalize(signals):
    # print('func:normalize>>{},type:{}'.format(signals,type(signals)))
    # signals 空值或者zero值，分母计算会提示 warning，
    # 虽然已经在文件头添加忽略,但建议判断最佳
    normalized_signals = (signals - np.mean(signals)) / np.std(signals, ddof=1)
    # normalized_signals = normalized_signals = (signals - np.min(signals)) / (np.max(signals)-np.min(signals))
    return normalized_signals

def detrend(signals, param_lambda):
    # https://blog.csdn.net/piaoxuezhong/article/details/79211586
    signal_len = len(signals)
    I = np.identity(signal_len)
    B = np.array([1, -2, 1])

    # 当 signal_len - 2<0情况，出现负值问题未判断！ 例如signals:[],这部分不懂判断后处理方式
    ones = np.ones((signal_len - 2, 1))
    multi = B * ones
    D2 = np.zeros((signal_len - 2, signal_len))
    for i in range(D2.shape[0]):
        D2[i, i:i + 3] = multi[i]
    tr_D2 = np.transpose(D2)
    multi_D2 = np.dot(tr_D2, D2)
    inverse = I - (np.linalg.inv(I + (multi_D2 * pow(param_lambda, 2))))
    detrend_signals = np.dot(inverse, signals)
    return detrend_signals


def filter_signal_infrared(signals):
    filtered_signals = butterworth_filter(signals, RR_Min_HZ, RR_Max_HZ, FPS, order=5)
    return filtered_signals

def butterworth_filter(data, low, high, sample_rate, order=5):
    nyquist_rate = sample_rate * 0.5
    low /= nyquist_rate
    high /= nyquist_rate
    b, a = signal.butter(N=order, Wn=[low, high], btype='bandpass')
    return signal.lfilter(b, a, data)

def rr_extraction(PPG_values):
    # 傅里叶变换
    fft = np.abs(np.fft.rfft(PPG_values))
    buffer_size = len(PPG_values)
    # 当 buffer_size==0 的问题需要判断
    if buffer_size == 0: #3-19
        rr_value= 0
    else:
        freqs = FPS / buffer_size * np.arange(buffer_size / 2 + 1)
        # 找到在正常呼吸率范围内的频率最高值
        while True:
            max_idx = fft.argmax()  # 寻找数组的最大索引值
            bps = freqs[max_idx]
            if bps < RR_Min_HZ or bps > RR_Max_HZ:
                fft[max_idx] = 0
            else:
                rr_value = bps * 60.0
                break
        # print('rr:',rr_value)
    return rr_value
def first_frame_noseROI_infrared(min_x, min_y,max_x,max_y):
    # 仿射变换矩阵（拟合得到）https://blog.csdn.net/liuweiyuxiang/article/details/82799999
    matrix_affine_trans =  np.array([[0.803271838597598,0.00363813872899273,-354.141567587306],
                                     [0.0158988245484757,0.791743191445523,-192.690752783920],
                                     [0,0,1]])   # main_visible_to_infrared.m
    x_min = min_x   # 可见光鼻子左上角点的位置
    y_min = min_y   # 可见光鼻子左上角点的位置
    x_max = max_x   # 可见光鼻子右下角点的位置
    y_max = max_y   # 可见光鼻子右下角点的位置
    AA = np.array([[x_min], [y_min], [1]])  # 构造变换前的一个向量
    CC = np.array([[x_max], [y_max], [1]])  # 构造变换前的一个向量
    pos_xy_min_infrared = np.dot(matrix_affine_trans, AA)  # 求解仿射变换之后的一个向量：包含热红外鼻子的xmin,ymin
    pos_xy_max_infrared = np.dot(matrix_affine_trans, CC)  # 求解仿射变换之后的一个向量：包含热红外鼻子的xmax,ymax
    # 求热红外鼻子ROI的框，返回box，作为第一帧热红外鼻子的box
    ymin_nose_infrared = int(pos_xy_min_infrared[1])
    ymax_nose_infrared = int(pos_xy_max_infrared[1])
    xmin_nose_infrared = int(pos_xy_min_infrared[0])
    xmax_nose_infrared = int(pos_xy_max_infrared[0])
    # w = ymax_nose_infrared - ymin_nose_infrared
    # h = xmax_nose_infrared - xmin_nose_infrared
    # box = [xmin_nose_infrared, ymin_nose_infrared, w, h]
    box = [xmin_nose_infrared, ymin_nose_infrared, xmax_nose_infrared, ymax_nose_infrared]
    return box

def main(args):
    # cap = cv2.VideoCapture(args.video)
    # i = 0
    ppg_infrared_nose = []
    datapath_infrared = 'infrared.MP4'  # [280, 266, 73, 66]
    infrared_img_arr = readvideo_infrared(datapath_infrared)
    datapath_visible = 'visible.MP4'   # [783, 578, 94, 56]    [783, 578, 73, 66]
    visible_img_arr = readvideo_infrared(datapath_visible)
    print('len(infrared_img_arr):',len(infrared_img_arr))
    total_num = len(infrared_img_arr)
    # total_num = 100
    visible_xy_arr = []
    infrared_xy_arr = []
    for i in range(total_num):
        img = visible_img_arr[i]
        img1 = infrared_img_arr[i]
        if i == 0:
            # init the target
            # ROI Selection
            # cv2.namedWindow("SiamFC", cv2.WND_PROP_FULLSCREEN)
            # try:
            #     init_rect = cv2.selectROI('SiamFC', img, False, False)
            #     x, y, w, h = init_rect
            # except:
            #     exit()
            # init_state = [x, y, w, h]
            init_state =  [769, 548, 92, 53]

            # 可以用mediapipe来定位第一帧位置，人脸地标点参考https://developers.google.com/mediapipe/solutions/vision/face_landmarker/

            print('init_state =',init_state)
            trk = TrackerSiamFC(net_path=args.model)
            trk.init(img, init_state)
            i += 1
        else:
            # track the target
            A = []
            pos = trk.update(img)
            pos = _x1y1wh_to_xyxy(pos)
            pos = [int(l) for l in pos]
            A.append(pos[0])
            A.append(pos[1])
            A.append(1)
            infrared_xy_arr.append(A)
            ##
            A = []
            A.append(pos[2])
            A.append(pos[3])
            A.append(1)
            infrared_xy_arr.append(A)

            # 转灰度图
            img_infrared = img1
            # roi_infrared_nose = img_infrared[pos[1]:pos[3], pos[0]:pos[2]]
            # cv2.imshow('roi_infrared_nose', roi_infrared_nose)
            print('img_infrared.shape',img_infrared.shape)
            print('img.shape', img.shape)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.rectangle(img, (pos[0], pos[1]),
                          (pos[2], pos[3]), (0, 255, 255), 3)
            # print('pos:',pos)
            # BB = np.array([[0.803271838597598,0.00363813872899273,-354.141567587306],[0.0158988245484757,0.791743191445523,-192.690752783920],[0,0,1]])
            # x_min =pos[0]
            # y_min =pos[1]
            # x_max = pos[2]
            # y_max =pos[3]
            # AA = np.array([[x_min],[y_min],[1]])
            # print('AA.shape',AA.shape)
            # CC = np.array([[x_max],[y_max],[1]])
            # print('CC.shape', CC.shape)
            # print('BB.shape', BB.shape)
            # pos_xy_min_infrared = np.dot(BB,AA)
            # pos_xy_max_infrared = np.dot(BB,CC)
            # print('pos_xy_min_infrared',pos_xy_min_infrared)
            # print('pos_xy_max_infrared',pos_xy_max_infrared)
            # print('pos_xy_min_infrared[0]',pos_xy_min_infrared[0])
            # print('pos_xy_min_infrared[1]', pos_xy_min_infrared[1])

            x_min =pos[0]
            y_min =pos[1]
            x_max = pos[2]
            y_max =pos[3]
            box = first_frame_noseROI_infrared(x_min, y_min, x_max, y_max)
            infrared = img1[box[1]:box[3], box[0]:box[2]]


            # infrared =img1[int(pos_xy_min_infrared[1]):int(pos_xy_max_infrared[1]),int(pos_xy_min_infrared[0]):int(pos_xy_max_infrared[0])]
            # cv2.namedWindow("SiamFC1", cv2.WND_PROP_FULLSCREEN)
            cv2.imshow("SiamFC1", infrared)
            cv2.waitKey(1)

            # imshow
            cv2.namedWindow("SiamFC", cv2.WND_PROP_FULLSCREEN)
            cv2.imshow("SiamFC", img[:, :, (2, 1, 0)])
            # cv2.waitKey(25)
            cv2.waitKey(1)
            i += 1
            print('i:', i)

    #         #  热红外鼻子信号
    #         signal_infrared_nose = get_avg_gray_pixel(roi_infrared_nose)
    #         ppg_infrared_nose.append(signal_infrared_nose)
    #
    # print('ppg_infrared_nose:', ppg_infrared_nose)
    # print('len(ppg_infrared_nose):', len(ppg_infrared_nose))
    #
    # PPG_nose = infrared_preprocessing(ppg_infrared_nose)
    # print('len(PPG_nose):', len(PPG_nose))
    # # 计算呼吸率
    # rr = rr_extraction(PPG_nose)
    # print('实时呼吸率是{0}'.format(rr))
    #
    # np.savetxt('infrared_xy_arr.csv', infrared_xy_arr, delimiter=',')  # 存储信号


if __name__ == "__main__":
    args = parse_args()
    main(args)
