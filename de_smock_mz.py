import os, errno
import cv2
import numpy as np

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def zmMinFilterGray(src, r=7):
    '''最小值滤波，r是滤波器半径'''
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))


def guidedfilter(I, p, r, eps):
    # height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def Defog(m, r, eps, w, maxV1):                 # 输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m, axis=2)      # 得到暗通道图像

    Dark_Channel = zmMinFilterGray(V1, 7)

    V1 = guidedfilter(V1, Dark_Channel, r, eps)  # 使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)                  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
    V1 = np.minimum(V1 * w, maxV1)               # 对值范围进行限制
    return V1, A, Dark_Channel

def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    Mask_img, A, Dark_Channel = Defog(m, r, eps, w, maxV1)   # 得到遮罩图像和大气光照

    for k in range(3):
        Y[:,:,k] = (m[:,:,k] - Mask_img)/(1-Mask_img/A)  # 颜色校正
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))       # gamma校正,默认不进行该操作
    return Y, Dark_Channel


def CalaNormalParam(img, low_clip = 0.1, high_clip = 0.9):
    totel = img.shape[0] * img.shape[1]
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.squeeze()
    max_value = np.where(hist == np.max(hist))
    current = 0
    low_val, high_val = 0, 0
    for i in range(len(hist)):
        current += hist[i]
        if float(current) / totel < low_clip:
            low_val = i
        if float(current) / totel < high_clip:
            high_val = i
    hist_w = high_val - low_val

    if len(max_value)==1:
        max_value = max_value[0][0]
    else:
        max_value = 0

    return max_value, hist_w

if __name__ == '__main__':

    img = cv2.imread("./test/coal.jpg")
    raw = cv2.resize(img, (512, 512))
    dehazed_image, Dark_Channel = deHaze(raw/255., w=1)
    dehazed_image = np.uint8(np.clip(dehazed_image*255., 0, 255))

    Dark_Channel = cv2.cvtColor(np.uint8(Dark_Channel*255.), cv2.COLOR_GRAY2BGR)

    b = dehazed_image[:, :, 0]
    bp, bw = CalaNormalParam(b, low_clip=0.1, high_clip=0.9)
    w = 34 / bw
    p = 27 - bp
    norm_img = np.uint8(dehazed_image * w + p)

    show = np.hstack((raw, Dark_Channel, dehazed_image, norm_img))
    cv2.namedWindow("dark_channel", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.imshow("dark_channel", show)

    cv2.waitKey(0)
    cv2.destroyAllWindows()  # 正常关闭所有绘制窗口

    """ 多张图像测试 """
    # from tqdm import tqdm
    # import time
    # from feature_anly import SNR
    # cls = ["gm"]
    # path = "I:/230711/230711mb/test"
    # t1 = time.time()
    #
    # for cl in cls:
    #     img_path = "%s/%s/"%(path, cl)
    #     save_path = "I:/230711/230711mb/DCP-ALN/%s/"%(cl)
    #     mkdir_p(save_path)
    #
    #     for name in tqdm(os.listdir(img_path)):
    #         raw = cv2.imread(img_path + name)
    #         # raw = cv2.resize(img, (512, 512))
    #         dehazed_image, _ = deHaze(raw / 255.)
    #         dehazed_image = np.uint8(dehazed_image * 255.)
    #
    #         # dehazed_image = cv2.cvtColor(dehazed_image, cv2.COLOR_BGR2GRAY)
    #         # dehazed_image = cv2.cvtColor(dehazed_image, cv2.COLOR_GRAY2BGR)
    #
    #         # b = dehazed_image[:, :, 0]
    #         b = np.uint8(np.clip(np.mean(dehazed_image, axis=2), 0, 255.))
    #         bp, bw = CalaNormalParam(b, low_clip=0.1, high_clip=0.9)
    #         w, p = 1, 0
    #         # if bw != 0:
    #         #     w = 29 / bw
    #         # if bp != 0:
    #         #     p = 30 - bp
    #
    #         """ 线阵 w24,p63 面阵 w34,p27"""
    #         if bw != 0:
    #             w = 24 / bw
    #         if bp != 0:
    #             p = 63 - bp
    #
    #         norm_img = np.uint8(dehazed_image * w + p)
    #         cv2.imwrite(save_path+name, norm_img)

        # """ 加上了灰度值和snr阈值判定 """
        # for name in tqdm(os.listdir(img_path)):
        #     raw = cv2.imread(img_path + name)
        #     # raw = cv2.resize(img, (512, 512))
        #     (meant, stddv) = cv2.meanStdDev(raw)
        #     snrt = SNR(raw)
        #     if meant[0][0]>46 and meant[0][0]<160 and snrt>2.5 and snrt<8.0:
        #         dehazed_image, _ = deHaze(raw / 255.)
        #         dehazed_image = np.uint8(dehazed_image * 255.)
        #         b = np.uint8(np.clip(np.mean(dehazed_image, axis=2), 0, 255.))
        #         bp, bw = CalaNormalParam(b, low_clip=0.1, high_clip=0.9)
        #         w, p = 1, 0
        #
        #         if bw != 0:
        #             w = 34 / bw
        #         if bp != 0:
        #             p = 27 - bp
        #
        #         norm_img = np.uint8(dehazed_image * w + p)
        #         cv2.imwrite(save_path + name, norm_img)
        #         print(name)
        #     else:
        #         cv2.imwrite(save_path + name, raw)