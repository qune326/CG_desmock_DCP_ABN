import cv2, os, xlwt
import numpy as np
from tqdm import tqdm

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

""" 图像颜色空间分析 """
# cls = ["JPEGImages"]
# path = "F:/data/VOCdevkit/VOC2007L300/"
# # cls = ["stack2", "mz340"]
# # path = "F:\data\VOCdevkit"
# xls = xlwt.Workbook()
# metric = ['id', 'B', "G", 'R', 'H', "S", 'V', 'H', 'L', 'S']
#
# for cl in cls:
#     sheet = xls.add_sheet(cl)
#     for mi, mn in enumerate(metric):
#         sheet.write(0, mi, mn)
#     # img_path = "%s/VOC2007%s/JPEGImages/"%(path, cl)
#     img_path = "%s/%s/" % (path, cl)
#     # img_id = 1
#     filenames = os.listdir(img_path)
#     for ni in tqdm(range(len(filenames))):
#         name = filenames[ni]
#         img = cv2.imread(img_path+name)
#         # img_id = int(name.split(')')[0][4:])
#         img_name = name.split(".")[0]
#
#         b = np.mean(np.array(img[:, :, 0]) / 255.)
#         g = np.mean(np.array(img[:, :, 1]) / 255.)
#         r = np.mean(np.array(img[:, :, 2]) / 255.)
#
#         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
#         h = np.mean(np.array(hsv[:, :, 0]) / 255.)
#         s = np.mean(np.array(hsv[:, :, 1]) / 255.)
#         v = np.mean(np.array(hsv[:, :, 2]) / 255.)
#
#         hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
#         hh = np.mean(np.array(hls[:, :, 0]) / 180.)
#         ll = np.mean(np.array(hls[:, :, 1]) / 255.)
#         ss = np.mean(np.array(hls[:, :, 2]) / 255.)
#
#         sheet.write(ni+1, 0, img_name)
#         sheet.write(ni+1, 1, b)
#         sheet.write(ni+1, 2, g)
#         sheet.write(ni+1, 3, r)
#         sheet.write(ni+1, 4, h)
#         sheet.write(ni+1, 5, s)
#         sheet.write(ni+1, 6, v)
#         sheet.write(ni+1, 7, hh)
#         sheet.write(ni+1, 8, ll)
#         sheet.write(ni+1, 9, ss)
#
#         xls.save("I:/230711/230711mb/pinjie/de_smock_color_anly.xls")

""" 基于直方图分析 """
# cls = ["c1", "c2", "c3", "g1", "g2", "g3", "m1", "m2", "m3"]
# path = "D:/WLY/PycharmProjects/PAPER/ywcl-ATD"
cls = ["JPEGImages"]
path = "F:/data/VOCdevkit/VOC2007L300/"
xls = xlwt.Workbook()
metric = ['id', 'Bp', "Bw"]

for cl in cls:
    sheet = xls.add_sheet(cl)
    for mi, mn in enumerate(metric):
        sheet.write(0, mi, mn)
    # img_path = "%s/%s/"%(path, cl)
    img_path = "%s%s/" % (path, cl)
    img_id = 1
    for name in tqdm(os.listdir(img_path)):
        img = cv2.imread(img_path+name, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (512, 512))
        # img_id = int(name.split(')')[0][4:])
        img_name = name.split(".")[0]

        # b = img[:, :, 0]

        bp, bw = CalaNormalParam(img, low_clip=0.1, high_clip=0.9)

        sheet.write(img_id, 0, img_name)
        sheet.write(img_id, 1, int(bp))
        sheet.write(img_id, 2, int(bw))

        img_id += 1

        xls.save("I:/230711/230711mb/pinjie/no_smock_bwp.xls")