# 用于计算图像的特征信息

import cv2, xlwt, os
import numpy as np
import math
from tqdm import tqdm

# 计算灰度直方图
def calchist(img, save_excel=True, excel_name=None):
    if len(img.shape)==2:  # 单通道
        hist = cv2.calcHist(img,[0], None, [256], [0, 256])
        if save_excel:
            xls = xlwt.Workbook()
            sheet = xls.add_sheet('pixel')
            for i in range(len(hist)):
                sheet.write(i, 0, i)
                sheet.write(i, 1, int(hist[i]))
            xls.save(excel_name)

    else:
        hist = []
        for c in range(3):
            histr = cv2.calcHist([img], [c], None, [256], [0, 256])
            hist.append(histr)
        if save_excel:
            xls = xlwt.Workbook()
            sheet = xls.add_sheet('pixel')
            for i in range(len(hist[0])):
                sheet.write(i, 0, i)
                for c in range(3):
                    sheet.write(i, c+1, int(hist[c][i]))
            xls.save(excel_name)

    return hist

# 峰值信噪比
def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# 信息熵
def entropy(labels, base=None):
    value,counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    base = math.exp(1) if base is None else base
    return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()#log(a) b=log (c) b÷log (c) a

# Brenner 梯度函数
def brenner(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-2):
        for y in range(0, shape[1]):
            out += (int(img[x + 2, y]) - int(img[x, y])) ** 2
    return out

# Laplacian梯度函数
def Laplacian(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    return cv2.Laplacian(img,cv2.CV_64F).var()

# SMD（灰度方差）
def SMD(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(1, shape[1]):
            out += math.fabs(int(img[x, y]) - int(img[x, y - 1]))
            out += math.fabs(int(img[x, y] - int(img[x + 1, y])))
    return out

# SMD2（灰度方差乘积）
def SMD2(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out+=math.fabs(int(img[x,y])-int(img[x+1,y]))*math.fabs(int(img[x,y]-int(img[x,y+1])))
    return out

# 方差函数
def variance(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    out = 0
    u = np.mean(img)
    shape = np.shape(img)
    for x in range(0,shape[0]):
        for y in range(0,shape[1]):
            out+=(img[x,y]-u)**2
    return out

# 能量梯度函数
def energy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out+=((int(img[x+1,y])-int(img[x,y]))**2)+((int(img[x,y+1]-int(img[x,y])))**2)
    return out

# Vollath函数
def Vollath(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    shape = np.shape(img)
    u = np.mean(img)
    out = -shape[0]*shape[1]*(u**2)
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]):
            out+=int(img[x,y])*int(img[x+1,y])
    return out

# 信噪比
def SNR(a, axis=None, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

# 对比度
def contrast(img):
    m, n = img.shape
    #图片矩阵向外扩展一个像素
    img1_ext = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REPLICATE) / 1.0   # 除以1.0的目的是uint8转为float型，便于后续计算
    rows_ext,cols_ext = img1_ext.shape
    b = 0.0
    for i in range(1,rows_ext-1):
        for j in range(1,cols_ext-1):
            b += ((img1_ext[i,j]-img1_ext[i,j+1])**2 + (img1_ext[i,j]-img1_ext[i,j-1])**2 +
                    (img1_ext[i,j]-img1_ext[i+1,j])**2 + (img1_ext[i,j]-img1_ext[i-1,j])**2)

    cg = b/(4*(m-2)*(n-2)+3*(2*(m-2)+2*(n-2))+2*4) #对应上面48的计算公式
    return cg


if __name__ == '__main__':

    """ 计算烟雾图像特征 """
    import time
    image_path = "./yw3cl-DCP-ALN4/mix/"
    xls = xlwt.Workbook()
    sheet = xls.add_sheet('metric')
    metric = ['id', "time", 'mean', 'std', 'snr', 'entropy', 'energy', 'contrast']
    for mi, mn in enumerate(metric):
        sheet.write(0, mi, mn)
    # img_id = 1
    t0 = 0

    for name in tqdm(os.listdir(image_path)):
        # img_id = (int(name.split('-')[4]) - 34) * 60 + int(name.split('-')[5]) - 13
        img_id = int(name.split(')')[0][2:])
        img_name = name.split('.')[0]

        mtime = time.ctime(os.path.getmtime(image_path+name))
        mtimeH = mtime.split(' ')[3]
        m = int(mtimeH.split(":")[1])
        s = int(mtimeH.split(":")[2])

        t = m * 60 + s
        if img_id == 1:
            t0 = t

        image = cv2.imread(image_path+name, 0)
        image = cv2.resize(image, (512, 512))

        # 计算灰度直方图
        # excel_name = 'gray_pixel{}.xls'.format(name.split('.')[0])
        # hist = calchist(image, True, excel_name)

        # 计算方差、标准差
        (mean, stddv) = cv2.meanStdDev(image)
        snr = SNR(image)
        entr = entropy(image)
        ener = energy(image)
        con = contrast(image)

        sheet.write(img_id, 0, t-t0)
        sheet.write(img_id, 1, img_name)
        sheet.write(img_id, 2, mean[0][0])
        sheet.write(img_id, 3, stddv[0][0])
        sheet.write(img_id, 4, float(snr))
        sheet.write(img_id, 5, entr)
        sheet.write(img_id, 6, ener)
        sheet.write(img_id, 7, con)

        # img_id +=1

    xls.save('img_inf.xls')

    """ 煤矸图像特征分析 """
    # def creat_sheet(xls, metrics, cls):
    #     sheets = []
    #     for mi in metrics:
    #         sheet = xls.add_sheet(mi)
    #         for ci, cl in enumerate(cls):
    #             sheet.write(0, ci + 1, cl)
    #         sheets.append(sheet)
    #     return sheets
    #
    # path = "G:/paper/temp/feature_extract/"
    # cls = ['c1', 'c2', 'g1', 'g2']
    # metric = ['mean', 'std', 'snr', 'entropy', 'energy', 'contrast']
    # xls = xlwt.Workbook()
    # sheets = creat_sheet(xls, metric, cls)
    # for ci, cl in enumerate(cls):
    #     img_path = path + '{}/'.format(cl)
    #     for name in os.listdir(img_path):
    #         img_id = name.split('(')[1].split(')')[0]
    #         img_id = int(img_id)
    #         img = cv2.imread(img_path + name, cv2.IMREAD_GRAYSCALE)
    #
    #         (mean, stddv) = cv2.meanStdDev(img)
    #         snr = SNR(img)
    #         entr = entropy(img)
    #         ener = energy(img)
    #         con = contrast(img)
    #
    #         if ci == 0:
    #             for mi, me in enumerate(metric):
    #                 sheets[mi].write(img_id, 0, img_id)
    #
    #         sheets[0].write(img_id, ci + 1, mean[0][0])
    #         sheets[1].write(img_id, ci + 1, stddv[0][0])
    #         sheets[2].write(img_id, ci + 1, float(snr))
    #         sheets[3].write(img_id, ci + 1, entr)
    #         sheets[4].write(img_id, ci + 1, ener)
    #         sheets[5].write(img_id, ci + 1, con)
    #
    #         xls.save('CGmetric1-2.xls')

    """ 皮带速度和曝光时间批量计算 """

    # def creat_sheet(xls, metrics, cls, sds):
    #     sheets = []
    #     for mi in metrics:
    #         sheet = xls.add_sheet(mi)
    #         for sdi, sd in enumerate(sds):
    #             sheet.write(0, sdi * cll + 1, sd)
    #             for ci, cl in enumerate(cls):
    #                 sheet.write(1, sdi * cll + ci + 1, cl)
    #
    #         sheets.append(sheet)
    #     return sheets
    #
    #
    # path = "D:/231122/sd/crop/"
    # cls = ["jm", "hm", "bg", "gg"]
    # sds = ['0', '0.4', '0.6', '0.8', '1.0', '1.2']
    # cll = len(cls)
    # ts = ['1000', '1500', '2000', '3000', '4000', '5000']
    # metric = ['mean', 'std', 'snr', 'entropy', 'energy', 'contrast']
    # xls = xlwt.Workbook()
    # sheets = creat_sheet(xls, metric, cls, sds)
    # for sdi, sdn in enumerate(sds):
    #     img_path = path + '{}c/'.format(sdn)
    #     if sdi == 0:
    #         for mi, me in enumerate(metric):
    #             for ti, tn in enumerate(ts):
    #                 sheets[mi].write(ti + 2, 0, tn)
    #
    #     for name in os.listdir(img_path):
    #         img_t = name.split('-')[1]
    #         img_ti = ts.index(img_t)
    #         img_c = name.split('-')[-1][0:2]
    #         img_ci = cls.index(img_c)
    #
    #         img = cv2.imread(img_path + name, cv2.IMREAD_GRAYSCALE)
    #
    #         (mean, stddv) = cv2.meanStdDev(img)
    #         snr = SNR(img)
    #         entr = entropy(img)
    #         ener = energy(img)
    #         con = contrast(img)
    #
    #         sheets[0].write(img_ti + 2, sdi * cll + img_ci + 1, mean[0][0])
    #         sheets[1].write(img_ti + 2, sdi * cll + img_ci + 1, stddv[0][0])
    #         sheets[2].write(img_ti + 2, sdi * cll + img_ci + 1, float(snr))
    #         sheets[3].write(img_ti + 2, sdi * cll + img_ci + 1, entr)
    #         sheets[4].write(img_ti + 2, sdi * cll + img_ci + 1, ener)
    #         sheets[5].write(img_ti + 2, sdi * cll + img_ci + 1, con)
    #
    #         xls.save('CGmetric_sd.xls')

    """ 光照和曝光时间批量计算(目标块) """

    # def creat_sheet(xls, metrics, cls, ts):
    #     sheets = []
    #     for mi in metrics:
    #         sheet = xls.add_sheet(mi)
    #         for ti, tn in enumerate(ts):
    #             sheet.write(0, ti * cll + 1, tn)
    #             for ci, cl in enumerate(cls):
    #                 sheet.write(1, ti * cll + ci + 1, cl)
    #
    #         sheets.append(sheet)
    #     return sheets
    #
    #
    # ts = ['1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900'
    #     , '2000', '2100', '2200', '2300', '2400', '2500', '2600', '2700', '2800', '2900'
    #     , '3000', '3500', '4000', '4500', '5000']
    # gzs = ["l2000", "l2500", "l3000", "l3500"]
    # metric = ['mean', 'std', 'snr', 'entropy', 'energy', 'contrast']
    # cls = ["jm", "hm", "bg", "gg"]
    #
    # # 测试程序用
    # # ts = ['1000', '1100']
    # # gzs = ["l2000", "l2500"]
    # # metric = ['coarseness', 'contrast', 'directionality']
    # # cls = ["bg", "gg"]
    #
    # cll = len(cls) + 1
    #
    # # path = "D:/231122/gz/crop/"
    #
    # for gz in gzs:
    #     xls = xlwt.Workbook()
    #     sheets = creat_sheet(xls, metric, cls, ts)
    #     path = "D:/231122/gz/crop/{}/".format(gz)
    #
    #     for cli, cln in enumerate(cls):
    #         file_path = path + cln + '/'
    #
    #         for name in os.listdir(file_path):
    #             img_id = int(name.split('-')[1][2:])
    #             img_ts = name.split('-')[-1].split('.')[0]
    #             img_ti = ts.index(img_ts)
    #
    #             img = cv2.imread(file_path + name, cv2.IMREAD_GRAYSCALE)
    #
    #             (mean, stddv) = cv2.meanStdDev(img)
    #             snr = SNR(img)
    #             entr = entropy(img)
    #             ener = energy(img)
    #             con = contrast(img)
    #
    #             sheets[0].write(img_id + 1, img_ti * cll + cli + 1, mean[0][0])
    #             sheets[1].write(img_id + 1, img_ti * cll + cli + 1, stddv[0][0])
    #             sheets[2].write(img_id + 1, img_ti * cll + cli + 1, float(snr))
    #             sheets[3].write(img_id + 1, img_ti * cll + cli + 1, entr)
    #             sheets[4].write(img_id + 1, img_ti * cll + cli + 1, ener)
    #             sheets[5].write(img_id + 1, img_ti * cll + cli + 1, con)
    #
    #             if cli == 0 and img_ti == 0:
    #                 for mi, me in enumerate(metric):
    #                     sheets[mi].write(img_id + 1, 0, img_id)
    #
    #         xls.save('CGmetric_gz_{}.xls'.format(gz))

    """ 光照和曝光时间批量计算(整幅图像) """
    # def creat_sheet(xls, metrics, cls, ts):
    #     sheets = []
    #     for mi in metrics:
    #         sheet = xls.add_sheet(mi)
    #         for ti, tn in enumerate(ts):
    #             sheet.write(0, ti + 1, tn)
    #         for ci, cl in enumerate(cls):
    #             sheet.write(ci + 1, 0, cl)
    #
    #         sheets.append(sheet)
    #     return sheets
    #
    #
    # ts = ['1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900'
    #     , '2000', '2100', '2200', '2300', '2400', '2500', '2600', '2700', '2800', '2900'
    #     , '3000', '3500', '4000', '4500', '5000']
    # gzs = ["l2000", "l2500", "l3000", "l3500"]
    # metric = ['mean', 'std', 'snr', 'entropy', 'energy', 'contrast']
    # cls = ["jm", "hm", "bg", "gg"]
    #
    #
    # cll = len(cls) + 1
    #
    # # path = "D:/231122/gz/crop/"
    #
    # for gz in gzs:
    #     xls = xlwt.Workbook()
    #     sheets = creat_sheet(xls, metric, cls, ts)
    #     path = "D:/231122/gz/{}/".format(gz)
    #
    #     for name in os.listdir(path):
    #
    #         img_cl = name[:2]
    #         img_cli = cls.index(img_cl)
    #         img_ts = name[2:6]
    #         img_ti = ts.index(img_ts)
    #
    #         img = cv2.imread(path + name, cv2.IMREAD_GRAYSCALE)
    #
    #         (mean, stddv) = cv2.meanStdDev(img)
    #         snr = SNR(img)
    #         entr = entropy(img)
    #         ener = energy(img)
    #         con = contrast(img)
    #
    #         sheets[0].write(img_cli + 1, img_ti + 1, mean[0][0])
    #         sheets[1].write(img_cli + 1, img_ti + 1, stddv[0][0])
    #         sheets[2].write(img_cli + 1, img_ti + 1, float(snr))
    #         sheets[3].write(img_cli + 1, img_ti + 1, entr)
    #         sheets[4].write(img_cli + 1, img_ti + 1, ener)
    #         sheets[5].write(img_cli + 1, img_ti + 1, con)
    #
    #         xls.save('CGmetric_gzimage_{}.xls'.format(gz))
