import os

import cv2
import numpy as np
from skimage.feature import graycomatrix, greycoprops
import math
import xlwt

def coarseness(image, kmax):
	image = np.array(image)
	w = image.shape[0]
	h = image.shape[1]
	kmax = kmax if (np.power(2,kmax) < w) else int(np.log(w) / np.log(2))
	kmax = kmax if (np.power(2,kmax) < h) else int(np.log(h) / np.log(2))
	average_gray = np.zeros([kmax,w,h])
	horizon = np.zeros([kmax,w,h])
	vertical = np.zeros([kmax,w,h])
	Sbest = np.zeros([w,h])

	for k in range(kmax):
		window = np.power(2,k)
		for wi in range(w)[window:(w-window)]:
			for hi in range(h)[window:(h-window)]:
				average_gray[k][wi][hi] = np.sum(image[wi-window:wi+window, hi-window:hi+window])
		for wi in range(w)[window:(w-window-1)]:
			for hi in range(h)[window:(h-window-1)]:
				horizon[k][wi][hi] = average_gray[k][wi+window][hi] - average_gray[k][wi-window][hi]
				vertical[k][wi][hi] = average_gray[k][wi][hi+window] - average_gray[k][wi][hi-window]
		horizon[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))
		vertical[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))

	for wi in range(w):
		for hi in range(h):
			h_max = np.max(horizon[:,wi,hi])
			h_max_index = np.argmax(horizon[:,wi,hi])
			v_max = np.max(vertical[:,wi,hi])
			v_max_index = np.argmax(vertical[:,wi,hi])
			index = h_max_index if (h_max > v_max) else v_max_index
			Sbest[wi][hi] = np.power(2,index)

	fcrs = np.mean(Sbest)
	return fcrs


def contrast(image):
	image = np.array(image)
	image = np.reshape(image, (1, image.shape[0]*image.shape[1]))
	m4 = np.mean(np.power(image - np.mean(image),4))
	v = np.var(image)
	std = np.power(v, 0.5)
	alfa4 = m4 / np.power(v,2)
	fcon = std / np.power(alfa4, 0.25)
	return fcon

def directionality(image, n=16, t=12):
	"""

	:param image:
	:param n: theta 角的量级
	:param t: theta 角的统计阈值
	:return:
	"""
	image = np.array(image, dtype = 'int64')
	h = image.shape[0]
	w = image.shape[1]
	convH = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
	convV = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
	deltaH = np.zeros([h,w])
	deltaV = np.zeros([h,w])
	theta = np.zeros([h,w])

	# calc for deltaH
	for hi in range(h)[1:h-1]:
		for wi in range(w)[1:w-1]:
			deltaH[hi][wi] = np.sum(np.multiply(image[hi-1:hi+2, wi-1:wi+2], convH))
	for wi in range(w)[1:w-1]:
		deltaH[0][wi] = image[0][wi+1] - image[0][wi]
		deltaH[h-1][wi] = image[h-1][wi+1] - image[h-1][wi]
	for hi in range(h):
		deltaH[hi][0] = image[hi][1] - image[hi][0]
		deltaH[hi][w-1] = image[hi][w-1] - image[hi][w-2]

	# calc for deltaV
	for hi in range(h)[1:h-1]:
		for wi in range(w)[1:w-1]:
			deltaV[hi][wi] = np.sum(np.multiply(image[hi-1:hi+2, wi-1:wi+2], convV))
	for wi in range(w):
		deltaV[0][wi] = image[1][wi] - image[0][wi]
		deltaV[h-1][wi] = image[h-1][wi] - image[h-2][wi]
	for hi in range(h)[1:h-1]:
		deltaV[hi][0] = image[hi+1][0] - image[hi][0]
		deltaV[hi][w-1] = image[hi+1][w-1] - image[hi][w-1]

	deltaG = (np.absolute(deltaH) + np.absolute(deltaV)) / 2.0
	deltaG_vec = np.reshape(deltaG, (deltaG.shape[0] * deltaG.shape[1]))

	# calc the theta
	for hi in range(h):
		for wi in range(w):
			if (deltaH[hi][wi] == 0 and deltaV[hi][wi] == 0):
				theta[hi][wi] = 0
			elif(deltaH[hi][wi] == 0):
				theta[hi][wi] = np.pi
			else:
				theta[hi][wi] = np.arctan(deltaV[hi][wi] / deltaH[hi][wi]) + np.pi / 2.0
	theta_vec = np.reshape(theta, (theta.shape[0] * theta.shape[1]))
	# list_file.write(str(deltaG_vec))
	# list_file.write('\n')

	hd = np.zeros(n)
	dlen = deltaG_vec.shape[0]
	for ni in range(n):
		for k in range(dlen):
			if((deltaG_vec[k] >= t) and (theta_vec[k] >= (2*ni-1) * np.pi / (2 * n)) and (theta_vec[k] < (2*ni+1) * np.pi / (2 * n))):
				hd[ni] += 1
	# hd = hd / np.mean(hd)
	hd = hd / np.sum(hd)
	hd_max_index = np.argmax(hd)
	fdir = 0
	for ni in range(n):
		fdir += np.power((ni - hd_max_index), 2) * hd[ni]

	# hd = np.zeros(theta.shape[0] * theta.shape[1])
	# for k in range(dlen):
	# 	if deltaG_vec[k] >= t:
	# 		hd[k] = theta_vec[k]
	#
	# m = np.mean(hd)
	# fdir = np.mean(np.absolute(hd-m))
	return fdir

def linelikeness(image, dist, levels):
	image = np.array(image)
	image = image / (256. / levels)
	image = np.uint8(image)
	glcm = graycomatrix(image, [dist], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4, np.pi * 4 / 4,
									 np.pi * 5 / 4, np.pi * 6 / 4, np.pi * 7 / 4], levels, symmetric=True, normed=True)
	f = np.zeros(8)
	g = np.zeros(8)
	for i in range(levels):
		for j in range(levels):
			f += glcm[i,j,0,:] * math.cos((i-j)*2*math.pi/levels)
			g += glcm[i,j,0,:]
	# print(f, g)
	tempM = f/g
	Flin = np.max(tempM)
	return Flin


def regularity(image, filter):
	h, w = image.shape
	crs = []
	con = []
	dire = []
	lin = []

	for i in range(0, h-filter, filter):
		for j in range(0, w-filter, filter):
			crs.append(coarseness(image[i:i+filter,j:j+filter],5))
			con.append(contrast(image[i:i+filter,j:j+filter]))
			dire.append(directionality(image[i:i+filter,j:j+filter]))
			lin.append(linelikeness(image[i:i+filter,j:j+filter], 4, 16)*10)

	crs = np.array(crs)
	con = np.array(con)
	dire = np.array(dire)
	lin = np.array(lin)
	Dcrs = np.std(crs)
	Dcon = np.std(con)
	Ddir = np.std(dire)
	Dlin = np.std(lin)

	return [Dcrs, Dcon, Ddir, Dlin]

def roughness(fcrs, fcon):
	return fcrs + fcon

def tamura_feature(image, dist=1, levels=256):
	if len(image.shape)==3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	mean = np.mean(image)
	std = np.std(image)
	var = np.var(image)

	image = image / (256. / levels)
	image = np.uint8(image)

	# 计算纹理粗糙度
	rough = std / mean

	# 计算纹理方向性
	glcm = graycomatrix(image, [dist], [0], levels, symmetric=True, normed=True)
	moments = cv2.moments(glcm[:,:,0,0])
	f2 = moments['mu20'] + moments['mu02']
	f3 = (moments['mu20'] - moments['mu02']) ** 2 + 4 * moments['mu11']**2
	f4 = (moments['mu30'] - 3 * moments['mu12']) ** 2 + (3 * moments['mu21'] - moments['mu03']) ** 2
	direct = np.sqrt(f2 + f3 + f4)

	# 计算纹理规则性
	regul = std / var

	feature = [rough, direct, regul]
	return feature


if __name__ == '__main__':

	# img = cv2.imread('G:/paper/temp/feature_extract/g2/bg (1).jpg',cv2.IMREAD_GRAYSCALE)
	# print(img.shape)
	# fcrs = coarseness(img, 5)
	# print("coarseness: %f" % fcrs)
	# fcon = contrast(img)
	# print("contrast: %f" % fcon)
	# fdir= directionality(img)
	# print("directionality: %f" % fdir)
	#
	# flin = linelikeness(img, 4, 16)
	# freg = regularity(img,64)
	# frgh = roughness(fcrs, fcon)
	# print("linelikeness: %f" % flin)
	# print("regularity[Dcrs, Dcon, Ddir, Dlin]: {}".format(freg))
	# print("roughness: %f" % frgh)
	#
	# # print(fdir)
	#
	# feat = tamura_feature(img, levels=16)
	# print("rough, direct, regul: {}".format(feat))

	""" 批量计算 """
	# def creat_sheet(xls, metrics, cls):
	# 	sheets = []
	# 	for mi in metrics:
	# 		sheet = xls.add_sheet(mi)
	# 		for ci, cl in enumerate(cls):
	# 			sheet.write(0,ci+1, cl)
	# 		sheets.append(sheet)
	# 	return sheets
	#
	# path = "G:/paper/temp/feature_extract/"
	# cls = ['c1', 'c2', 'g1', 'g2']
	# metric = ['coarseness', 'contrast', 'directionality', 'linelikeness', 'roughness']
	# xls = xlwt.Workbook()
	# sheets = creat_sheet(xls, metric, cls)
	# for ci, cl in enumerate(cls):
	# 	img_path = path + '{}/'.format(cl)
	# 	for name in os.listdir(img_path):
	# 		img_id = name.split('(')[1].split(')')[0]
	# 		img_id = int(img_id)
	# 		img = cv2.imread(img_path + name, cv2.IMREAD_GRAYSCALE)
	#
	# 		fcrs = coarseness(img, 5)
	# 		fcon = contrast(img)
	# 		fdir = directionality(img)/100
	# 		flin = linelikeness(img, 4, 256)
	# 		frgh = roughness(fcrs, fcon)
	#
	# 		if ci == 0:
	# 			for mi, me in enumerate(metric):
	# 				sheets[mi].write(img_id, 0, img_id)
	#
	# 		sheets[0].write(img_id, ci + 1, fcrs)
	# 		sheets[1].write(img_id, ci + 1, fcon)
	# 		sheets[2].write(img_id, ci + 1, fdir)
	# 		sheets[3].write(img_id, ci + 1, flin)
	# 		sheets[4].write(img_id, ci + 1, frgh)
	#
	# 		xls.save('CGtamura1-3.xls')

	""" 测试方向度 """
	# def creat_sheet(xls, metrics, cls):
	# 	sheets = []
	# 	for mi in metrics:
	# 		sheet = xls.add_sheet(mi)
	# 		for ci, cl in enumerate(cls):
	# 			sheet.write(0,ci+1, cl)
	# 		sheets.append(sheet)
	# 	return sheets
	#
	#
	# list_file = open('abc.txt', 'w')
	# path = "G:/paper/temp/feature_extract/"
	# cls = ['c1', 'c2', 'g1', 'g2']
	# metric = ['dir10', 'dir12', 'dir14', 'dir16', 'dir18']
	# xls = xlwt.Workbook()
	# sheets = creat_sheet(xls, metric, cls)
	# for ci, cl in enumerate(cls):
	# 	img_path = path + '{}/'.format(cl)
	# 	for name in os.listdir(img_path):
	# 		img_id = name.split('(')[1].split(')')[0]
	# 		img_id = int(img_id)
	# 		img = cv2.imread(img_path + name, cv2.IMREAD_GRAYSCALE)
	#
	#
	# 		if ci == 0:
	# 			for mi, me in enumerate(metric):
	# 				sheets[mi].write(img_id, 0, img_id)
	#
	# 		for di, dd in enumerate(metric):
	# 			n = int(metric[di].split('r')[1])
	# 			fdir = directionality(img, t=n)
	# 			sheets[di].write(img_id, ci + 1, fdir)
	#
	# 		xls.save('CGtamura1-3.xls')
	#
	# list_file.close()

	""" 皮带速度和曝光时间批量计算 """
	# def creat_sheet(xls, metrics, cls, sds):
	# 	sheets = []
	# 	for mi in metrics:
	# 		sheet = xls.add_sheet(mi)
	# 		for sdi, sd in enumerate(sds):
	# 			sheet.write(0, sdi * cll + 1, sd)
	# 			for ci, cl in enumerate(cls):
	# 				sheet.write(1, sdi * cll + ci + 1, cl)
	#
	# 		sheets.append(sheet)
	# 	return sheets
	#
	# path = "D:/231122/sd/crop/"
	# cls = ["jm", "hm", "bg", "gg"]
	# sds = ['0', '0.4', '0.6', '0.8', '1.0', '1.2']
	# cll = len(cls)+1
	# ts = ['1000', '1500', '2000', '3000', '4000', '5000']
	# metric = ['coarseness', 'contrast', 'directionality']
	# xls = xlwt.Workbook()
	# sheets = creat_sheet(xls, metric, cls, sds)
	# for sdi, sdn in enumerate(sds):
	# 	img_path = path + '{}c/'.format(sdn)
	# 	if sdi == 0:
	# 		for mi, me in enumerate(metric):
	# 			for ti, tn in enumerate(ts):
	# 				sheets[mi].write(ti + 2, 0, tn)
	#
	# 	for name in os.listdir(img_path):
	# 		img_t = name.split('-')[1]
	# 		img_ti = ts.index(img_t)
	# 		img_c = name.split('-')[-1][0:2]
	# 		img_ci = cls.index(img_c)
	#
	# 		img = cv2.imread(img_path + name, cv2.IMREAD_GRAYSCALE)
	#
	# 		fcrs = coarseness(img, 5)
	# 		fcon = contrast(img)
	# 		fdir = directionality(img)/100
	# 		# flin = linelikeness(img, 4, 256)
	# 		# frgh = roughness(fcrs, fcon)
	#
	# 		sheets[0].write(img_ti + 2, sdi * cll + img_ci + 1, fcrs)
	# 		sheets[1].write(img_ti + 2, sdi * cll + img_ci + 1, fcon)
	# 		sheets[2].write(img_ti + 2, sdi * cll + img_ci + 1, fdir)
	# 		# sheets[3].write(img_ti + 2, sdi * sdl + img_ci + 1, flin)
	# 		# sheets[4].write(img_ti + 2, sdi * sdl + img_ci + 1, frgh)
	#
	# 		xls.save('CGtamura_sd.xls')

	""" 光照和曝光时间批量计算 """

	# def creat_sheet(xls, metrics, cls, ts):
	# 	sheets = []
	# 	for mi in metrics:
	# 		sheet = xls.add_sheet(mi)
	# 		for ti, tn in enumerate(ts):
	# 			sheet.write(0, ti * cll + 1, tn)
	# 			for ci, cl in enumerate(cls):
	# 				sheet.write(1, ti * cll + ci + 1, cl)
	#
	# 		sheets.append(sheet)
	# 	return sheets
	#
	# ts = ['1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900'
	# 	, '2000', '2100', '2200', '2300', '2400', '2500', '2600', '2700', '2800', '2900'
	# 	, '3000', '3500', '4000', '4500', '5000']
	# gzs = ["l2000", "l2500", "l3000", "l3500"]
	# metric = ['coarseness', 'contrast', 'directionality']
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
	# for gz in gzs:
	# 	xls = xlwt.Workbook()
	# 	sheets = creat_sheet(xls, metric, cls, ts)
	# 	path = "D:/231122/gz/crop/{}/".format(gz)
	#
	# 	for cli, cln in enumerate(cls):
	# 		file_path = path + cln + '/'
	#
	# 		for name in os.listdir(file_path):
	# 			img_id = int(name.split('-')[1][2:])
	# 			img_ts = name.split('-')[-1].split('.')[0]
	# 			img_ti = ts.index(img_ts)
	#
	# 			img = cv2.imread(file_path + name, cv2.IMREAD_GRAYSCALE)
	#
	# 			fcrs = coarseness(img, 5)
	# 			fcon = contrast(img)
	# 			fdir = directionality(img) / 10
	# 			# flin = linelikeness(img, 4, 256)
	# 			# frgh = roughness(fcrs, fcon)
	#
	# 			sheets[0].write(img_id + 1, img_ti * cll + cli + 1, fcrs)
	# 			sheets[1].write(img_id + 1, img_ti * cll + cli + 1, fcon)
	# 			sheets[2].write(img_id + 1, img_ti * cll + cli + 1, fdir)
	#
	# 			if cli == 0 and img_ti == 0:
	# 				for mi, me in enumerate(metric):
	# 					sheets[mi].write(img_id + 1, 0, img_id)
	#
	# 		xls.save('CGtamura_gz_{}.xls'.format(gz))


	""" 光照和曝光时间批量计算(整幅图像) """

	def creat_sheet(xls, metrics, cls, ts):
		sheets = []
		for mi in metrics:
			sheet = xls.add_sheet(mi)
			for ti, tn in enumerate(ts):
				sheet.write(0, ti + 1, tn)
			for ci, cl in enumerate(cls):
				sheet.write(ci + 1, 0, cl)

			sheets.append(sheet)
		return sheets


	ts = ['1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900'
		, '2000', '2100', '2200', '2300', '2400', '2500', '2600', '2700', '2800', '2900'
		, '3000', '3500', '4000', '4500', '5000']
	gzs = ["l2000", "l2500", "l3000", "l3500"]
	metric = ['coarseness', 'contrast', 'directionality']
	cls = ["jm", "hm", "bg", "gg"]

	cll = len(cls) + 1

	# path = "D:/231122/gz/crop/"

	for gz in gzs:
		xls = xlwt.Workbook()
		sheets = creat_sheet(xls, metric, cls, ts)
		path = "D:/231122/gz/{}/".format(gz)

		for name in os.listdir(path):
			img_cl = name[:2]
			img_cli = cls.index(img_cl)
			img_ts = name[2:6]
			img_ti = ts.index(img_ts)

			img = cv2.imread(path + name, cv2.IMREAD_GRAYSCALE)

			fcrs = coarseness(img, 5)
			fcon = contrast(img)
			fdir = directionality(img) / 10

			sheets[0].write(img_cli + 1, img_ti + 1, fcrs)
			sheets[1].write(img_cli + 1, img_ti + 1, fcon)
			sheets[2].write(img_cli + 1, img_ti + 1, fdir)

			xls.save('CGtamura_gzimage_{}.xls'.format(gz))

