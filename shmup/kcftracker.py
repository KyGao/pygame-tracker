# -*- coding: utf-8 -*

import numpy as np
import cv2

import fhog
import kcftools as tools


class KCFTracker:
	"""
	定义KCFTracker类
	"""
	def __init__(self, hog=False, fixed_window=True, multiscale=False):
		"""
		初始化KCF类的各种参数
		:param hog: 是否使用hog特征
		:param fixed_window: 自适应窗口，默认为True
		:param multiscale: 是否在多尺度下进行检测
		"""
		self.lambdar = 0.0001   # 正则化参数，
		self.padding = 2.5   # 对目标周围进行填充
		self.output_sigma_factor = 0.125   # 高斯带宽

		if hog:  # 使用hog特征
			self.interp_factor = 0.012   # 线性插值因子
			self.sigma = 0.6  # 高斯核带宽
			self.cell_size = 4   # HOG细胞单元大小
			self._hogfeatures = True
		else:  # 否则只使用灰度图像特征
			self.interp_factor = 0.075
			self.sigma = 0.2
			self.cell_size = 1
			self._hogfeatures = False

		if multiscale:
			self.template_size = 96   # 模板大小
			self.scale_step = 1.05   # 多尺度估计缩放步长
			self.scale_weight = 0.96   # 用于增加稳定性

		elif fixed_window:
			self.template_size = 96
			self.scale_step = 1

		else:
			self.template_size = 1
			self.scale_step = 1

		self._tmpl_sz = [0, 0]
		self._roi = [0., 0., 0., 0.]
		self.size_patch = [0, 0, 0]
		self._scale = 1.
		self._alphaf = None
		self._prob = None
		self._tmpl = None
		self.hann = None

	def subPixelPeak(self, left, center, right):
		"""
		对目标的位置进行插值，计算一维亚像素峰值
		:param left: 左值
		:param center: 中心值
		:param right: 右值
		:return: 返回需要改变的偏移量大小
		"""
		divisor = 2*center - right - left
		return 0 if abs(divisor) < 1e-3 else 0.5 * (right - left) / divisor

	def createHanningMats(self):
		"""
		初始化汉宁窗，只在第一帧时被调用
		:return:
		"""
		hann2t, hann1t = np.ogrid[0:self.size_patch[0], 0:self.size_patch[1]]

		hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (self.size_patch[1] - 1)))
		hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (self.size_patch[0] - 1)))
		hann2d = hann2t * hann1t

		if self._hogfeatures:
			hann1d = hann2d.reshape(self.size_patch[0] * self.size_patch[1])
			self.hann = np.zeros((self.size_patch[2], 1), np.float32) + hann1d
		else:
			self.hann = hann2d
		self.hann = self.hann.astype(np.float32)

	def createGaussianPeak(self, sizey, sizex):
		"""
		创建高斯峰函数
		:param sizey: 二维高斯峰y向大小
		:param sizex: 二维高斯峰x向大小
		:return:
		"""
		syh, sxh = sizey / 2, sizex / 2
		output_sigma = np.sqrt(sizex * sizey) / self.padding * self.output_sigma_factor
		mult = -0.5 / (output_sigma * output_sigma)
		y, x = np.ogrid[0:sizey, 0:sizex]
		y, x = (y-syh)**2, (x-sxh)**2
		res = np.exp(mult * (y+x))
		return tools.fftd(res)

	def gaussianCorrelation(self, x1, x2):
		"""
		使用带宽sigma计算高斯卷积核以用于所有图像X和Y之间的相对位移
		:param x1: 输入图像x1
		:param x2: 输入图像x2
		:return: 返回二者的相对位移
		"""
		if self._hogfeatures:		# 使用HOG特征
			c = np.zeros((self.size_patch[0], self.size_patch[1]), np.float32)
			for i in range(self.size_patch[2]):
				x1aux = x1[i, :].reshape((self.size_patch[0], self.size_patch[1]))		# 按照原来的排列形式重排第i个属性
				x2aux = x2[i, :].reshape((self.size_patch[0], self.size_patch[1]))
				caux = cv2.mulSpectrums(tools.fftd(x1aux), tools.fftd(x2aux), 0, conjB=True)		# 利用核相关性公式
				caux = tools.real(tools.fftd(caux, True))
				c += caux
			c = tools.rearrange(c)
		else:		# 使用灰度特征
			c = cv2.mulSpectrums(tools.fftd(x1), tools.fftd(x2), 0, conjB=True)
			c = tools.fftd(c, True)
			c = tools.real(c)
			c = tools.rearrange(c)

		if x1.ndim == 3 and x2.ndim == 3:
			d = (np.sum(x1[:, :, 0] * x1[:, :, 0]) + np.sum(x2[:, :, 0] * x2[:, :, 0]) - 2.0 * c)\
				/ (self.size_patch[0] * self.size_patch[1] * self.size_patch[2])
		elif x1.ndim == 2 and x2.ndim == 2:
			d = (np.sum(x1 * x1) + np.sum(x2 * x2) - 2.0 * c)\
				/ (self.size_patch[0] * self.size_patch[1] * self.size_patch[2])

		d = d * (d >= 0)
		d = np.exp(-d / (self.sigma * self.sigma))

		return d

	def getFeatures(self, image, inithann, scale_adjust=1.0):
		"""
		提取窗口特征
		//步骤：
		//1.根据给定的框框找到合适的框框
		//2.提取HOG特征
		//3.对特征进行归一化和截断
		//4.对特征进行PCA降维
		//5.创建一个常数阵，对所有特征根据cell的位置进行加权
		:param image: 输入图像
		:param inithann: 是否使用汉宁窗
		:param scale_adjust: 调整因子
		:return: 返回提取到的特征
		"""
		extracted_roi = [0, 0, 0, 0]		# 用于提取特征的窗口
		cx = self._roi[0] + self._roi[2] / 2
		cy = self._roi[1] + self._roi[3] / 2

		if inithann: 		# 初始化汉宁窗，只在第一帧的时候执行一次
			padded_w = self._roi[2] * self.padding
			padded_h = self._roi[3] * self.padding

			if self.template_size > 1:		# 按照长宽比修改_tmpl长宽大小，保证长边为96
				if padded_w >= padded_h:
					self._scale = padded_w / float(self.template_size)
				else:
					self._scale = padded_h / float(self.template_size)
				self._tmpl_sz[0] = int(padded_w / self._scale)
				self._tmpl_sz[1] = int(padded_h / self._scale)
			else:
				self._tmpl_sz[0] = int(padded_w)
				self._tmpl_sz[1] = int(padded_h)
				self._scale = 1.

			if self._hogfeatures:		# 设置_tmpl_sz的长宽:向上取原来长宽的最小2*cell_size倍
				self._tmpl_sz[0] = int(self._tmpl_sz[0]) // (2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
				self._tmpl_sz[1] = int(self._tmpl_sz[1]) // (2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
			else:
				self._tmpl_sz[0] = int(self._tmpl_sz[0]) // 2 * 2
				self._tmpl_sz[1] = int(self._tmpl_sz[1]) // 2 * 2

		extracted_roi[2] = int(scale_adjust * self._scale * self._tmpl_sz[0]) 		# 检测区域大小
		extracted_roi[3] = int(scale_adjust * self._scale * self._tmpl_sz[1])
		extracted_roi[0] = int(cx - extracted_roi[2] / 2)		# 检测区域的左上角坐标
		extracted_roi[1] = int(cy - extracted_roi[3] / 2)

		z = tools.subwindow(image, extracted_roi, cv2.BORDER_REPLICATE)			# 提取目标区域像素，超边界则做填充
		if z.shape[1] != self._tmpl_sz[0] or z.shape[0] != self._tmpl_sz[1]:

			z = cv2.resize(z, tuple(self._tmpl_sz))		# 按照比例缩小边界大小

		if self._hogfeatures:		# 提取HOG特征点
			mapp = {'sizeX': 0, 'sizeY': 0, 'numFeatures': 0, 'map': 0}
			mapp = fhog.getFeatureMaps(z, self.cell_size, mapp)
			mapp = fhog.normalizeAndTruncate(mapp, 0.2)			# 对HOG特征归一化和截断
			mapp = fhog.PCAFeatureMaps(mapp)		# PCA降维
			self.size_patch = list(map(int, [mapp['sizeY'], mapp['sizeX'], mapp['numFeatures']]))
			FeaturesMap = mapp['map'].reshape((self.size_patch[0] * self.size_patch[1], self.size_patch[2])).T
		else:
			if z.ndim == 3 and z.shape[2] == 3:
				FeaturesMap = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)		# 如果图像时三通道则转换为灰度图
			elif z.ndim == 2:
				FeaturesMap = z
			FeaturesMap = FeaturesMap.astype(np.float32) / 255.0 - 0.5
			self.size_patch = [z.shape[0], z.shape[1], 1]

		if inithann:
			self.createHanningMats()

		FeaturesMap = self.hann * FeaturesMap
		return FeaturesMap

	def detect(self, z, x):
		"""
		根据上一帧结果计算当前帧的目标位置
		:param z: 之前的训练结果
		:param x: 当前的特征图
		:return: 目标位置
		"""
		k = self.gaussianCorrelation(x, z)		# 计算x和z之间的高斯相关核
		res = tools.real(tools.fftd(tools.complexMultiplication(self._alphaf, tools.fftd(k)), True))	# 计算目标得分

		_, pv, _, pi = cv2.minMaxLoc(res)		# 定位峰值坐标位置
		p = [float(pi[0]), float(pi[1])]

		if pi[0] > 0 and pi[0] < res.shape[1]-1:
			p[0] += self.subPixelPeak(res[pi[1], pi[0]-1], pv, res[pi[1], pi[0]+1])
		if 0 < pi[1] < res.shape[0] - 1:
			p[1] += self.subPixelPeak(res[pi[1]-1, pi[0]], pv, res[pi[1]+1, pi[0]])

		p[0] -= res.shape[1] / 2.
		p[1] -= res.shape[0] / 2.

		return p, pv

	def train(self, x, train_interp_factor):
		"""
		根据每一帧的结果训练样本并更新模板
		:param x: 新的目标图像
		:param train_interp_factor: 训练因子
		:return: 返回最新的训练模板
		"""
		k = self.gaussianCorrelation(x, x)
		alphaf = tools.complexDivision(self._prob, tools.fftd(k) + self.lambdar)

		self._tmpl = (1 - train_interp_factor) * self._tmpl + train_interp_factor * x		# 更新模板特征
		self._alphaf = (1 - train_interp_factor) * self._alphaf + train_interp_factor * alphaf		# 更新岭回归系数的值

	def init(self, roi, image):
		"""
		初始化跟踪器，包括回归参数的计算，变量的初始化
		:param roi: 目标的初始框
		:param image: 初始帧
		:return:
		"""
		self._roi = list(map(float, roi))
		assert(roi[2] > 0 and roi[3] > 0)
		self._tmpl = self.getFeatures(image, 1)
		self._prob = self.createGaussianPeak(self.size_patch[0], self.size_patch[1])
		self._alphaf = np.zeros((self.size_patch[0], self.size_patch[1], 2), np.float32)
		self.train(self._tmpl, 1.0)

	def update(self, image):
		"""
		获取当前帧的目标位置以及尺度
		:param image: 当前帧画面的整幅图像
		:return:
		"""
		# 如果框越界，就让框保持在越界的位置
		if self._roi[0] + self._roi[2] <= 0:  self._roi[0] = -self._roi[2] + 1
		if self._roi[1] + self._roi[3] <= 0:  self._roi[1] = -self._roi[2] + 1
		if self._roi[0] >= image.shape[1] - 1:  self._roi[0] = image.shape[1] - 2
		if self._roi[1] >= image.shape[0] - 1:  self._roi[1] = image.shape[0] - 2

		# 跟踪框的中心坐标
		cx = self._roi[0] + self._roi[2] / 2.
		cy = self._roi[1] + self._roi[3] / 2.

		# 在尺度不变的情况下检测峰值结果
		loc, peak_value = self.detect(self._tmpl, self.getFeatures(image, 0, 1.0))

		if self.scale_step != 1:
			# 检测一个略小的尺度
			new_loc1, new_peak_value1 = self.detect(self._tmpl, self.getFeatures(image, 0, 1.0 / self.scale_step))
			# 检测一个略大的尺度
			new_loc2, new_peak_value2 = self.detect(self._tmpl, self.getFeatures(image, 0, self.scale_step))

			# 在较小的尺度下权重降低了，得到的峰值都是最大的，则小尺度下检测道德可以被认为是目标
			if self.scale_weight * new_peak_value1 > peak_value and new_peak_value1 > new_peak_value2:
				loc = new_loc1
				peak_value = new_peak_value1
				self._scale /= self.scale_step
				self._roi[2] /= self.scale_step
				self._roi[3] /= self.scale_step
			# 在一个较大的尺度小权重降低了，得到的峰值是最大的，则大尺度下检测到的可以被认为是目标
			elif self.scale_weight * new_peak_value2 > peak_value:
				loc = new_loc2
				peak_value = new_peak_value2
				self._scale *= self.scale_step
				self._roi[2] *= self.scale_step
				self._roi[3] *= self.scale_step

		self._roi[0] = cx - self._roi[2] / 2.0 + loc[0] * self.cell_size * self._scale
		self._roi[1] = cy - self._roi[3] / 2.0 + loc[1] * self.cell_size * self._scale

		if self._roi[0] >= image.shape[1]-1:  self._roi[0] = image.shape[1] - 1
		if self._roi[1] >= image.shape[0]-1:  self._roi[1] = image.shape[0] - 1
		if self._roi[0] + self._roi[2] <= 0:  self._roi[0] = -self._roi[2] + 2
		if self._roi[1] + self._roi[3] <= 0:  self._roi[1] = -self._roi[3] + 2
		assert(self._roi[2] > 0 and self._roi[3] > 0)

		x = self.getFeatures(image, 0, 1.0)
		self.train(x, self.interp_factor)

		return self._roi
