import cv2

# OpenCV 进行基本的图像处理
# 1.对验证码图片进行高斯模糊滤波处理，消除部分噪声干扰
# 2.对验证码图片应用边缘检测算法，通过调整相应阈值识别出滑块边缘
# 3.对上一步得到的各个边缘轮廓信息，通过对比面积、位置、周长等特征筛选出最可能的轮廓位置，得到缺口位置。


# GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None)
# src：即需要被处理的图像。
# ksize：进行高斯滤波处理所用的高斯内核大小，它需要是一个元组，包含 x 和 y 两个维度。
# sigmaX：表示高斯核函数在 X 方向的的标准偏差。
# sigmaY：表示高斯核函数在 Y 方向的的标准偏差，若 sigmaY 为 0，就将它设为 sigmaX，如果 sigmaX 和 sigmaY 都是 0，那么 sigmaX 和 sigmaY 就通过 ksize 计算得出。

# Canny(image, threshold1, threshold2, edges=None, apertureSize=None, L2gradient=None)
# image：即需要被处理的图像。
# threshold1、threshold2：两个阈值，分别为最小和最大判定临界点。
# apertureSize：用于查找图像渐变的 Sobel 内核的大小。
# L2gradient：指定用于查找梯度幅度的等式。

# findContours(image, mode, method, contours=None, hierarchy=None, offset=None)
# image：即需要被处理的图像。
# mode：定义轮廓的检索模式，详情见 OpenCV 的 RetrievalModes 的介绍。
# method：定义轮廓的近似方法，详情见 OpenCV 的 ContourApproximationModes 的介绍。
GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5)
GAUSSIAN_BLUR_SIGMA_X = 0
CANNY_THRESHOLD1 = 200
CANNY_THRESHOLD2 = 450


# 传入待处理图像信息，返回高斯滤波处理后的图像
def get_gaussian_blur_image(image):
    return cv2.GaussianBlur(image, GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_SIGMA_X)


# 传入待处理图像信息，返回边缘检测处理后的图像，
def get_canny_image(image):
    return cv2.Canny(image, CANNY_THRESHOLD1, CANNY_THRESHOLD2)


# 传入待处理图像信息，返回检测到的轮廓信息
def get_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return contours


# 定义目标轮廓的下限和上限面积，分别为 contour_area_min 和 contour_area_max。
def get_contour_area_threshold(image_width, image_height):
    contour_area_min = (image_width * 0.13) * (image_height * 0.23) * 0.8
    contour_area_max = (image_width * 0.13) * (image_height * 0.23) * 1.2
    return contour_area_min, contour_area_max


# 定义目标轮廓的下限和上限周长，分别为 arc_length_min 和 arc_length_max。
def get_arc_length_threshold(image_width, image_height):
    arc_length_min = ((image_width * 0.13) + (image_height * 0.23)) * 2 * 0.8
    arc_length_max = ((image_width * 0.13) + (image_height * 0.23)) * 2 * 1.2
    return arc_length_min, arc_length_max


# 定义目标轮廓左侧的下限和上限偏移量，分别为 offset_min 和 offset_max。
def get_offset_threshold(image_width):
    offset_min = 0.2 * image_width
    offset_max = 0.85 * image_width
    return offset_min, offset_max


if __name__ == '__main__':
    image_name = 'img'
    image_raw = cv2.imread(f'{image_name}.png')
    image_height, image_width, _ = image_raw.shape
    # 图像灰度化处理
    image_gray = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)
    image_gaussian_blur = get_gaussian_blur_image(image_gray)
    image_canny = get_canny_image(image_gaussian_blur)
    cv2.imwrite(f'{image_name}_canny.png', image_canny)
    contours = get_contours(image_canny)
    contour_area_min, contour_area_max = get_contour_area_threshold(image_width, image_height)
    arc_length_min, arc_length_max = get_arc_length_threshold(image_width, image_height)
    offset_min, offset_max = get_offset_threshold(image_width)
    print('面积:', contour_area_min, contour_area_max)
    print('周长:', arc_length_min, arc_length_max)
    print('偏移量:', offset_min, offset_max)
    offset = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # 面积
        contourArea = w * h
        # 周长
        arcLength = (w + h) * 2
        print('面积', contourArea, '周长', arcLength, '偏移量', x)
        if contour_area_min < contourArea < contour_area_max and \
                arc_length_min < arcLength < arc_length_max and \
                offset_min < x < offset_max:
            # rectangle(InputOutputArray img, Point pt1, Point pt2,const Scalar& color, int thickness = 1, int lineType = LINE_8, int shift = 0);
            # img	被处理的图像
            # pt1	绘制矩形的左上点坐标
            # pt2	绘制矩形的右下点坐标
            # color	颜色 Scalar(255,255,0)
            # thickness	矩形框的线条宽度 详看#FILLED
            # lineType	线型 默认 LINE_8， 详看#LineTypes
            # shift	移位点坐标中的小数位数
            cv2.rectangle(image_raw, (x, y), (x + w, y + h), (0, 0, 255), 2)
            offset = x

    cv2.imwrite(f'{image_name}_result.png', image_raw)
    print('offset', offset)
