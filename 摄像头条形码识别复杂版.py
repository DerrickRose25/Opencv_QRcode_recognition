import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar

# 二维码定位
global check  # 全局变量check为校验位


def detect(image):
    global check
    # 把图像从RGB转换成灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 进行Sobel算子运算
    # 使用scharr操作(指定ksize=-1)构造灰度图在水平和竖直方向上的梯度幅值表示
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)  # 对x方向求导
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)  # 对y方向求导
    # Scharr操作后，从X梯度减去Y梯度得到轮廓图，此时噪点较多
    gradient = cv2.subtract(gradX, gradY)
    # 经过处理后，用convertScaleAbs()函数将其转回原来的uint8形式。否则将无法显示图像，而只是一副灰色的窗口
    gradient = cv2.convertScaleAbs(gradient)
    # cv2.imshow('gradient', gradient)
    # 然后对梯度图采用用9x9的核进行平均模糊,进行于降噪
    # 然后进行二值化处理，要么是255(白)要么是0(黑)
    blurred = cv2.blur(gradient, (9, 9))  # 通过低通滤波平滑图像
    ret, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)  # 进行图像固定阈值二值化
    # cv2.imshow("thresh",thresh)
    # 通过形态学操作，建立一个7*21的长方形内核，内核宽度大于长度，因此可以消除条形码中垂直条之间的缝隙
    # 将建立的内核应用到二值图中，以此来消除竖杠间的缝隙
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))  # 条形码
    '''kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))              #二维码'''
    # 对图像进行闭运算
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("closed",closed)
    # 所得图像仍有许多白点，通过腐蚀和膨胀去除白点,最后一个参数是腐蚀的次数
    closed = cv2.erode(closed, None, iterations=4)
    # cv2.imshow("closed1",closed)
    closed = cv2.dilate(closed, None, iterations=6)
    # cv2.imshow("closed2", closed)
    # 寻找轮廓
    contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 如果没有找到，返回空
    if len(contours) == 0:
        check = False
        return None
    # e
    c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(c)  # 生成最小外接矩形
    # box为一个ndarry数组，返回4个顶点位置
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)
    # cv2.imshow("frame", frame)
    check = True
    return box


# 二维码识别
def scan():
    box = detect(frame)  # 调用detect()函数来查找二维码返回二维码的位置
    # print(box)
    # 这下面的3步得到扫描区域，扫描区域要比检测出来的位置要大
    if check == True:
        min = np.min(box, axis=0)
        max = np.max(box, axis=0)
        roi = frame[min[1] - 10:max[1] + 10, min[0] - 10:max[0] + 10]
        # 把区域里的二维码传换成RGB，并把它转换成pil里面的图像，因为zbar得调用pil里面的图像，而不能用opencv的图像
        # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        # cv2.imshow("roi",roi)
        # print(roi.shape)
        if roi.any() != 0:
            barcodes = pyzbar.decode(roi)
            for barcode in barcodes:
                # 提取条形码的边界框的位置
                # 画出图像中条形码的边界框
                # (x, y, w, h) = barcode.rect
                # cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # 条形码数据为字节对象，所以如果我们想在输出图像上
                # 画出来，就需要先将它转换成字符串
                barcodeData = barcode.data.decode("utf-8")
                barcodeType = barcode.type
                # 向终端打印条形码数据和条形码类型
                print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
                return roi


if __name__ == '__main__':
    cameraCapture = cv2.VideoCapture(0)
    while True:
        # 获取当前帧
        ret, frame = cameraCapture.read()
        scan()
        cv2.imshow("camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cameraCapture.release()
    cv2.destroyAllWindows()

    '''image = cv2.imread("Images/coke.jpg")
    detect(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
