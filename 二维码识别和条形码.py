'''import cv2
import numpy as np
import zbar

cameraCapture = cv2.VideoCapture(0)

while(True):
    #获取一帧
    ret, frame = cameraCapture.read()
    # 将这帧转换为灰度图
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == ord('q'):
        break
cameraCapture.release()
cv2.destroyAllWindows()'''
import pyzbar.pyzbar as pyzbar
from PIL import Image,ImageEnhance
image = "image.png"
img = Image.open(image)
#img = ImageEnhance.Brightness(img).enhance(2.0)#增加亮度
#img = ImageEnhance.Sharpness(img).enhance(17.0)#锐利化
#img = ImageEnhance.Contrast(img).enhance(4.0)#增加对比度
#img = img.convert('L')#灰度化
img.show()
barcodes = pyzbar.decode(img)
for barcode in barcodes:
    barcodeData = barcode.data.decode("utf-8")
    print(barcodeData)
