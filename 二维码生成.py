#简单用法：
'''import qrcode
import PIL
img = qrcode.make('hello, qrcode')
img.save('test.png')'''
#高级用法：
import qrcode
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)

data = input()      #运行时输入数据
qr.add_data(data)
qr.make(fit=True)
img = qr.make_image()
#img = qr.make_image(fill_color="green", back_color="white")            #设置二维码颜色
#img.show()
# 保存二维码为文件
img.save('a.jpg')
