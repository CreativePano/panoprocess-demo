import base64
# from typing import io

# 图片处理函数 START
import cv2
import numpy as np
# 导入图像处理库和Numpy库
from PIL import Image, ImageEnhance

# 1. 浮雕
def filter_image(base64_str):
    # 将 base64 字符串转换为字节对象
    img_bytes = base64.b64decode(base64_str)
    # 将字节对象转换为 numpy 数组
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    # 将 numpy 数组转换为 opencv 图像对象
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    # 使用 opencv 进行滤镜处理（比如浮雕）
    # 参考 https://docs.opencv.org/4.5.4/d7/dbd/group__imgproc.html#ga27c049795ce870216ddfb366086b5a04
    kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
    filtered_img = cv2.filter2D(img, -1, kernel)
    # 将 opencv 图像对象转换为 numpy 数组
    filtered_img_array = cv2.imencode('.jpg', filtered_img)[1]
    # 将 numpy 数组转换为字节对象
    filtered_img_bytes = filtered_img_array.tobytes()
    # 将字节对象转换为 base64 字符串
    filtered_base64_str = base64.b64encode(filtered_img_bytes).decode()
    # 返回处理结果（也是 base64）
    return filtered_base64_str


# 2.电影青橙：这个滤镜可以让图片有一种电影的感觉，适用于色彩不鲜明的图片
def filter_image_two(base64_str):
    print('two start')
    # 将 base64 字符串转换为字节对象
    img_bytes = base64.b64decode(base64_str)
    # 将字节对象转换为 numpy 数组
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    # 将 numpy 数组转换为 opencv 图像对象
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    # 转换为np数组
    img_array = np.array(img)
    # 定义电影青橙滤镜效果的矩阵
    filter_matrix = np.array([[0.9, 0.1, 0], [0.2, 0.7, 0.1], [0, 0.3, 0.7]])

    # 将原始图片的RGB值与滤镜矩阵相乘，得到新的RGB值
    new_img_array = np.dot(img_array, filter_matrix)

    # 将新的RGB值限制在0到255之间
    new_img_array = np.clip(new_img_array, 0, 255)

    # 将新的RGB值转换为整数类型
    new_img_array = new_img_array.astype(np.uint8)

    # 将新的RGB值转换为图片对象
    new_img = Image.fromarray(new_img_array)

    # 调整光感-45，亮度-50，对比度+18，饱和度-35，结构+65，高光+30，色温-100，色调-85参数
    # 创建一个ImageEnhance.Color对象，用于调整色温和色调
    color_enhancer = ImageEnhance.Color(new_img)

    # 根据给定的参数调整色温和色调，并返回新的图片对象
    new_img = color_enhancer.enhance(-100 / 100)  # 色温-100，表示偏冷色调
    new_img = color_enhancer.enhance(-85 / 100)  # 色调-85，表示偏绿色

    # 创建一个ImageEnhance.Brightness对象，用于调整亮度
    brightness_enhancer = ImageEnhance.Brightness(new_img)

    # 根据给定的参数调整亮度，并返回新的图片对象
    new_img = brightness_enhancer.enhance(80 / 100)  # 亮度-50%，表示变暗

    # 创建一个ImageEnhance.Contrast对象，用于调整对比度
    contrast_enhancer = ImageEnhance.Contrast(new_img)

    # 根据给定的参数调整对比度，并返回新的图片对象
    new_img = contrast_enhancer.enhance(118 / 100)  # 对比度+18%，表示增强

    # 创建一个ImageEnhance.Sharpness对象，用于调整结构和高光
    sharpness_enhancer = ImageEnhance.Sharpness(new_img)

    # 根据给定的参数调整结构和高光，并返回新的图片对象
    new_img = sharpness_enhancer.enhance(165 / 100)  # 结构+65%，表示锐化边缘细节

    # 将new_img转换为opencv图像对象
    new_img = cv2.cvtColor(np.asarray(new_img), cv2.COLOR_RGB2BGR)

    # 将 opencv 图像对象转换为 numpy 数组
    filtered_img_array = cv2.imencode('.jpg', new_img)[1]
    # 将 numpy 数组转换为字节对象
    filtered_img_bytes = filtered_img_array.tobytes()
    # 将字节对象转换为 base64 字符串
    filtered_base64_str = base64.b64encode(filtered_img_bytes).decode()
    # 返回处理结果（也是 base64）
    return filtered_base64_str
    print('two end')
    # 返回处理过后的函数
    return new_base64_str


# 3.卡梅尔 : 这个滤镜可以让图片多一点蓝调，适用于原图片很暗的效果
def filter_image_three(base64_str):
    print("three start")
    # 将 base64 字符串转换为字节对象
    img_bytes = base64.b64decode(base64_str)
    # 将字节对象转换为 numpy 数组
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    # 将 numpy 数组转换为 opencv 图像对象
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    img = np.array(img)

    # 定义卡梅尔滤镜的参数
    lightness = -25  # 光感
    brightness = -20  # 亮度
    contrast = 20  # 对比度
    saturation = 15  # 饱和度
    highlight = 20  # 高光
    temperature = -40  # 色温
    grain = 10  # 颗粒

    # 调整图像的亮度和对比度
    img = img * (contrast / 127 + 1) - contrast + brightness

    # 调整图像的饱和度（假设饱和度为RGB三通道的平均值）
    gray = img.mean(axis=2, keepdims=True)
    img = gray + (img - gray) * (saturation / 127 + 1)

    # 调整图像的色温（假设白平衡为6500K）
    temp_ratio = temperature / 100
    red_channel = img[:, :, 0] * (1 + temp_ratio)
    blue_channel = img[:, :, 2] * (1 - temp_ratio)
    img[:, :, 0] = np.clip(red_channel, 0, 255)
    img[:, :, 2] = np.clip(blue_channel, 0, 255)

    # 调整图像的高光（假设高光区域为亮度大于200的像素）
    mask = img > 200
    img[mask] += highlight

    # 添加图像的颗粒（假设颗粒为正态分布的噪声）
    noise = np.random.normal(0, grain, img.shape)
    img += noise
    # 将图像转换回PIL格式并保存
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = Image.fromarray(img)
    #将图像转换为opencv图像对象
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    # 将 opencv 图像对象转换为 numpy 数组
    filtered_img_array = cv2.imencode('.jpg', img)[1]
    # 将 numpy 数组转换为字节对象
    filtered_img_bytes = filtered_img_array.tobytes()
    # 将字节对象转换为 base64 字符串
    filtered_base64_str = base64.b64encode(filtered_img_bytes).decode()
    # 返回处理结果（也是 base64）
    print("three end")
    return filtered_base64_str


# 4.动漫青森,这个滤镜可以让图片看起来有一种动漫的感觉
def filter_image_four(base64_str):
    print("filter_image_four start")
    # 将 base64 字符串转换为字节对象
    img_bytes = base64.b64decode(base64_str)
    # 将字节对象转换为 numpy 数组
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    # 将 numpy 数组转换为 opencv 图像对象
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    img = np.array(img)

    # 对整个数组进行颜色变换和亮度调整
    result = np.zeros_like(img)
    result[:, :, 0] = np.clip(img[:, :, 0] * 1.1 + 10, 0, 255)  # R
    result[:, :, 1] = np.clip(img[:, :, 1] * 1.05 + 5, 0, 255)  # G
    result[:, :, 2] = np.clip(img[:, :, 2] * 0.9 - 10, 0, 255)  # B

    # 将结果数组转换为BGR格式并保存图片
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    # 将 opencv 图像对象转换为 numpy 数组
    filtered_img_array = cv2.imencode('.jpg', result)[1]
    # 将 numpy 数组转换为字节对象
    filtered_img_bytes = filtered_img_array.tobytes()
    # 将字节对象转换为 base64 字符串
    filtered_base64_str = base64.b64encode(filtered_img_bytes).decode()
    # 返回处理结果（也是 base64）
    print("filter_image_four end")
    return filtered_base64_str


# 5.这个滤镜可以让图片看起来有一种南法假日的感觉，适合晴天街景
def filter_image_five(base64_str):
    print("filter_image_five begin")
    # 将 base64 字符串转换为字节对象
    img_bytes = base64.b64decode(base64_str)
    # 将字节对象转换为 numpy 数组
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    # 将 numpy 数组转换为 opencv 图像对象
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]  # 色调通道
    s = hsv[:, :, 1]  # 饱和度通道
    v = hsv[:, :, 2]  # 亮度通道
    # 调整H的值，可以改变颜色的种类，比如红、橙、黄、绿等。H的值是一个角度，从0到360度，表示在一个圆形的颜色轮上不同的位置。
    # 调整S的值，可以改变颜色的纯度，也就是混合了多少灰色。S的值是一个百分比，从0到100%，表示从灰色到鲜艳的程度。
    # 调整V的值，可以改变颜色的明暗，也就是混合了多少黑色。V的值也是一个百分比，从0到100%，表示从黑色到白色的程度。
    h = np.clip(h * 1.2 - 10, 0, 255)  # 调整色调
    s = np.clip(s * 1.2 + 10, 0, 255)  # 调整饱和度
    v = np.clip(v * 0.8 + 20, 0, 250)  # 调整亮度

    hsv[:, :, 0] = h  # 更新色调通道
    hsv[:, :, 1] = s  # 更新饱和度通道
    hsv[:, :, 2] = v  # 更新亮度通道
    img_pandora = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # 将 opencv 图像对象转换为 numpy 数组
    filtered_img_array = cv2.imencode('.jpg', img_pandora)[1]
    # 将 numpy 数组转换为字节对象
    filtered_img_bytes = filtered_img_array.tobytes()
    # 将字节对象转换为 base64 字符串
    filtered_base64_str = base64.b64encode(filtered_img_bytes).decode()
    # 返回处理结果（也是 base64）
    print("filter_image_five end")
    return filtered_base64_str


# 调整饱和度
def adjust_saturation(img,s):
    s = s * 0.01 # 缩放因子，s为100时表示不变，小于100时降低饱和度，大于100时提高饱和度
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) # 将BGR格式转换为HSV格式
    hsv_img[:,:,1] = hsv_img[:,:,1] * s # 对第二通道（S通道）进行缩放操作
    hsv_img[hsv_img > 255] = 255 # 对超出范围 [0,255] 的值进行截断处理
    dst = cv2.cvtColor(hsv_img,cv2.COLOR_HSV2BGR) # 将HSV格式转换回BGR格式
    return dst

# 调整色相
def adjust_hue(img,s):
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) # 将BGR格式转换为HSV格式
    hsv_img[:,:,0] = hsv_img[:,:,0] + s # 对第二通道（S通道）进行缩放操作
    hsv_img[hsv_img > 255] = 255 # 对超出范围 [0,255] 的值进行截断处理
    dst = cv2.cvtColor(hsv_img,cv2.COLOR_HSV2BGR) # 将HSV格式转换回BGR格式
    return dst

#调整锐化 高斯模糊和加权平均的方法来实现锐化
def adjust_sharpen(img, alpha):
    # 对图像进行高斯模糊
    gaussian = cv2.GaussianBlur(img, (7, 7), 0)
    # 计算锐化后的图像
    sharpened = float(alpha) * img + float(1 - alpha) * gaussian
    return sharpened

#6.五个可以自己调整的参数 亮度 对比度 饱和度 色相 锐化
def filter_image_result_oneself():
    img = cv2.imread('test.jpg')
    alpha = 1.1  # 对比度  
    beta = 15  # 亮度
    saturation = 40  # 饱和度
    hue = 20  # 色相
    sharpen = 1.5  # 锐化
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)  # 亮度、对比度调整
    new_img = adjust_saturation(new_img, saturation)  # 饱和度调整
    new_img = adjust_hue(new_img, hue)   # 色相调整
    new_img = adjust_sharpen(new_img, sharpen)  # 锐化调整
    cv2.imwrite('66.jpg', new_img)

# 6.
def filter_image_one(base64_str, brightness,saturation,temp,tint):
    print('开始执行 one 方法')
    # 将 base64 字符串转换为字节对象
    img_bytes = base64.b64decode(base64_str)
    # 将字节对象转换为 numpy 数组
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    # 将 numpy 数组转换为 opencv 图像对象
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # 调整图片的亮度、饱和度、色温和色调
    # 将BGR格式转换成HSV格式，方便调整饱和度、色温和色调
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 分别获取H,S,V通道的值，并进行相应的变化
    h, s, v = cv2.split(hsv)
    s = s + saturation  # 饱和度变化
    v = v + brightness  # 亮度变化
    # 将色温和色调的变化转换成角度，并加到H通道上
    angle = temp / 200 * 180 + tint / 200 * 180
    h = h + angle
    # 将H,S,V通道合并
    hsv[:, :, 0] = np.clip(h, 0, 255)  # R
    hsv[:, :, 1] = np.clip(s, 0, 255)  # G
    hsv[:, :, 2] = np.clip(v, 0, 255)  # B
    # 并转换回BGR格式
    img_new = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 应用油画滤镜或者高斯模糊滤镜，这里以油画滤镜为例，你也可以尝试其他滤镜效果
    #  img_new = cv2.xphoto.oilPainting(img_new, 7, 1)  # 油画滤镜参数为半径和动态范围
    img_new = cv2.GaussianBlur(img_new, (5, 5), 0)
    # 将 opencv 图像对象转换为 numpy 数组
    filtered_img_array = cv2.imencode('.jpg', img_new)[1]
    # 将 numpy 数组转换为字节对象
    filtered_img_bytes = filtered_img_array.tobytes()
    # 将字节对象转换为 base64 字符串
    filtered_base64_str = base64.b64encode(filtered_img_bytes).decode()
    print('执行 one 方法结束')
    # 返回处理结果（也是 base64）
    return filtered_base64_str

# 打开根目录下的图片并运行图片处理函数
def main():
    ################ 读取图片，转为 base64  ################
    with open("test.jpg", "rb") as f:   # 以二进制读取本地图片
        base64_data = base64.b64encode(f.read())    # 读取文件内容，转换为base64编码
        base64_data = base64_data.decode()  # 转换为字符串

    ################接收 base64 编码的图片，进行处理，返回处理后的 base64 编码的图片################
    print('开始执行')

    # 选择调用哪个方法
    # 命令行输入 input
    keyin = input('请输入数字：')
    if keyin == '0':
        base64_data = filter_image_one(base64_data, 30, 10, 30, 10)  # 获取返回值
        imgdata = base64.b64decode(base64_data)  # 解码，将base64编码转换为图片
        file = open('0.jpg', 'wb')  # 打开文件，准备写入
        file.write(imgdata)  # 写入文件
        file.close()  # 关闭文件
    elif keyin == '1':
        base64_data2 = filter_image_three(base64_data)  # 获取返回值
        imgdata = base64.b64decode(base64_data2)  # 解码，将base64编码转换为图片
        file = open('1.jpg', 'wb')  # 打开文件，准备写入
        file.write(imgdata)  # 写入文件
        file.close()  # 关闭文件
    elif keyin == '2':
        base64_data3 = filter_image_four(base64_data)
        imgdata = base64.b64decode(base64_data3)  # 解码，将base64编码转换为图片
        file = open('2.jpg', 'wb')  # 打开文件，准备写入
        file.write(imgdata)  # 写入文件
        file.close()  # 关闭文件
    elif keyin == '3':
        base64_data4 = filter_image_five(base64_data)
        imgdata = base64.b64decode(base64_data4)
        file = open('3.jpg', 'wb')  # 打开文件，准备写入
        file.write(imgdata)  # 写入文件
        file.close()  # 关闭文件
    else:
        print('输入错误')


if __name__ == '__main__':
    main()
