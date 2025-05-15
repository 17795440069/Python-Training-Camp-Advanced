# exercises/conv.py
"""
练习：二维卷积 (Convolution)

描述：
实现一个简单的二维卷积操作。

请补全下面的函数 `conv2d`。
"""
import numpy as np

def conv2d(image, kernel):
    """
    执行二维卷积操作 (无填充, 步幅为 1)。
    Args:
        x (np.array): 输入二维数组, 形状 (H, W)。
        kernel (np.array): 卷积核二维数组, 形状 (kH, kW)。
    Return:
        np.array: 卷积结果, 形状 (out_H, out_W)。
                  out_H = H - kH + 1
                  out_W = W - kW + 1
    """
    # 请在此处编写代码
    # 提示：
    # 1. 获取输入 x 和卷积核 kernel 的形状。
    # 2. 计算输出的高度和宽度。
    # 3. 初始化输出数组。
    # 4. 使用嵌套循环遍历输出数组的每个位置 (i, j)。
    # 5. 提取输入 x 中与当前卷积核对应的区域 (patch)。
    # 6. 计算 patch 和 kernel 的元素乘积之和 (np.sum(patch * kernel))。
    # 7. 将结果存入输出数组 out[i, j]。
    # 获取输入图像和卷积核的尺寸
    padding = 0
    stride = 1

    img_h, img_w = image.shape
    kernel_h, kernel_w = kernel.shape

    # 计算输出图像的尺寸
    output_h = (img_h - kernel_h + 2 * padding) // stride + 1
    output_w = (img_w - kernel_w + 2 * padding) // stride + 1

    # 对输入图像进行边缘填充
    if padding > 0:
        padded_image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    else:
        padded_image = image

    # 初始化输出图像
    output = np.zeros((output_h, output_w))

    # 遍历输出图像的每个位置
    for y in range(output_h):
        for x in range(output_w):
            # 计算输入图像中对应的区域
            y_start = y * stride
            y_end = y_start + kernel_h
            x_start = x * stride
            x_end = x_start + kernel_w

            # 提取输入图像的区域并与卷积核进行逐元素相乘后求和
            region = padded_image[y_start:y_end, x_start:x_end]
            output[y, x] = np.sum(region * kernel)

    return output

    pass 