import cv2
import numpy as np
import matplotlib.pyplot as plt


def image_high_low_split(image_path, ksize=5, sigma=1.0):
    """
    图片高低频分离：高斯滤波提取低频，原图减低频得到高频
    :param image_path: 图片路径
    :param ksize: 高斯核大小（必须是奇数，越大模糊越强，低频提取越明显）
    :param sigma: 高斯核标准差（越大模糊越强）
    :return: 原图、低频图、高频图（均为uint8格式，便于显示和保存）
    """
    # 1. 读取图片（保持彩色通道）
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片，请检查路径：{image_path}")

    # 转换为RGB格式（OpenCV默认BGR，matplotlib显示需要RGB）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 2. 高斯滤波提取低频部分（高斯核必须是奇数）
    if ksize % 2 == 0:
        ksize += 1  # 确保核大小为奇数
    low_freq = cv2.GaussianBlur(img_rgb, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    # 3. 计算高频部分（原图 - 低频图），处理数值溢出（确保在0-255范围内）
    # 转换为float32避免整数溢出，再归一化到0-255
    high_freq = img_rgb.astype(np.float32) - low_freq.astype(np.float32)
    # 高频部分可能有负数值，先归一化到0-255（增强可视化效果）
    high_freq = cv2.normalize(high_freq, None, 0, 255, cv2.NORM_MINMAX)
    high_freq = high_freq.astype(np.uint8)  # 转换回整数格式

    return img_rgb, low_freq, high_freq


def show_and_save_results(original, low_freq, high_freq, save_path=None):
    """
    可视化并保存结果
    :param original: 原图（RGB格式）
    :param low_freq: 低频图（RGB格式）
    :param high_freq: 高频图（RGB格式）
    :param save_path: 保存路径（None则不保存）
    """
    # 设置画布大小
    plt.figure(figsize=(15, 5))

    # 显示原图
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis("off")

    # 显示低频图
    plt.subplot(1, 3, 2)
    plt.imshow(low_freq)
    plt.title("Low Frequency (Gaussian Blur)")
    plt.axis("off")

    # 显示高频图
    plt.subplot(1, 3, 3)
    plt.imshow(high_freq)
    plt.title("High Frequency (Original - Low Frequency)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # 保存结果（如果指定路径）
    if save_path is not None:
        # 转换回BGR格式保存（OpenCV默认BGR）
        original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        low_freq_bgr = cv2.cvtColor(low_freq, cv2.COLOR_RGB2BGR)
        high_freq_bgr = cv2.cvtColor(high_freq, cv2.COLOR_RGB2BGR)

        # 拼接保存
        result = np.hstack((original_bgr, low_freq_bgr, high_freq_bgr))
        cv2.imwrite(save_path, result)
        print(f"结果已保存到：{save_path}")


if __name__ == "__main__":
    # 配置参数
    IMAGE_PATH = "E:/图片1.png"  # 输入图片路径（请替换为你的图片路径）
    KSIZE = 15  # 高斯核大小（奇数，建议5-31之间调整）
    SIGMA = 3.0  # 高斯标准差（建议0.5-5.0之间调整）
    SAVE_PATH = "high_low_split_result.jpg"  # 结果保存路径（None则不保存）

    try:
        # 执行高低频分离
        original, low_freq, high_freq = image_high_low_split(IMAGE_PATH, KSIZE, SIGMA)

        # 显示并保存结果
        show_and_save_results(original, low_freq, high_freq, SAVE_PATH)

    except Exception as e:
        print(f"程序执行出错：{e}")