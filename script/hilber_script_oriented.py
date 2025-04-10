import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from hilbertcurve.hilbertcurve import HilbertCurve

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def rotate_image(image, angle):
    """
    旋转图像，angle必须是90的倍数
    """
    # 将角度转换为逆时针旋转的次数(因为np.rot90是逆时针旋转)
    k = int((angle % 360) / 90)
    return np.rot90(image, k=k)


def rotate_image_arbitrary(image, angle):
    """
    任意角度旋转图像，使用0 padding扩展到2倍尺寸
    返回旋转后的图像和填充位置的索引
    """
    # 获取图像尺寸
    h, w = image.shape[:2]

    # 获取填充后图像的中心点
    center = (w // 2, h // 2)

    # 创建 2w×2h 全零画布
    canvas = np.zeros((2 * h, 2 * w), dtype=np.uint8)
    # 将原图像放入中心位置
    canvas[h // 2:3 * h // 2, w // 2:3 * w // 2] = image
    # 获取画布的中心点
    canvas_center = (2 * w // 2, 2 * h // 2)

    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(canvas_center, -angle, 1.0)  # 注意OpenCV中角度是逆时针为负

    # 进行旋转，保持原像素值不变
    rotated = cv2.warpAffine(canvas.astype(np.uint8), M, (2 * w, 2 * h),
                             flags=cv2.INTER_NEAREST,
                             borderValue=0)

    # 找到 padding 的索引（值为0的位置）
    padding_indices = np.argwhere(rotated == 0)

    return rotated, padding_indices


def get_hilbert_sequence(image, padding_indices=None):
    """
    获取图像对应的Hilbert曲线一维序列
    """
    h, w = image.shape[:2]
    hilbert_curve = HilbertCurve(int(np.log2(h)), 2)

    hilbert_path = []

    for i in range(h * w):
        coords = hilbert_curve.point_from_distance(i)
        x, y = coords[0], coords[1]
        if 0 <= x < h and 0 <= y < w:
            hilbert_path.append((x, y))

    # 返回Hilbert路径对应的图像像素值，排除padding_indices中的位置
    sequence = [image[y, x] for x, y in hilbert_path]

    # 删除sequence中所有值为0或者相邻重复的元素         
    # 创建一个新的序列，用于存储有效的像素值
    filtered_sequence = []
    for value in sequence:
        # 仅当值不为0且在filtered_sequence中不存在时，才将其添加
        if value != 0 and value not in filtered_sequence:
            filtered_sequence.append(value)
    sequence = filtered_sequence

    # 确保返回的序列长度为16
    return sequence


def display_sequence(sequence, angle):
    """
    显示一维序列
    """
    sequence_image = np.array(sequence).reshape(1, -1)
    plt.imshow(sequence_image, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(f'旋转{angle}度后的一维序列')
    for i in range(len(sequence)):
        plt.text(i, 0, f"{sequence[i]}",
                 ha='center', va='center', color='white')
    plt.grid(True)


def visualize_and_save_sequences(original_image, angles=None, show_image=False):
    """
    可视化不同角度旋转后的图像和对应的一维序列，并保存结果
    """
    # 获取图像尺寸
    original_h, original_w = original_image.shape[:2]

    if angles is None:
        angles = [0, 90, 180, 270]

    save_dir = 'hilbert_visualization'
    os.makedirs(save_dir, exist_ok=True)

    # 打开文件准备写入结果，指定编码为utf-8
    with open('hilbert_sequences_oriented.txt', 'w', encoding='utf-8') as f:
        for angle in angles:
            # 根据角度选择旋转方法
            if angle % 90 == 0:
                # 对于90度的倍数使用原来的方法
                rotated_image = rotate_image(original_image, angle)
                padding_indices = None
            else:
                # 对于任意角度使用新的方法
                rotated_image, padding_indices = rotate_image_arbitrary(original_image, angle)

            # 获取旋转后的图像尺寸
            h, w = rotated_image.shape[:2]

            # 获取一维序列
            sequence = get_hilbert_sequence(rotated_image, padding_indices)

            # 写入文件
            f.write(f"逆时针旋转角度（对图像）: {angle}度\n")
            f.write(f"一维序列: {sequence}\n")
            f.write("-" * 50 + "\n")

            # 可视化
            plt.figure(figsize=(15, 5))

            # 显示旋转后的原始图像
            plt.subplot(131)
            plt.imshow(rotated_image, cmap='viridis', extent=[-0.5, 3.5, 3.5, -0.5])
            plt.colorbar()
            plt.title(f'旋转{angle}度后的图像')
            for i in range(h):
                for j in range(w):
                    plt.text(j, i, f"{rotated_image[i, j]:.1f}",
                             ha='center', va='center', color='white')
            plt.grid(False)

            # 显示Hilbert曲线路径
            plt.subplot(132)
            order = int(np.log2(h))  # 计算曲线的阶数
            hilbert_path = [(coords[0], coords[1])
                            for coords in [HilbertCurve(order, 2).point_from_distance(i)
                                           for i in range(h * w)]]
            points = np.array(hilbert_path)
            plt.plot(points[:, 1], points[:, 0], 'b-', linewidth=2)
            plt.grid(True)
            plt.title(f'Hilbert曲线路径 (阶数: {order})')
            for i, (x, y) in enumerate(hilbert_path):
                plt.text(y, x, str(h * w - i), ha='center', va='center')
            plt.xlim(-0.5, h - 0.5)
            plt.ylim(-0.5, h - 0.5)

            # 显示一维展开的序列
            plt.subplot(133)
            display_sequence(sequence, angle)

            plt.tight_layout()  # 保存图像
            plt.savefig(os.path.join(save_dir, f'rotation_{angle}_degrees.png'))

            if show_image:
                plt.show()
            else:
                plt.close()


def hilbert_flatten(tensor):
    """
    输入一个4D张量，输出一个2D张量
    tensor: 形状为 [4090, 256, 8, 8] 的张量
    返回: 形状为 [4090, 256*64] 的张量
    """
    # 获取张量的形状
    n, c, h, w = tensor.shape

    # 创建Hilbert曲线对象
    hilbert_curve = HilbertCurve(int(np.log2(h)), 2)

    # 计算Hilbert曲线的索引
    hilbert_indices = torch.tensor([hilbert_curve.point_from_distance(i) for i in range(h * w)])

    # 使用索引重排张量
    # 先将张量的形状调整为 [4090, 256, 8*8]
    reshaped_tensor = tensor.view(n, c, h * w)

    # 使用高级索引重排
    flattened_sequences = reshaped_tensor[:, :, hilbert_indices[:, 0] * w + hilbert_indices[:, 1]]

    # 将结果展平为 [4090, 256*64]
    return flattened_sequences.view(n, -1)


def main():
    # 创建4x4的原始图像
    original_image = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])

    # 测试不同角度，包括非90度的角度
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    # angles = [45, 135, 225, 315]
    # angles = [0, 90, 180, 270]
    visualize_and_save_sequences(original_image, angles=angles, show_image=False)


if __name__ == "__main__":
    main()
