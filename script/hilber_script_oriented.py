import numpy as np
import matplotlib.pyplot as plt
from hilbertcurve.hilbertcurve import HilbertCurve
import cv2
import os

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
    
    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)  # 注意OpenCV中角度是逆时针为负
    
    # 进行旋转，保持原像素值不变
    rotated = cv2.warpAffine(image.astype(np.uint8), M, (w, h),
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0)
    
    # 记录填充位置的索引
    # original_area = padded_image[start_h:start_h+h, start_w:start_w+w]
    padding_indices = []

    # 只记录新添加的填充像素
    for i in range(h):
        for j in range(w):
            if rotated[i, j] == 0:
                padding_indices.append((i, j))
    
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
    
    # 删除padding_indices中的位置
    if padding_indices is not None:
        sequence = [value for idx, value in enumerate(sequence) if (idx not in padding_indices)]
    
    # 确保返回的序列长度为16
    return sequence

def visualize_and_save_sequences(original_image, angles=None, show_image=False):
    """
    可视化不同角度旋转后的图像和对应的一维序列，并保存结果
    """
    # 获取图像尺寸
    h, w = original_image.shape[:2]

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
                h, w = rotated_image.shape[:2]
                # padding_image_h = 2**(int(np.log2(h))+1)
                # padding_image_w = padding_image_h
            
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
                                       for i in range(h*w)]]
            points = np.array(hilbert_path)
            plt.plot(points[:, 1], points[:, 0], 'b-', linewidth=2)
            plt.grid(True)
            plt.title(f'Hilbert曲线路径 (阶数: {order})')
            for i, (x, y) in enumerate(hilbert_path):
                plt.text(y, x, str(h*w-i), ha='center', va='center')
            plt.xlim(-0.5, h - 0.5)
            plt.ylim(-0.5, h - 0.5)
            
            # 显示一维展开的序列
            plt.subplot(133)
            sequence_image = np.array(sequence).reshape(1, -1)
            plt.imshow(sequence_image, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title(f'旋转{angle}度后的一维序列')
            for i in range(h*w):
                plt.text(i, 0, f"{sequence[i]}", 
                        ha='center', va='center', color='white')
            plt.grid(True)
            
            plt.tight_layout()        # 保存图像
            plt.savefig(os.path.join(save_dir, f'rotation_{angle}_degrees.png'))

            if show_image:
                plt.show()
            else:
                plt.close()


def main():
    # 创建4x4的原始图像
    original_image = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])

    # 测试不同角度，包括非90度的角度
    # angles = [0, 45, 90, 135, 180, 225, 270, 315]
    angles = [45, 135, 225, 315]
    # angles = [0, 90, 180, 270]
    visualize_and_save_sequences(original_image, angles=angles, show_image=False)


if __name__ == "__main__":
    main()
