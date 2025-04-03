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
    """
    # 获取图像尺寸
    h, w = image.shape[:2]
    
    # 创建2倍大小的画布
    padded_h = h * 2
    padded_w = w * 2
    
    # 创建填充后的图像
    padded_image = np.zeros((padded_h, padded_w), dtype=image.dtype)
    
    # 将原图放在中心位置
    start_h = (padded_h - h) // 2
    start_w = (padded_w - w) // 2
    padded_image[start_h:start_h+h, start_w:start_w+w] = image
    
    # 获取填充后图像的中心点
    center = (padded_w // 2, padded_h // 2)
    
    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)  # 注意OpenCV中角度是逆时针为负
    
    # 进行旋转，使用0填充空白区域
    rotated = cv2.warpAffine(padded_image.astype(np.float32), M, (padded_w, padded_h),
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0)
    
    return rotated

def get_hilbert_sequence(image, order=3, dimension=2):
    """
    获取图像对应的Hilbert曲线一维序列
    """
    hilbert_curve = HilbertCurve(order, dimension)
    
    hilbert_path = []

    for i in range(2**(order*2)):
        coords = hilbert_curve.point_from_distance(i)
        x, y = coords[0], coords[1]
        if 0 <= x < order and 0 <= y < order:
            hilbert_path.append((x, y))
    
    # 返回Hilbert路径对应的图像像素值
    return [image[y, x] for x, y in hilbert_path]

def visualize_and_save_sequences(original_image, angles=None, show_image=True):
    """
    可视化不同角度旋转后的图像和对应的一维序列，并保存结果
    """
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
            else:
                # 对于任意角度使用新的方法
                rotated_image = rotate_image_arbitrary(original_image, angle)
            
            # 获取一维序列
            sequence = get_hilbert_sequence(rotated_image)
            
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
            for i in range(4):
                for j in range(4):
                    plt.text(j, i, f"{rotated_image[i, j]:.1f}", 
                            ha='center', va='center', color='white')
            plt.grid(False)
            
            # 显示Hilbert曲线路径
            plt.subplot(132)
            hilbert_path = [(coords[0], coords[1]) 
                          for coords in [HilbertCurve(2, 2).point_from_distance(i) 
                                       for i in range(16)]]
            points = np.array(hilbert_path)
            plt.plot(points[:, 1], points[:, 0], 'b-', linewidth=2)
            plt.grid(True)
            plt.title('Hilbert曲线路径')
            for i, (x, y) in enumerate(hilbert_path):
                plt.text(y, x, str(16-i), ha='center', va='center')
            plt.xlim(-0.5, 3.5)
            plt.ylim(-0.5, 3.5)
            
            # 显示一维展开的序列
            plt.subplot(133)
            sequence_image = np.array(sequence).reshape(1, -1)
            plt.imshow(sequence_image, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title(f'旋转{angle}度后的一维序列')
            for i in range(16):
                plt.text(i, 0, f"{sequence[i]:.1f}", 
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
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    visualize_and_save_sequences(original_image, angles=angles, show_image=False)


if __name__ == "__main__":
    main()
