import numpy as np
import matplotlib.pyplot as plt
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

def get_hilbert_sequence(image):
    """
    获取图像对应的Hilbert曲线一维序列
    """
    p = 2  # 阶数
    N = 2  # 维度
    hilbert_curve = HilbertCurve(p, N)
    
    hilbert_path = []
    for i in range(16):
        coords = hilbert_curve.point_from_distance(i)
        x, y = coords[0], coords[1]
        if 0 <= x < 4 and 0 <= y < 4:
            hilbert_path.append((x, y))
    
    return [image[x, y] for x, y in hilbert_path]

def visualize_and_save_sequences(original_image, angles=[0, 90, 180, 270]):
    """
    可视化不同角度旋转后的图像和对应的一维序列，并保存结果
    """
    # 打开文件准备写入结果，指定编码为utf-8
    with open('hilbert_sequences.txt', 'w', encoding='utf-8') as f:
        for angle in angles:
            # 旋转图像
            rotated_image = rotate_image(original_image, angle)
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
                    plt.text(j, i, str(rotated_image[i, j]), 
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
                plt.text(i, 0, str(sequence[i]), 
                        ha='center', va='center', color='white')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()


def main():
    # 创建4x4的原始图像
    original_image = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])

    # 运行可视化和保存
    visualize_and_save_sequences(original_image, angles=[0, 90, 180, 270])


if __name__ == "__main__":
    main()
