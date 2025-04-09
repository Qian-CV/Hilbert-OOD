import os
from PIL import Image
import numpy as np

def create_gif_from_pngs(folder_path, output_gif_path, duration=800):
    """
    从指定文件夹中的PNG图像生成GIF动图。
    
    :param folder_path: 包含PNG图像的文件夹路径
    :param output_gif_path: 输出GIF文件的路径
    :param duration: 每帧的持续时间（毫秒）
    """
    # 获取文件夹中的所有PNG文件
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    png_files.sort()  # 按文件名排序

    images = []
    for png_file in png_files:
        image_path = os.path.join(folder_path, png_file)
        images.append(Image.open(image_path))

    # 保存为GIF动图
    images[0].save(output_gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)

# 计算两个向量之间的欧几里得距离
def euclidean_distance(seq1, seq2):
    return np.sqrt(np.sum((np.array(seq1) - np.array(seq2)) ** 2))

# 计算两个向量之间的内积
def inner_product(seq1, seq2):
    return np.dot(seq1, seq2)

# 计算两个向量之间的余弦相似度
def cosine_similarity(seq1, seq2):
    return np.dot(seq1, seq2) / (np.linalg.norm(seq1) * np.linalg.norm(seq2))

# 计算两个向量之间的曼哈顿距离
def manhattan_distance(seq1, seq2):
    return np.sum(np.abs(np.array(seq1) - np.array(seq2)))

# 计算两个向量之间的切比雪夫距离
def chebyshev_distance(seq1, seq2):
    return np.max(np.abs(np.array(seq1) - np.array(seq2)))

# 计算两个向量之间的杰卡德相似度    
def jaccard_similarity(seq1, seq2):
    intersection = set(seq1) & set(seq2)
    union = set(seq1) | set(seq2)
    return len(intersection) / len(union)

# 计算两个向量之间的汉明距离
def hamming_distance(seq1, seq2):
    return np.sum(np.array(seq1) != np.array(seq2))


if __name__ == "__main__":
    create_gif_from_pngs('G:/PhD_file/co_work/Hilbert-MQ+WLQ-nips/2实验数据/2_hilbert旋转等变-任意角度/hilbert_visualization', 'G:/PhD_file/co_work/Hilbert-MQ+WLQ-nips/2实验数据/2_hilbert旋转等变-任意角度/hilbert_visualization/hilbert_visualization.gif')
