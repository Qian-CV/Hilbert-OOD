from utile import *
import re
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def calculate_distance(seq1, seq2, distance_type):
    """
    计算两个序列之间的距离
    method: 'euclidean' 或 'inner_product'
    """
    # 将字符串转换为列表
    seq1 = [int(x) for x in seq1.strip('[]').split(',')]
    seq2 = [int(x) for x in seq2.strip('[]').split(',')]
    
    # 确保两个序列长度相同
    min_len = min(len(seq1), len(seq2))
    seq1 = seq1[:min_len]
    seq2 = seq2[:min_len]

    if distance_type == 'euclidean':
        return euclidean_distance(seq1, seq2)
    elif distance_type == 'inner_product':
        return inner_product(seq1, seq2)
    else:
        raise ValueError("不支持的距离计算方法")

def read_sequences_from_file(filename):
    """
    从文件中读取序列和角度
    """
    angles = []
    sequences = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        if "逆时针旋转角度" in lines[i]:
            # 提取角度
            angle_match = re.search(r'(\d+)度', lines[i])
            if angle_match:
                angle = int(angle_match.group(1))
                angles.append(angle)
                
                # 提取序列
                if i + 1 < len(lines) and "一维序列:" in lines[i + 1]:
                    seq_match = re.search(r'一维序列: (\[.*?\])', lines[i + 1])
                    if seq_match:
                        sequence = seq_match.group(1)
                        sequences.append(sequence)
        i += 1
    
    return angles, sequences

def plot_distance_vs_angle(angles, sequences, method='euclidean'):
    """
    绘制距离与旋转角度的关系图
    """
    # 计算所有角度对之间的距离
    distances = []
    angle_pairs = []
    
    for i in range(len(angles)):
        for j in range(i + 1, len(angles)):
            distance = calculate_distance(sequences[i], sequences[j], method)
            distances.append(distance)
            angle_pairs.append((angles[i], angles[j]))
    
    # 绘制距离与旋转角度的关系
    plt.figure(figsize=(12, 6))
    
    # 绘制距离值
    plt.plot(range(len(distances)), distances, marker='o', linestyle='-', linewidth=2, markersize=8)
    
    # 设置x轴标签
    plt.xticks(range(len(distances)), [f"{angle1}°-{angle2}°" for angle1, angle2 in angle_pairs], rotation=45)
    
    # 设置标题和标签
    plt.title(f'序列之间的距离与旋转角度的关系 ({method})', fontsize=14)
    plt.xlabel('角度对', fontsize=12)
    plt.ylabel('距离', fontsize=12)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(f'distance_vs_angle_{method}.png', dpi=300)
    plt.show()
    
    # 打印距离值
    print(f"\n{method}距离值:")
    for (angle1, angle2), distance in zip(angle_pairs, distances):
        print(f"{angle1}°-{angle2}°: {distance:.4f}")

if __name__ == "__main__":
    # 读取序列和角度
    angles, sequences = read_sequences_from_file('hilbert_sequences_oriented.txt')
    
    # 绘制欧几里得距离与旋转角度的关系
    plot_distance_vs_angle(angles, sequences, method='euclidean')
    
    # 绘制内积与旋转角度的关系
    plot_distance_vs_angle(angles, sequences, method='inner_product')
