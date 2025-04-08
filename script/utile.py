import os
from PIL import Image

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

if __name__ == "__main__":
    create_gif_from_pngs('hilbert_visualization', 'hilbert_visualization/hilbert_visualization.gif')
