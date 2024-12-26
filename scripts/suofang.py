from PIL import Image
import os


def resize_images(input_folder, output_folder, target_size=(128, 128)):
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 检查文件是否为图像文件
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # 打开图像文件
            with Image.open(os.path.join(input_folder, filename)) as img:
                # 缩放图像
                img_resized = img.resize(target_size, Image.ANTIALIAS)
                # 保存缩放后的图像到输出文件夹
                img_resized.save(os.path.join(output_folder, filename))


# 调用函数并指定输入和输出文件夹路径
input_folder = "E:/caoyuzhu/Medical-Transformer-main/Test/img"
output_folder = "E:/caoyuzhu/Medical-Transformer-main/Test/imgs"
resize_images(input_folder, output_folder)
