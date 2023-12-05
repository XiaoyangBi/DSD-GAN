from PIL import Image
import os

# 输入文件夹和输出文件夹的路径
input_folder = "D:\FYP\dataset\painting_original"
output_folder = "D:\FYP\dataset\painting_ad512"

# 目标尺寸
target_size = (512, 512)  # 或 (1024, 1024) 或其他尺寸

# 确保输出文件夹存在，如果不存在则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 列出输入文件夹中的所有文件
files = os.listdir(input_folder)

# 遍历每个文件并调整大小
for file in files:
    try:
        # 打开图像文件
        img = Image.open(os.path.join(input_folder, file))

        # 调整图像大小
        img = img.resize(target_size, Image.ANTIALIAS)

        # 保存调整大小后的图像到输出文件夹
        img.save(os.path.join(output_folder, file))

        print(f"已处理 {file}")
    except Exception as e:
        print(f"处理 {file} 时出错: {e}")

print("完成批量调整大小操作")
