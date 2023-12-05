import os

# 指定图片文件夹的路径
folder_path = 'D:\FYP\dataset\Aug_painting\Smithsonian-7'

# 获取文件夹中的所有图片文件
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]

# 开始从1开始计数并重命名图片文件
for i, image_file in enumerate(image_files, start=568):
    # 构建新的文件名，例如：brush_1.jpg, brush_2.jpg, brush_3.jpg, ...
    file_extension = os.path.splitext(image_file)[1]
    new_filename = f"smithsonian_{i}{file_extension}"

    # 构建旧文件的完整路径和新文件的完整路径
    old_filepath = os.path.join(folder_path, image_file)
    new_filepath = os.path.join(folder_path, new_filename)

    # 重命名文件
    os.rename(old_filepath, new_filepath)

    print(f"重命名 {image_file} 为 {new_filename}")


