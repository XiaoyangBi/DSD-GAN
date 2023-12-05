from PIL import Image
import os

def reverse_colors(image):
    return Image.eval(image, lambda x: 255 - x)

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.gif')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                image = Image.open(input_path)
                inverted_image = reverse_colors(image)
                inverted_image.save(output_path)
                print(f"Processed: {input_path}")
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

if __name__ == "__main__":
    input_folder = "D:/FYP/dataset/train_data/Brush"  # 输入文件夹
    output_folder = "D:\FYP\dataset\Inverted_train_data\Brush"  # 输出文件夹

    process_folder(input_folder, output_folder)
