import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

input_root = r"D:\tuan_cuoi_ne\assets\VN-celeb"
output_root = "train_data"

# Khởi tạo device đúng cách
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def process_all_images(input_root, output_root):

    for person_name in tqdm(os.listdir(input_root)):
        person_folder = os.path.join(input_root, person_name)
        if not os.path.isdir(person_folder):
            continue
        output_folder = os.path.join(output_root, person_name)
        os.makedirs(output_folder, exist_ok=True)

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
def process_all_images(input_root, output_root):
    for person_name in tqdm(os.listdir(input_root)):
        person_folder = os.path.join(input_root, person_name)
        if not os.path.isdir(person_folder):
            continue
        output_folder = os.path.join(output_root, person_name)
        os.makedirs(output_folder, exist_ok=True)

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)

            # Đọc ảnh
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Cannot read image {img_path}")
                continue
            face_resized = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
            save_path = os.path.join(output_folder, img_name)
            cv2.imwrite(save_path, face_resized)

if __name__ == "__main__":
    process_all_images(input_root, output_root)
