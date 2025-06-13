import os
import random

dataset_folder = "DATA/crop_400/labels"
train_file = 'DATA/crop_400/train.txt'
val_file = 'DATA/crop_400/test.txt'

train_ratio = 0.8
val_ratio = 0.2

image_files = [f for f in os.listdir(dataset_folder) if f.endswith('.tif')]

random.shuffle(image_files)
           
total_images = len(image_files)
num_train = int(total_images * train_ratio)
num_val = total_images - num_train

train_images = image_files[:num_train]
val_images = image_files[num_train:]

with open(train_file, 'w') as f:
    for image in train_images:
        f.write(image + '\n')

with open(val_file, 'w') as f:
    for image in val_images:
        f.write(image + '\n')

