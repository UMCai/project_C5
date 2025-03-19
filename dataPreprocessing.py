import logging
import numpy as np
import os
from pathlib import Path
import sys
import tempfile
import torch
import glob
import json
import random

from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.engines import SupervisedTrainer
from monai.handlers import StatsHandler
from monai.inferers import SimpleInferer
from monai.networks import eval_mode
from monai.networks.nets import densenet121
from monai.transforms import LoadImageD, EnsureChannelFirstD, ScaleIntensityD, Compose

# to download the dataset
root_dir = "./data"
assert os.path.exists(root_dir), "Data not found. Please download the dataset."
transform = Compose(
    [
        LoadImageD(keys="image", image_only=True),
        EnsureChannelFirstD(keys="image"),
        ScaleIntensityD(keys="image"),
    ]
)
dataset = MedNISTDataset(root_dir=root_dir, transform=transform, section="training", download=True, cache_rate=0.0)

# to split the data into three potions
# pretrain 40%, train 30%, test 30%
data_path = 'data/MedNIST'
assert os.path.exists(data_path), "Data not found. Please download the dataset."

# Create a dictionary to store file paths for each class
class_file_paths = {}
for class_dir in os.listdir(data_path):
    class_path = os.path.join(data_path, class_dir)
    if os.path.isdir(class_path):
        class_file_paths[class_dir] = glob.glob(os.path.join(class_path, "*"))

# Print or save the dictionary
print("all classes are:", class_file_paths.keys())
print("for each class:", {k: len(v) for k, v in class_file_paths.items()})
print("Total number of classes:", len(class_file_paths))


# Optionally, save the dictionary to a JSON file
with open(os.path.join(data_path, "data_split.json"), "w") as f:
    json.dump(class_file_paths, f, indent=4)

# Load the data split from the JSON file
data_split_path = os.path.join(data_path, "data_split.json")
with open(data_split_path, 'r') as f:
    data_split = json.load(f)

# split the data into three portions

# Split the data into pretrain, train, and test sets
split_ratios = [0.4, 0.3, 0.3]  # Pretrain: 40%, Train: 30%, Test: 30%
data_splits = {"pretrain": {}, "train": {}, "test": {}}

for class_name, file_paths in class_file_paths.items():
    random.shuffle(file_paths)  # Shuffle the file paths randomly
    total_files = len(file_paths)
    
    # Calculate split indices
    pretrain_end = int(total_files * split_ratios[0])
    train_end = pretrain_end + int(total_files * split_ratios[1])
    
    # Assign files to each split
    data_splits["pretrain"][class_name] = file_paths[:pretrain_end]
    data_splits["train"][class_name] = file_paths[pretrain_end:train_end]
    data_splits["test"][class_name] = file_paths[train_end:]

# Print the number of files in each split for verification
print("Pretrain split:", {k: len(v) for k, v in data_splits["pretrain"].items()})
print("Train split:", {k: len(v) for k, v in data_splits["train"].items()})
print("Test split:", {k: len(v) for k, v in data_splits["test"].items()})

# Translate 'class_name': [file_paths] into data_dicts [{'image': file_path, 'label': class_name}]
data_dicts = {"pretrain": [], "train": [], "test": []}

for split in ["pretrain", "train", "test"]:
    for class_name, file_paths in data_splits[split].items():
        for file_path in file_paths:
            data_dicts[split].append({"image": file_path, "label": class_name})

# Print the number of data_dicts in each split for verification
print("Pretrain data_dicts:", len(data_dicts["pretrain"]))
print("Train data_dicts:", len(data_dicts["train"]))
print("Test data_dicts:", len(data_dicts["test"]))

# Optionally, save the data_dicts to JSON files
with open(os.path.join(data_path, "pretrain_data_dicts.json"), "w") as f:
    json.dump(data_dicts["pretrain"], f, indent=4)
with open(os.path.join(data_path, "train_data_dicts.json"), "w") as f:
    json.dump(data_dicts["train"], f, indent=4)
with open(os.path.join(data_path, "test_data_dicts.json"), "w") as f:
    json.dump(data_dicts["test"], f, indent=4)