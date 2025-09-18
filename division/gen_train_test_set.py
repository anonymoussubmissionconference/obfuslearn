from PIL import Image
import numpy as np
import os
from pathlib import Path
import math
from collections import Counter
import csv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import torch

def annotation_file_train_test(image_dir, valid_size, target_dir, random_seed=42):
    file_paths=[]
    lables = []
    os.makedirs(target_dir, exist_ok=True)
    train_file_name = os.path.join(target_dir,  "train"+str(valid_size)+".csv")
    test_file_name = os.path.join(target_dir, "test" + str(valid_size)+".csv")
    class_weight_name = os.path.join(target_dir, "weights" + str(valid_size)+".csv")
    for label, subdirectory in enumerate(os.listdir(image_dir)):
        subdirectory_path = os.path.join(image_dir, subdirectory)
        if os.path.isdir(subdirectory_path) and os.listdir(subdirectory_path):
            for filename in os.listdir(subdirectory_path):
                file_path = os.path.join(subdirectory, filename)
                file_paths.append(file_path)
                lables.append(label)

    data = list(zip(file_paths, lables))
    train_indices, test_indices = train_test_split(list(range(len(data))), test_size=valid_size, random_state=random_seed)

    train_dataset = Subset(data, train_indices)
    test_dataset = Subset(data, test_indices)
    train_labels = [label for _, label in train_dataset]
    counted_train_labels = Counter(train_labels)
    sorted_train_labels = dict(sorted(counted_train_labels.items(), key=lambda x: x[0]))
    #print(sorted_train_labels)
    total_samples = sum(sorted_train_labels.values())
    class_weights = [total_samples / count for count in sorted_train_labels.values()]
    weights = [class_weights[target] for _, target in train_dataset]
    print(np.array(weights).shape)
    test_labels = [label for _, label in test_dataset]
    counted_test_labels = Counter(test_labels)
    sorted_test_labels = sorted(counted_test_labels.items(), key=lambda x:x[0])
    #print(sorted_test_labels)
    with open(train_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filepath', 'label'])
        writer.writerows(train_dataset)
    with open(test_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filepath', 'label'])
        writer.writerows(test_dataset)

    with open(class_weight_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['sample_probability'])
        writer.writerows(zip(weights))

annotation_file_train_test(r'../dataset/malimg',  0.1, r'./malimg')


