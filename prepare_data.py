import os
import numpy as np
import cv2
import csv


def load_csv():
    img_paths_array = []
    label = np.zeros((1821, 4))
    # label = np.zeros((250, 4))
    with open("plant-pathology/train.csv") as csv_file:
        # skips header
        next(csv_file)
        reader = csv.reader(csv_file)
        image_dataset_path = os.path.join("plant-pathology", "images")
        row_record = []
        for idx, row_record in enumerate(reader):
            # if idx == 250: break
            img_path = os.path.join(image_dataset_path, row_record[0]+".jpg")
            img_paths_array.append(img_path)
            label[idx] = row_record[1:]
    return np.asarray(img_paths_array), np.asarray(label)


def shuffle_data(images, labels):
    # its ensure same shuffle rate
    np.random.seed(42)
    np.random.shuffle(images)
    np.random.seed(42)
    np.random.shuffle(labels)

    return images, labels


def split_data(raw_data, val_split=0.1, test_split=0.1):
    # simple np.array slicing
    train_data = raw_data[:int(len(raw_data)*(1-val_split-test_split))]
    val_data = raw_data[int(len(raw_data)*(1-val_split-test_split)):int(len(raw_data)*(1-test_split))]
    test_data = raw_data[int(len(raw_data)*(1-test_split)):]

    return train_data, val_data, test_data


def load_images(image_paths: tuple) -> tuple:
    train_paths, val_paths, test_paths = image_paths

    train_images = []
    val_images = []
    test_images = []

    paths = (train_paths, val_paths, test_paths)
    images = (train_images, val_images, test_images)

    for data_subset_paths, data_subset_images in zip(paths, images):
        for i in range(len(data_subset_paths)):
            img = cv2.imread(data_subset_paths[i])
            img_resized = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
            if img_resized.shape[0] != 341 or img_resized.shape[1] != 512:
                img_resized = cv2.resize(img, (512, 341))

            data_subset_images.append(img_resized)

    return np.asarray(train_images), np.asarray(val_images), np.asarray(test_images)


def load_single_image_path(image_path):
    loaded_images = []

    for i in range(len(image_path)):
        img = cv2.imread(image_path[i])
        img_resized = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
        if img_resized.shape[0] != 341 or img_resized.shape[1] != 512:
            img_resized = cv2.resize(img, (512, 341))
        loaded_images.append(img_resized)

    return np.asarray(loaded_images)


def load_some_images(image_path, loaded_amount):
    loaded_images = []

    for i in range(loaded_amount):
        img = cv2.imread(image_path[i])
        img_resized = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
        if img_resized.shape[0] != 341 or img_resized.shape[1] != 512:
            img_resized = cv2.resize(img, (512, 341))
        loaded_images.append(img_resized)

    return np.asarray(loaded_images)





