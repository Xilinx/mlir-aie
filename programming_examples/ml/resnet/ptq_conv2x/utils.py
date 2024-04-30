import json
import cv2
import numpy as np

def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="latin1")
    return dict


def load_class_label(class_label_file: str, num_classes: int) -> list:
    class_label = json.load(open(class_label_file))
    class_label_list = [class_label[str(i)] for i in range(num_classes)]

    return class_label_list


def extract_cifar():
    data_batch_1 = unpickle(datafile)
    metadata = unpickle(metafile)

    images = data_batch_1["data"]
    labels = data_batch_1["labels"]
    images = np.reshape(images, (10000, 3, 32, 32))

    import os

    dirname = "cifar_images"
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    # Extract and dump first 10 images
    for i in range(0, 100):
        im = images[i]
        im = im.transpose(1, 2, 0)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        im_name = f"./cifar_images/image_{i}.png"
        cv2.imwrite(im_name, im)