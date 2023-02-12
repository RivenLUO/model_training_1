from PIL import Image
import tensorflow as tf
import os
import numpy as np


def resize(folder_dir, save_dir, resize_width):
    """
    :param folder_dir:
    :param save_dir:
    :param resize_width:
    :return:
    """
    for filename in os.listdir(folder_dir):
        if filename.endswith(".jpg"):
            with Image.open(os.path.join(folder_dir, filename)) as image_ori:
                image_reshp = image_ori.resize((resize_width, resize_width), resample=Image.BICUBIC)
                output_path = os.path.join(save_dir,
                                           os.path.splitext(filename)[0] + '_' + str(resize_width) + 'x' + str(resize_width) + '.jpg')
                image_reshp.save(output_path)


def convert_to_array(folder_dir, save_dir):
    """
    Converting the JPG images under a specific folder to numpy files

    :param save_dir:
    :param folder_dir:
    :return:
    """
    for filename in os.listdir(folder_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder_dir, filename)
            img = tf.keras.preprocessing.image.load_img(image_path)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
            np.save(os.path.join(save_dir, os.path.splitext(filename)[0]), tensor)
