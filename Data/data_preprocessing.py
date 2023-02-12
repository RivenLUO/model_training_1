from tqdm import tqdm
from project_scripts import image_processing
import numpy as np
import os

"""
    Run codes below after annotation.
"""

# Process raw JPG images------------------------------------------------------------------------------------------------
folder_dir = "Sample_web_green"
save_dir = "Sample_web_Green_224"
image_processing.resize(folder_dir, save_dir, resize_width=224)
image_processing.convert_to_array(save_dir, save_dir)

# Generate datasets for duel comparison model---------------------------------------------------------------------------
folder_dir = "../Data/Sample_web_Green_224"
duel_question_csv_path = '/Data/Sample_web_green/duels_question_1.csv'

duel_question_array = np.genfromtxt(duel_question_csv_path, delimiter=',', dtype=None, encoding=None)

train_left = np.zeros((len(duel_question_array), 224, 224, 3))
train_right = np.zeros((len(duel_question_array), 224, 224, 3))
train_label = np.zeros((len(duel_question_array), 2))

for i in tqdm(range(len(duel_question_array))):
    point = 0
    for filename in os.listdir(folder_dir):
        if filename.endswith(".npy"):
            image_tensor = np.load(os.path.join(folder_dir, filename))
            if duel_question_array[i, 0] in filename:
                train_left[i] = image_tensor
                point += 1
            if duel_question_array[i, 1] in filename:
                train_right[i] = image_tensor
                point += 1
        if point == 2:
            if duel_question_array[i, 2] == 'left':
                train_label[i] = [1, 0]
            if duel_question_array[i, 2] == 'right':
                train_label[i] = [0, 1]
            if duel_question_array[i, 2] == 'No preference':
                train_label[i] = [0, 0]
            break

np.save(os.path.join(folder_dir, "train_left_duel_1"), train_left)
np.save(os.path.join(folder_dir, "train_right_duel_1"), train_right)
np.save(os.path.join(folder_dir, "train_label_duel_1"), train_label)

