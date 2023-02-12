import numpy as np
import os
from project_scripts import training, models

if __name__ == "__main__":
    # data preparing
    width = 224
    folder_dir = r"E:\thesis MSc Geography\model_training\Data\Sample_web_green"
    x_left_training = np.load(os.path.join(folder_dir, "train_left_2.npy"), allow_pickle=True)
    x_right_training = np.load(os.path.join(folder_dir, "train_right_2.npy"), allow_pickle=True)
    y_training = np.load(os.path.join(folder_dir, "train_label_2.npy"), allow_pickle=True)

    # training
    model_save_dir = r"/training_test/script_test-23_1_19"
    training.simple_training(x_left_training, x_right_training, y_training,
                             models.comparison_model, [width,], model_save_dir, val_split=0.2, epochs=50, batch_size=8)


