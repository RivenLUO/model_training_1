# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------
# import os
# import numpy as np
#
# from hyperas import optim
# from hyperas.distributions import choice, uniform
# from hyperopt import STATUS_OK, tpe, Trials
#
# from keras import Input, Model
# from keras.applications import VGG19
# from keras.layers import concatenate, Conv2D, Dropout, Flatten, Dense, BatchNormalization
# from keras.optimizers import SGD
# from project_scripts._utils import safe_folder_creation, save_structure_in_json
#
# import inspect
# import json
#
#
# # ----------------------------------------------------------------------------------------------------------------------
# # Functions definitions
# # ----------------------------------------------------------------------------------------------------------------------
# def load_data():
#     """
#     Load the data used to train the comparisons model.
#
#     :return: training data and labels
#     :rtype : tuple(np.array)
#     """
#     save_folder = r"E:\thesis MSc Geography\model_training\Data\Sample_web_green"
#     data_left = np.load(os.path.join(save_folder, "train_left.npy"))
#     data_right = np.load(os.path.join(save_folder, "train_right.npy"))
#     data_label = np.load(os.path.join(save_folder, "train_label.npy"))
#
#     data = [data_left, data_right]
#
#     return data, data_label
#
#
# def save_hyperas_results(best_model, best_run, result_folder, data, model):
#     """
#     Create folder and file containing results of the Hyperas hyperparameters optimization.
#
#     :param best_model: best model found
#     :type best_model: keras.Model
#     :param best_run: dictionary of parameters of the best model
#     :type best_run: dict
#     :param result_folder:
#     :type result_folder:
#     :param data: loading data function
#     :type data: function
#     :param model: creation model function
#     :type model: function
#     """
#
#     # Create result folder
#     result_folder = safe_folder_creation(result_folder)
#
#     # Save best model
#     best_model.save(os.path.join(result_folder, "best_model.h5"))
#
#     # Save also weights and structure for backup
#     weights_path = os.path.join(result_folder, 'weights.h5')
#     best_model.save_weights(weights_path)
#     json_path = os.path.join(result_folder, 'structure.json')
#     save_structure_in_json(best_model, json_path)
#
#     # Save model function an used data in a text file
#     s = "Grid search parameters Hyperas \n"
#     s += "\nData function code:\n\n" + inspect.getsource(data)
#     s += "\nModel source code:\n\n" + inspect.getsource(model)
#     s += "\nHyperas results:\n\n" + json.dumps(best_run)
#
#     # Save in a file
#     with open(os.path.join(result_folder, 'Hyperas_params.txt'), "w+") as f:
#         f.write(s)
#
#
# def model(data, data_label):
#     """
#     Defines the comparisons model, all hyperparameters in double brackets will be optimize by Hyperas.
#     :return: a dictionary with following keys :
#                 - loss : the metrics function to be minimized by Hyperopt.
#                 - status : a boolean that tells if everything went fine.
#                 - model : the model on which hyperparameters optimization occurs.
#     """
#     img_size = 224
#     vgg_feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
#     for layer in vgg_feature_extractor.layers[:-4]:
#         layer.trainable = False
#
#     img_a = Input(shape=(img_size, img_size, 3), name="left_image")
#     img_b = Input(shape=(img_size, img_size, 3), name="right_image")
#
#     out_a = vgg_feature_extractor(img_a)
#     out_b = vgg_feature_extractor(img_b)
#
#     concat = concatenate([out_a, out_b])
#
#     x = Conv2D({{choice([64, 128, 256, 512])}}, (3, 3), activation='relu', padding='same', name="Conv_1")(concat)
#     x = Dropout({{uniform(0, 0.5)}}, name="Drop_1")(x)
#     x = Conv2D({{choice([64, 128, 256, 512])}}, (3, 3), activation='relu', padding='same', name="Conv_2")(x)
#     x = Dropout({{uniform(0, 0.5)}}, name="Drop_2")(x)
#     x = Conv2D({{choice([64, 128, 256, 512])}}, (3, 3), activation='relu', padding='same', name="Conv_3")(x)
#     x = BatchNormalization()(x)
#     x = Flatten()(x)
#     x = Dense(2, activation='softmax', name="Dense_Final")(x)
#
#     comparisons_model = Model([img_a, img_b], x)
#
#     sgd = SGD(lr={{choice([1e-4, 1e-5, 1e-6])}}, decay={{choice([1e-4, 1e-5, 1e-6])}}, momentum={{uniform(0, 0.9)}},
#               nesterov=True)
#     comparisons_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#
#     result = comparisons_model.fit(
#         [data[0], data[1]],
#         data_label,
#         batch_size=16,
#         epochs=30,
#         validation_split=0.2)
#
#     validation_acc = np.amax(result.history['val_acc'])
#     print('Best validation acc of epoch:', validation_acc)
#
#     return {'loss': -validation_acc, 'status': STATUS_OK, 'model': comparisons_model}
#
#
# if __name__ == "__main__":
#     # ------------------------------------------------------------------------------------------------------------------
#     # Variables initialization
#     # ------------------------------------------------------------------------------------------------------------------
#     result_folder = r'D:\Guillaume\Ottawa\Data\Training_Models_Results\Comparisons_Trueskill\08_27_hyperas'
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # Run functions
#     # ------------------------------------------------------------------------------------------------------------------
#     best_run, best_model = optim.minimize(model=model,
#                                           data=load_data,
#                                           algo=tpe.suggest,
#                                           max_evals=1,
#                                           trials=Trials())
#     # Print results
#     print("Best performing model chosen hyper-parameters:")
#     print(best_run)
#     # Save results
#     save_hyperas_results(best_model, best_run, result_folder, load_data, model)
import os
import csv
import keras_tuner as kt
import keras
from project_scripts import models
import numpy as np

tuner = kt.BayesianOptimization(
    models.comparison_model(224, ),
    objective="val_accuracy",
    max_trials=100,
    executions_per_trial=2,
    directory="mnist_kt_test",
    overwrite=True,
)

folder_dir = r"E:\thesis MSc Geography\model_training\Data\Sample_web_Green_224"
x_left_training = np.load(os.path.join(folder_dir, "train_left_duel_2.npy"), allow_pickle=True)
x_right_training = np.load(os.path.join(folder_dir, "train_right_duel_2.npy"), allow_pickle=True)
y_training = np.load(os.path.join(folder_dir, "train_label_duel_2.npy"), allow_pickle=True)

num_val_samples = 1000
x_left_train, x_left_val = x_left_training[:-num_val_samples], x_left_training[-num_val_samples:]
x_right_train, x_right_val = x_right_training[:-num_val_samples], x_right_training[-num_val_samples:]

x_train = [x_left_train,x_right_train]
x_val = [x_left_val,x_right_val]
y_train, y_val = y_training[:-num_val_samples], y_training[-num_val_samples:]

callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5), ]

search_history = tuner.search(
    x_train,
    y_train,
    batch_size=8,
    epochs=50,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    verbose=2,
)

top_n = 4
best_hps = tuner.get_best_hyperparameters(top_n)

search_results = []
for i, model in enumerate(search_history):
    search_results.append(model.history.history)

with open("tuner_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(search_results)

with open('data.csv', 'w', newline='') as file:
    # Create a writer object
    writer = csv.writer(file)
    # Write the rows to the CSV file
    for row in best_hps:
        writer.writerow(row)

print(best_hps)
