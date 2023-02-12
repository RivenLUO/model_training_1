import keras
import os
from _utils import timeit
import _utils


@timeit
def simple_training(data_left, data_right, data_label,
                    model_function, model_function_args, results_save_dir, val_split, epochs, batch_size):
    """
    Train a siamese model and store results in a folder.

    :param results_save_dir:
    :param data_left: left images array
    :type data_left: np.array
    :param data_right: right images array
    :type data_right: np.array
    :param data_label: labels array
    :type data_label: np.array
    :param model_function: function which build the model to train with k fold
    :type model_function: function
    :param model_function_args: arguments of the model building function
    :type model_function_args: list
    :param folder_path: path of the folder where results are stored
    :type folder_path: str
    :param val_split: set proportion dedicated to the validation set
    :type val_split: float
    :param epochs: number of training epochs
    :type epochs: int
    :param batch_size: batch size
    :type batch_size: int
    :return:
    :rtype:
    """
    # Create folder to store results
    folder_path = _utils.safe_folder_creation(results_save_dir)

    # Build model
    train_data = [data_left, data_right]
    model = model_function(*model_function_args)

    # Train model
    model_history = model.fit(train_data, data_label,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_split=val_split)

    # Save model and plots
    model_path = os.path.join(folder_path, 'model.h5')
    model.save(model_path)
    _utils.plot_model_accuracy(model_history, folder_path)
    _utils.plot_model_loss(model_history, folder_path)

    # Save also weights and structure for backup
    weights_path = os.path.join(folder_path, 'weights.h5')
    model.save_weights(weights_path)
    json_path = os.path.join(folder_path, 'structure.json')
    _utils.save_structure_in_json(model, json_path)


# @timeit
# def k_fold(data_left, data_right, data_label, k, model_function, model_function_args, folder_path, epochs,
#            batch_size):
#     """
#     Execute a K-fold cross validation for a model.
#
#     :param data_left: left images array
#     :type data_left: np.array
#     :param data_right: right images array
#     :type data_right: np.array
#     :param data_label: labels array
#     :type data_label: np.array
#     :param k: number of fold
#     :type k: int
#     :param model_function: function which build the model to train with k fold
#     :type model_function: function
#     :param model_function_args: arguments of the model building function
#     :type model_function_args: list
#     :param folder_path: path of the folder where results are stored
#     :type folder_path: str
#     :param epochs: number of training epochs
#     :type epochs: int
#     :param batch_size: batch size
#     :type batch_size: int
#     """
#
#     # Variable initialization
#     nb_comp = len(data_label)
#     num_val_samples = nb_comp // k
#     all_scores = []
#
#     # Create folder to store results
#     folder_path = Utils.safe_folder_creation(folder_path)
#
#     # Train model for each fold
#     for i in range(k):
#         print('Processing fold #', i)
#
#         # Select validation set
#         val_left = data_left[i * num_val_samples: (i + 1) * num_val_samples]
#         val_right = data_right[i * num_val_samples: (i + 1) * num_val_samples]
#         val_data = [val_left, val_right]
#         val_label = data_label[i * num_val_samples: (i + 1) * num_val_samples]
#
#         # Get the remaining data to create the training set
#         partial_train_left = np.concatenate(
#             [data_left[:i * num_val_samples], data_left[(i + 1) * num_val_samples:]])
#         partial_train_right = np.concatenate(
#             [data_right[:i * num_val_samples], data_right[(i + 1) * num_val_samples:]])
#         partial_train_data = [partial_train_left, partial_train_right]
#         partial_train_label = np.concatenate(
#             [data_label[:i * num_val_samples], data_label[(i + 1) * num_val_samples:]])
#
#         # Build model
#         bk.clear_session()
#         model = model_function(*model_function_args)
#
#         # Train model
#         model_fitted = model.fit(partial_train_data, partial_train_label, validation_data=[val_data, val_label],
#                                  epochs=epochs, batch_size=batch_size, verbose=1)
#
#         # Evaluate model and store result
#         val_mse, val_mae = model.evaluate(val_data, val_label, verbose=1)
#         all_scores.append(val_mae)
#
#         # Save model and plots
#         model_path = os.path.join(folder_path, 'fitted_model_k{}.h5'.format(i + 1))
#         model.save(model_path)
#         plot_validation_info_kfold(model_fitted, i + 1, folder_path)
#
#         # Save also weights and structure for backup
#         weights_path = os.path.join(folder_path, 'weights_k{}.h5'.format(i + 1))
#         model.save_weights(weights_path)
#         json_path = os.path.join(folder_path, 'structure_k{}.json'.format(i + 1))
#         save_structure_in_json(model, json_path)
#
#     # Save K-fold summary in a text file
#     filename = os.path.join(folder_path, "k_fold_report.txt")
#     summary = k_fold_summary(k, model_function, model_function_args, nb_comp, num_val_samples, all_scores, filename)
#     print(summary)

def evaluate_model(x_test, y_test, model_path):
    model = keras.models.load_model(model_path)
    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=128)
    print("test loss, test acc:", results)
