from keras import layers, optimizers
from keras.applications import VGG19,vgg19
import keras


def comparison_model(img_size, weights=None):
    """

    :param img_size:
    :param weights:
    :return:
    """
    # Extracting features from VGG19 pretrained with 'imagenet'
    feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

    # Fine-tuning by freezing the last 4 convolutional layers of VGG19 (last block)
    for layer in feature_extractor.layers[:-4]:
        layer.trainable = False

    # Definition of the 2 inputs
    img_a = layers.Input(shape=(img_size, img_size, 3), name="left_image")
    img_b = layers.Input(shape=(img_size, img_size, 3), name="right_image")

    # Convert RGB to BGR
    img_a = vgg19.preprocess_input(img_a)
    img_b = vgg19.preprocess_input(img_b)

    out_a = feature_extractor(img_a)
    out_b = feature_extractor(img_b)

    # Concatenation of the inputs
    concat = layers.concatenate([out_a, out_b])

    # Add convolution layers on top
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name="Conv_1")(concat)
    x = layers.Dropout(0.43, name="Drop_1")(x)
    # x = BatchNormalization()(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name="Conv_2")(x)
    x = layers.Dropout(0.49, name="Drop_2")(x)
    # x = BatchNormalization()(x)
    x = layers.Flatten()(x)
    output = layers.Dense(2, activation='softmax', name="Final_dense")(x)

    comparison_model = keras.Model([img_a, img_b], output)

    # Loading already weights
    if weights:
        comparison_model.load_weights(weights)

    # Optimizer selection
    sgd = optimizers.SGD(learning_rate=1e-5, decay=1e-6, momentum=0.695, nesterov=True)
    comparison_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return comparison_model
