import os

import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, AveragePooling2D, BatchNormalization
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.models import Sequential


# custom layer for mimicking scaled average pooling
class ScaledPooling(AveragePooling2D):
    def __init__(self):
        super(ScaledPooling, self).__init__()
        # calculate scaling factor based on the pooling size
        # scale_factor = number of entries included in a pooling kernel
        self.scale_factor = self.pool_size[0] * self.pool_size[1]

    def call(self, inputs):
        # mimic scaled average pooling by calling the normal average pooling
        # layer and scaling according to the scaling factor
        return tf.multiply(AveragePooling2D.call(self, inputs), self.scale_factor)


# apply our adjusted quantization method
def adj_bias(bias, BITB, scaling):
    bias = np.tanh(bias)
    bias = quantize(bias, BITB)
    # scaling indicates, whether intermediate results are
    # scaled to integers or remain quantized floats
    if scaling:
        n = float(2 ** BITB - 1)
        return np.round(bias * n)
    else:
        return bias


# apply the DoReFa-quantization method
def adj_weights(weights, BITW, scaling):
    x = tf.tanh(weights)
    x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
    x = 2 * quantize(x, BITW) - 1
    # scaling indicates, whether intermediate results are
    # scaled to integers or remain quantized floats
    if scaling:
        n = float(2 ** BITW - 1)
        return np.round(x * n)
    else:
        return x


# adjust weight quantization from weight dictionary
def adjust_weights_from_dic(wdic, adjw=True, do_scaling=False):
    # auxiliary value for adjusted weights
    adj_wdic = {}

    # quantization precision
    # must match the precision set in the Tensorpack training script
    bitb = 8
    bitw = 8

    # quantize relevant parameters, i.e. parameters that were quantized
    # during training
    for key, value in wdic.items():
        if key.endswith('/W:0'):  # quantize weights
            # adjw defines whether to use fw / adjust_weights or
            # fb / adjust_bias to quantize the weights
            if adjw:
                adj_value = adj_weights(value, bitw, do_scaling)
            else:
                adj_value = adj_bias(value, bitb, do_scaling)
        elif key.endswith('/b:0'):  # quantize bias
            adj_value = adj_bias(value, bitb, do_scaling)
        else:  # don't quantize value
            adj_value = value

        # add adjusted value to new weight dictionary under the
        # same key
        adj_wdic[key] = adj_value

    return adj_wdic


# copy weights from dictionary to provided model
def copy_weights(model, w_dic):
    # loop through all layers of the model
    for layer in model.layers:

        # get the layer's name
        l_name = layer.name

        # copy parameters from dictionary to model
        if 'conv' in l_name or 'fc' in l_name:
            # convolutional or fully connected layer
            saved_weights = w_dic[l_name + '/W:0']

            bias_key = l_name + '/b:0'
            # only if bias_key exists, is the layer using bias
            if bias_key in w_dic.keys():  # bias used
                saved_bias = w_dic[l_name + '/b:0']
                layer.set_weights([saved_weights, saved_bias])
            else:  # bias not used
                layer.set_weights([saved_weights])

        else:
            print('Error: layer not supported: %s' % l_name)
            exit(-1)

    return model


# defining the model later used with the encrypted network
# -> copy network structure from training script
def define_model_yaleface():
    model = Sequential()
    model.add(Conv2D(filters=5, kernel_size=(9, 9), strides=(4, 4), padding='VALID',
                     use_bias=False,
                     input_shape=(192, 168, 1), name="conv0"))
    model.add(ScaledPooling())
    model.add(Activation(square_activation))
    model.add(Conv2D(filters=50, kernel_size=(5, 5), strides=(2, 2), padding='VALID',
                     use_bias=False,
                     name="conv1"))
    model.add(ScaledPooling())
    model.add(Activation(square_activation))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(128, name="fc0",
                    use_bias=False,
                    ))
    model.add(Dropout(0.5))
    model.add(Dense(38, name="fc1"))
    model.add(Activation(square_activation))

    model.compile(
        optimizer='adadelta',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


# load cropped yale face dataset, split into components and normalize if needed
def load_data_yaleB(normalization=False):
    # path to numpy-datasets used and prepared during training
    filename_train_ds = 'yaleB_cropped/train_dataset.npy'
    filename_test_ds = 'yaleB_cropped/test_dataset.npy'
    filename_val_ds = 'yaleB_cropped/validation_dataset.npy'

    if not os.path.exists(filename_train_ds):
        print('Error: dataset file does not exist')
        exit(-1)

    # load datasets
    train_ds = np.load(filename_train_ds, allow_pickle=True)
    test_ds = np.load(filename_test_ds, allow_pickle=True)
    val_ds = np.load(filename_val_ds, allow_pickle=True)

    # # can shuffle datasets here, if desired
    # random.shuffle(train_ds)
    # random.shuffle(val_ds)

    # split datasets into their x and y component
    x_train_ds = []
    y_train_ds = []
    x_val_ds = []
    y_val_ds = []
    x_test_ds = []
    y_test_ds = []
    # divide training dataset
    for entry in train_ds:
        x_train_ds.append(entry[0])
        y_train_ds.append(entry[1])
    # divide validation dataset
    for entry in val_ds:
        x_val_ds.append(entry[0])
        y_val_ds.append(entry[1])
    # divide testing dataset
    for entry in test_ds:
        x_test_ds.append(entry[0])
        y_test_ds.append(entry[1])

    # turn auxiliary variables into numpy arrays
    x_train_ds = np.array(x_train_ds)
    y_train_ds = np.array(y_train_ds)
    x_val_ds = np.array(x_val_ds)
    y_val_ds = np.array(y_val_ds)
    x_test_ds = np.array(x_test_ds)
    y_test_ds = np.array(y_test_ds)

    # normalize x-values if needed
    if normalization:
        x_train_ds = x_train_ds / 255
        x_val_ds = x_val_ds / 255
        x_test_ds = x_test_ds / 255

    return x_train_ds, y_train_ds, x_val_ds, y_val_ds, x_test_ds, y_test_ds


# preparing the weights of the provided model for saving by extracting
# the weights and adding them to a dictionary of numpy arrays
def prepare_weights_for_saving(model_to_save, add_scaling=False, sf_w=1, sf_b=1):
    # auxiliary variable
    tmp_weights = {}

    # loop through model layers
    for layer in model_to_save.layers:
        layer_weights = layer.get_weights()

        if len(layer_weights) == 1:  # no bias used

            # apply scaling to weights if wanted, otherwise just copy values
            if add_scaling:
                tmp_weights[(layer.name + '_weights')] = np.round(layer_weights[0] * sf_w)
            else:
                tmp_weights[(layer.name + '_weights')] = layer_weights[0]

            # as there is no bias used, set bias to None
            tmp_weights[(layer.name + '_bias')] = None

        elif len(layer_weights) == 2:  # bias used

            # apply scaling to weights and bias if wanted, otherwise just copy values
            if add_scaling:
                tmp_weights[(layer.name + '_weights')] = np.round(layer_weights[0] * sf_w)
                tmp_weights[(layer.name + '_bias')] = np.round(layer_weights[1] * sf_b)
            else:
                tmp_weights[(layer.name + '_weights')] = layer_weights[0]
                tmp_weights[(layer.name + '_bias')] = layer_weights[1]

    return tmp_weights


# print the weights of a model layer-by-layer
def print_weights(model):
    # loop through all layers of the model
    for layer in model.layers:
        # print layer config and weights
        print(layer.get_config())
        print(layer.get_weights())


# quantize x according to k
def quantize(x, k):
    n = float(2 ** k - 1)
    return np.round(x * n) / n


# squared activation function to replace ReLU
def square_activation(x):
    return tf.square(x)


# script for applying post-training-adustments to the models trained
# in Tensorpack
if __name__ == "__main__":

    # model with scaled average pooling
    # same model is later used in the encrypted network
    model = define_model_yaleface()

    # load 64-bit datasets, input images are 64x64x3 pixel and normalized
    print('Load data')
    x_train, y_train, x_val, y_val, x_test, y_test = load_data_yaleB(normalization=True)
    print('Done')

    # load tensorpack output weights and quantize them
    print('Load and adjust parameters')
    dic = np.load('yaleface_params.npz')
    adj_dic = adjust_weights_from_dic(dic, adjw=True, do_scaling=False)
    print('Done')

    # copy adjusted weights to model
    print('Copying weights')
    new_model = copy_weights(model, adj_dic)
    print('Done')

    # evaluate model and print out loss and accuracy
    print('Evaluating model')
    # evaluate training accuracy
    score = new_model.evaluate(x_train, y_train, verbose=0)
    print('Train loss:', score[0])
    print('Train accuracy:', score[1])
    # evaluate validation accuracy
    score = new_model.evaluate(x_val, y_val, verbose=0)
    print('Val loss:', score[0])
    print('Val accuracy:', score[1])
    # evaluate testing accuracy
    score = new_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Done')

    # saving the adjusted parameters
    print('Saving adjusted parameters')
    # scaling factor according to bitw and bitb (here both = 8)
    n = 2 ** 8 - 1
    # prepare parameters for saving
    save_weights = prepare_weights_for_saving(new_model, add_scaling=True, sf_w=n, sf_b=n)
    # save parameters
    np.save('yaleparams_scaled.npy', save_weights)
    print('Done')
