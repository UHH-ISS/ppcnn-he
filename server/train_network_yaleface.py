import random
import time
import os

import cv2
import numpy as np
import tensorflow as tf
import tensorpack as tp
from tensorpack import *
from tensorpack.tfutils import summary
from tensorpack.tfutils.varreplace import remap_variables
from tensorpack.train.model_desc import ModelDesc
from tensorpack.utils import logger

from dorefa import get_dorefa

# variables defining the quantization precision
BITB = 8
BITW = 8
BITA = 2  # only auxiliary, takes no effect
BITG = 6  # only auxiliary, takes no effect
# auxiliary variables defining the network's input and output size
IMAGE_SIZE_HEIGHT = 192
IMAGE_SIZE_WIDTH = 168
NUM_CLASSES = 38


# mimic scaled average pooling
# default scale = 4, as default pooling kernel = (2, 2)
# return: scaled input
def scaleAvgPool(x, scale=4):
    return tf.multiply(x, scale)


# squared activation function to replace ReLU
# Note: we do not need to define a custom gradient, as we are only using
# simple tensorflow functions. Therefore, tensorflow will be able to
# automatically handle the gradient propagation.
# return: squared input
def square_activation(x):
    return tf.square(x)


# our adapted version of the DoReFa-net approach implemented in
# tensorpack using only tanh
# return: our adapted quantization function
def get_fb(bitB):
    # quantization function (same as DoReFa)
    def quantize(x, k):
        n = float(2 ** k - 1)

        @tf.custom_gradient
        def _quantize(x):
            return tf.round(x * n) / n, lambda dy: dy

        return _quantize(x)

    # out adapted quantization method
    def fb(x):
        # limit value range to [-1, 1]
        x = tf.tanh(x)
        # quantize x according to the defined precision
        return quantize(x, bitB)

    return fb


# class describing the desired network structure and functionality
# an instance of this class will be handed to the tensorpack
# TrainConfig for training purposes
#
# the construction of this class and its methods is based on the
# various examples provided in the tensorpack source code
class BaselineModel(ModelDesc):

    # define meta-info of all network inputs
    def inputs(self):
        return [
            # shape of network inputs -> x
            tf.placeholder(tf.float32, (None, IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH, 1), 'input'),
            # shape of expected outputs -> y
            tf.placeholder(tf.int32, (None, NUM_CLASSES), 'label')
        ]

    # builds the network graph and specifies how the cost function should be
    # calculated
    # takes as input tensors that match the definition of inputs() above
    # return: total cost to be optimized
    def build_graph(self, image, label):

        # get the quantization functions
        # DoReFa-approach
        fw, fa, fg = get_dorefa(BITW, BITA, BITG)
        # our approach defined above
        fb = get_fb(BITB)

        # "monkey-patch" tf.get_variable to apply quantization
        # return: quantized variable if quantization should be
        #           applied and unaltered variable otherwise
        def new_get_variable(v):

            # get variable name
            name = v.op.name

            # define which variables should be quantized and how it
            # needs to be adapted, if different quantization variations
            # or combinations are desired
            if name.endswith('W'):  # layer weights
                logger.info("Quantizing weight {}".format(v.op.name))
                return fw(v)
            elif name.endswith('b'):  # layer bias
                logger.info("Quantizing bias {}".format(v.op.name))
                return fb(v)
            else:
                logger.info("NOT quantizing {}".format(v.op.name))
                return v

        # applying the "monkey-patch" for the network
        with remap_variables(new_get_variable):

            # can add parameter choices for all instances of a class with
            # argscope here
            #
            # kept argscope here as an example of how it could work, although
            # we have commented-out possible generalizations, to show how it
            # would work as we sometimes used this option while designing the
            # networks
            with argscope(Conv2D,
                          # activation=square_activation,
                          # use_bias=False,
                          # kernel_initializer=tf.initializers.glorot_uniform
                          ):
                # use logits and LinearWrap to define the network structure
                logits = (LinearWrap(image)
                          .Conv2D('conv0', filters=5, kernel_size=(9, 9), stride=(4, 4),
                                  use_bias=False,
                                  padding='VALID')
                          .AvgPooling('pool0', pool_size=(2, 2))
                          .apply(square_activation)
                          .Conv2D('conv1', filters=50, kernel_size=(5, 5), stride=(2, 2),
                                  use_bias=False,
                                  padding='VALID')
                          .AvgPooling('pool1', pool_size=(2, 2))
                          .apply(square_activation)
                          .Dropout('dropout0', rate=0.3)
                          .FullyConnected('fc0', 128, use_bias=False)
                          .Dropout('dropout1', rate=0.5)
                          .FullyConnected('fc1', NUM_CLASSES)
                          .apply(square_activation)()
                          )

                # calculate total_cost of the network which will be used for optimization

                # cast labels to float values -> only needed if labels are set up as int
                label_cast = tf.cast(label, tf.float32)
                # calculate categorical cross entropy loss
                cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_cast)
                cost = tf.reduce_mean(cost, name='categorical_cross_entropy_loss')

                # convert one-hot encoded label to integers
                converted_label = tf.argmax(label, 1)
                # calculate number of correct classifications / accuracy
                correct = tf.cast(tf.nn.in_top_k(predictions=logits, targets=converted_label, k=1), tf.float32,
                                  name='correct')
                accuracy = tf.reduce_mean(correct, name='accuracy')
                # calculate training error
                train_error = tf.reduce_mean(1 - correct, name='train_error')

                # monitor training error and accuracy (as moving averages)
                # values are automatically logged and printed after each epoch
                # See tutorial at https://tensorpack.readthedocs.io/tutorial/summary.html
                summary.add_moving_summary(train_error, accuracy)

                # applying regularization techniques
                # here: adding weight decay to the weights of the fully connected layers
                wd_cost = tf.multiply(1e-5,
                                      tp.regularize_cost('fc.*/W', tf.nn.l2_loss),
                                      name='regularize_loss')

                # combining categorical cross entropy loss and weight decay costs
                # to get total cost
                total_cost = tf.add_n([wd_cost, cost], name='total_cost')

                # monitor costs
                summary.add_moving_summary(cost, wd_cost, total_cost)

                # monitor histogram of all weights
                summary.add_param_summary(('.*/W', ['histogram', 'rms']))

                return total_cost

    # define what optimizer to use for training
    def optimizer(self):

        # apply exponential decay to the learning rate (lr)
        lr_schedule = tf.train.exponential_decay(
            learning_rate=1.0,  # initial learning rate
            global_step=get_global_step_var(),
            # adapt decay steps to the desired value
            decay_steps=5000,  # 10000, 1000, 5000
            decay_rate=0.96,
            staircase=True,  # decay lr at discrete intervals if true
            name='learning_rate'
        )

        # monitor learning rate
        tf.summary.scalar('lr', lr_schedule)

        # here: use Adadelta optimizer with learning rate schedule
        return tf.train.AdadeltaOptimizer(learning_rate=lr_schedule)


# prepare data for tensorpack usage
# return: dataflow for train-, test-, and validation-dataset
def get_data(ds_name, db_src_dir, batchsize, normalized=False, grayscale=False):

    # return: numpy datasets
    def load_data(src_dir, onehot=True, norm=False):

        # fill auxiliary dataset variables
        def fill_dataset(mode, path, label, do_normalization):

            # loop through all images at path-location
            for img in os.listdir(path):

                try:
                    if grayscale:
                        # load image as numpy array
                        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    else:
                        # load image as numpy array
                        img_array = cv2.imread(os.path.join(path, img))
                        # could also resize here if needed
                        # turn image into RGB as OpenCV loads images in BGR
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                    # normalize image array if wanted
                    if do_normalization:
                        img_array = img_array / 255

                    img_array = img_array.reshape(192, 168, 1)

                    # append image and label to the corresponding dataset
                    if mode == 'train':
                        training_data.append([img_array, label])
                    elif mode == 'test':
                        testing_data.append([img_array, label])
                    else:
                        validation_data.append([img_array, label])

                except Exception as e:
                    print('An unexpected exception occured during fill_dataset()')
                    pass

        # constructing the path for training, testing and validation
        # dataset from the database path
        train_dir = os.path.join(src_dir, 'train')
        test_dir = os.path.join(src_dir, 'test')
        val_dir = os.path.join(src_dir, 'val')

        # get number of subdirectories = num_classes
        subdirs = os.listdir(train_dir)
        num_classes = len(subdirs)

        # auxiliary variable used for one-hot encoding
        class_labels = np.eye(num_classes, dtype=int)

        # auxiliary variables to hold the datasets
        training_data = []
        testing_data = []
        validation_data = []

        if onehot:
            print('Encode Class Labels with One Hot Encoding')
        else:
            print('Turn ID into class label - no One Hot Encoding')

        # loop through all subdirectories / identities
        for identity in subdirs:

            id_value = subdirs.index(identity)

            # construct the ID specific train-, test- and
            # validation-directory paths
            path_train = os.path.join(train_dir, identity)
            path_test = os.path.join(test_dir, identity)
            path_val = os.path.join(val_dir, identity)

            # pick class label
            if onehot:
                label = class_labels[int(id_value)]
            else:
                label = int(id_value)

            # fill datasets with auxiliary function
            fill_dataset('train', path_train, label, do_normalization=norm)
            fill_dataset('test', path_test, label, do_normalization=norm)
            fill_dataset('val', path_val, label, do_normalization=norm)

        return training_data, testing_data, validation_data

    # paths to datasets
    if normalized:  # normalized-version
        filename_train_ds = os.path.join(ds_name, 'train_dataset_normalized_%s.npy' % str(batch_size))
        filename_test_ds = os.path.join(ds_name, 'test_dataset_normalized_%s.npy' % str(batch_size))
        filename_val_ds = os.path.join(ds_name, 'validation_dataset_normalized_%s.npy' % str(batch_size))
    else:  # original- / scaled-version
        filename_train_ds = os.path.join(ds_name, 'train_dataset_%s.npy' % str(batch_size))
        filename_test_ds = os.path.join(ds_name, 'test_dataset_%s.npy' % str(batch_size))
        filename_val_ds = os.path.join(ds_name, 'validation_dataset_%s.npy' % str(batch_size))

    # if path doe not exist, load data from database folder instead
    # of prepared numpy dataset
    if not os.path.exists(filename_train_ds):
        if not os.path.exists(ds_name):
            os.mkdir(ds_name)

        data_dir = db_src_dir

        # load images from database into numpy arrays
        train_ds, test_ds, val_ds = load_data(src_dir=data_dir, onehot=True, norm=normalized)

        # save loaded data as prepared numpy arrays for faster usage later on
        np.save(filename_train_ds, train_ds)
        np.save(filename_test_ds, test_ds)
        np.save(filename_val_ds, val_ds)

    else:  # load datasets from prepared numpy arrays
        train_ds = np.load(filename_train_ds, allow_pickle=True).tolist()
        test_ds = np.load(filename_test_ds, allow_pickle=True).tolist()
        val_ds = np.load(filename_val_ds, allow_pickle=True).tolist()

    # scramble train and validation dataset
    random.shuffle(train_ds)
    random.shuffle(val_ds)

    # create dataflow for tensorpack usage
    train_df = DataFromList(train_ds)
    test_df = DataFromList(test_ds)
    val_df = DataFromList(val_ds)

    # batch dataflow
    bd_train_df = BatchData(train_df, batch_size=batchsize)
    bd_test_df = BatchData(test_df, batch_size=batchsize)
    bd_val_df = BatchData(val_df, batch_size=batchsize, remainder=True)

    return bd_train_df, bd_test_df, bd_val_df


# script for training a baseline model with Tensorpack
if __name__ == '__main__':
    # disable tensorflow v2 compatibility with tensorpack functions
    tf.compat.v1.disable_v2_behavior()

    # set-up logger
    # parent-directory, to hold multiple runs
    root_logdir = os.path.join(os.curdir, 'yale_quant_model_workLap')
    # sub-directory, to hold a single run, here naming is based on
    # the time and start date of the run
    run_id = time.strftime('run_%Y_%m_%d-%H_%M_%S')
    run_logdir = os.path.join(root_logdir, run_id)
    logger.set_logger_dir(run_logdir)

    # load data with desired batch size
    batch_size = 2
    train_data, test_data, val_data = get_data(ds_name="yaleB_cropped",
                                               db_src_dir="CroppedYale_Split",
                                               batchsize=batch_size,
                                               normalized=True,
                                               grayscale=True)

    # prepare training
    steps_per_epoch = len(train_data)

    myModel = BaselineModel()

    # prepare training-config required by tensorpack to launch a trainer
    train_config = TrainConfig(
        # model to perform the training on
        model=myModel,
        # input source for training, needs to be a type of dataflow instance
        data=QueueInput(train_data),
        # define callbacks for things that should happen after every epoch
        callbacks=[
            ModelSaver(),  # save the model
            InferenceRunner(  # run inference for validation
                val_data,  # dataflow instance used for validation
                ScalarStats(['categorical_cross_entropy_loss', 'accuracy'])),
            MaxSaver('validation_accuracy')  # save the model with highest accuracy (prefix 'validation_')
        ],
        steps_per_epoch=steps_per_epoch,
        max_epoch=50,
    )
    # use SimleTrainer to train the network
    # -> builds the model provided and minimizes the cost
    launch_train_with_config(train_config, SimpleTrainer())
