import datetime
import multiprocessing
import time
from math import ceil
from math import floor
import os
from pathlib import Path
from zipfile import ZipFile
import shutil
from os.path import basename

import numpy as np
import seal


# compute scaled average pooling (encrypted)
# return: encrypted scaled average pooling output based on input parameters
def avgPool_enc(x, pool_size, strides, padding, activation=None, evaluator=None, relin_keys=None):
    # compute output size
    inp_shape = x.shape

    if padding == 'VALID':
        out_height = int(floor(((inp_shape[0] - pool_size[0]) / strides[0]) + 1))
        out_width = int(floor(((inp_shape[1] - pool_size[1]) / strides[1]) + 1))
    else:  # SAME padding
        out_height = int(ceil((inp_shape[0]) / strides[1]))
        out_width = int(ceil((inp_shape[1]) / strides[2]))
        print('Error: computation of SAME padding not yet supported')
        exit(-1)

    out_shape = (out_height, out_width, inp_shape[2])

    # auxiliary variable for results
    output = np.zeros(out_shape, dtype=object)

    # perform layer computation if a seal evaluator is given
    if evaluator is not None:

        # move filter/ window over image
        for h in range(out_height):
            for w in range(out_width):
                for k in range(out_shape[2]):
                    # get current window
                    window = x[
                             (h * strides[0]):(h * strides[0] + pool_size[0]),
                             (w * strides[1]):(w * strides[1] + pool_size[1]),
                             k
                             ]

                    # compute scaled average, i.e. sum of all elements in window
                    window = window.flatten()
                    res = seal.Ciphertext()
                    evaluator.add_many(window, res)

                    # apply activation function if one is provided
                    if activation == 'squared':
                        evaluator.square_inplace(res)
                        evaluator.relinearize_inplace(res, relin_keys)

                    # add result to output variable
                    output[h, w, k] = res

    return output


# compute convolution layer (encrypted)
# return: encrypted convolution layer output based on input parameters
def conv2d_enc(x, num_filters, kernel_size, strides, padding, str_pid, input_shape=None, activation=None, weights=None, bias=None,
               evaluator=None, relin_keys=None, context=None, path=None):

    # load input if no input is provided:
    if x is None:
        start_load_input = time.time()
        print(str_pid, ': Loading/encrypting input')
        x = load_inputs(context=context, path=path, np_shape=input_shape)
        print(str_pid, ': Done.')
        end_load_input = time.time()
        print(str_pid, ': Time between Start and End of loading/encrypting input: ',
              str(datetime.timedelta(seconds=(end_load_input - start_load_input))))

    inp_shape = x.shape
    # print('input_shape = ', inp_shape)

    # compute output size
    if padding == 'VALID':
        out_height = int(floor(((inp_shape[0] - kernel_size[0]) / strides[0]) + 1))
        out_width = int(floor(((inp_shape[1] - kernel_size[1]) / strides[1]) + 1))
    else:  # SAME padding
        out_height = int(ceil((inp_shape[0]) / strides[1]))
        out_width = int(ceil((inp_shape[1]) / strides[2]))
        print('Error: computation of SAME padding not yet supported')
        exit(-1)

    out_shape = (out_height, out_width, num_filters)
    # print('output shape = ', out_shape)

    # auxiliary variable for outputs / feature maps
    feature_maps = np.zeros(out_shape, dtype=object)

    # perform layer computation if a seal evaluator is given
    if evaluator is not None:

        # loop over each filter
        for l_filter in range(num_filters):

            # get weights for current filter
            current_weights = weights[:, :, :, l_filter]

            # auxiliary variable for intermediate results
            result = np.zeros((out_shape[0], out_shape[1]), dtype=object)

            # move filter/ window over image
            for h in range(out_shape[0]):  # out_height
                for w in range(out_shape[1]):  # out_width

                    # get the current region of input
                    current_region = x[
                                     (h * strides[0]):(h * strides[0] + kernel_size[0]),
                                     (w * strides[1]):(w * strides[1] + kernel_size[1])
                                     ]

                    # auxiliary variable for intermediate results
                    tmp_res = np.zeros(current_region.shape, dtype=object)

                    # perform element-wise multiplication between the current region
                    # and the filter
                    for a in range(current_region.shape[0]):  # region height
                        for b in range(current_region.shape[1]):  # region width
                            for c in range(current_region.shape[2]):  # channel / filter
                                tmp_res_single = seal.Ciphertext()
                                evaluator.multiply_plain(current_region[a, b, c], current_weights[a, b, c],
                                                         tmp_res_single)
                                evaluator.relinearize_inplace(tmp_res_single, relin_keys)

                                # add single intermediate result to array of intermediate results
                                tmp_res[a, b, c] = tmp_res_single

                    tmp_res = tmp_res.flatten()
                    conv_sum = seal.Ciphertext()
                    evaluator.add_many(tmp_res, conv_sum)

                    # add bias if one is provided
                    if bias is not None:
                        evaluator.add_plain_inplace(conv_sum, bias[l_filter])

                    # apply activation function if one is provided
                    if activation == 'squared':
                        evaluator.square_inplace(conv_sum)
                        evaluator.relinearize_inplace(conv_sum, relin_keys)

                    # save the intermediate result
                    result[h, w] = conv_sum

            # add result to output / feature maps
            feature_maps[:, :, l_filter] = result

    return feature_maps


# compute dense layer (encrypted)
# return: encrypted dense layer output based on input parameters
def dense_enc(x, num_nodes, activation=None, weights=None, bias=None, evaluator=None, relin_keys=None):
    # compute output size
    out_shape = num_nodes

    # auxiliary variable for outputs
    output = np.zeros(out_shape, dtype=object)

    # perform layer computation if a seal evaluator is given
    if evaluator is not None:

        # loop through nodes
        for node in range(num_nodes):

            # auxiliary variable for intermediate results
            tmp = np.zeros(shape=x.shape, dtype=object)

            # multiply input and weights
            for j in range(len(tmp)):
                tmp_res_single = seal.Ciphertext()
                evaluator.multiply_plain(x[j], weights[j, node], tmp_res_single)
                evaluator.relinearize_inplace(tmp_res_single, relin_keys)
                tmp[j] = tmp_res_single

            # auxiliary value to save summation result
            res = seal.Ciphertext()
            # sum up all elements of the vector
            evaluator.add_many(tmp, res)

            # add bias if one is provided
            if bias is not None:
                evaluator.add_plain_inplace(res, bias[node])

            # apply activation function if one is provided
            if activation == 'squared':
                evaluator.square_inplace(res)
                evaluator.relinearize_inplace(res, relin_keys)

            # add result to output variable
            output[node] = res

    return output


# batch weights, preparing them for encoding
# return: batched weight dictionary
def batch_weights(path, n):
    # load weight dictionary from file
    tmp = np.load(path, allow_pickle=True)[()]

    new_weights = {}

    for key in tmp:

        old_w = tmp[key]

        # batch param by turning a single value into a vector of length n
        if old_w is not None:
            # loop through given parameter shape to turn every
            # single entry into a batched vector
            if len(old_w.shape) == 1:  # bias
                new_w = np.zeros(shape=(old_w.shape[0], N))
                for a in range(old_w.shape[0]):
                    new_w[a, :] = [old_w[a]] * n
            elif len(old_w.shape) == 2:  # weights fc
                new_w = np.zeros(shape=(old_w.shape[0], old_w.shape[1], N))
                for a in range(old_w.shape[0]):
                    for b in range(old_w.shape[1]):
                        new_w[a, b, :] = [old_w[a, b]] * n
            elif len(old_w.shape) == 4:  # weights conv
                new_w = np.zeros(shape=(old_w.shape[0], old_w.shape[1], old_w.shape[2], old_w.shape[3], N))
                for a in range(old_w.shape[0]):
                    for b in range(old_w.shape[1]):
                        for c in range(old_w.shape[2]):
                            for d in range(old_w.shape[3]):
                                new_w[a, b, c, d, :] = [old_w[a, b, c, d]] * n
            else:
                print('Unexpected error while batching the weights.')
                print('Key describes neither a bias nor a fc- or conv-weight.')
                exit(-1)
        else:
            new_w = None

        # add loaded value to the new weight dictionary under the same key
        new_weights[key] = new_w

    return new_weights


# encode weights according to the encoder of the chosen parameter set
# return: encoded weight dictionary
def enco_weights(path, enco, batched_weights=None, name=''):
    # perform batch encoding
    print(name, 'LOAD INT-WEIGHTS AND ENCODE THEM')

    # auxiliary value for loaded weights
    weights_dic = {}

    if batched_weights is not None:
        tmp = batched_weights
    else:
        # load weight dictionary from file
        tmp = np.load(path, allow_pickle=True)[()]  # already batched

    # loop through dictionary entries
    for key in tmp:

        int_weights = tmp[key]

        if int_weights is not None:

            w_shape = int_weights.shape

            # loop through given parameter shape to encode every
            # single entry into a seal plaintext
            if len(w_shape) == 2:  # bias

                encoded_weights = np.zeros(shape=(w_shape[0]), dtype=object)

                for a in range(w_shape[0]):
                    tmp_plain = seal.Plaintext()
                    # load array entry (which is another array) as integer and turn it
                    # into a list to make a IntVector instance out of the list before
                    # handing it to the batch encoder
                    enco.encode(seal.IntVector(int_weights[a].astype(int).tolist()), tmp_plain)
                    encoded_weights[a] = tmp_plain

            elif len(w_shape) == 3:  # fully connected

                encoded_weights = np.zeros(shape=(w_shape[0], w_shape[1]), dtype=object)

                for a in range(w_shape[0]):
                    for b in range(w_shape[1]):
                        tmp_plain = seal.Plaintext()
                        enco.encode(seal.IntVector(int_weights[a, b].astype(int).tolist()), tmp_plain)
                        encoded_weights[a, b] = tmp_plain

            else:  # conv2d

                encoded_weights = np.zeros(shape=(w_shape[0], w_shape[1], w_shape[2], w_shape[3]), dtype=object)

                for a in range(w_shape[0]):
                    for b in range(w_shape[1]):
                        for c in range(w_shape[2]):
                            for d in range(w_shape[3]):
                                tmp_plain = seal.Plaintext()
                                enco.encode(seal.IntVector(int_weights[a, b, c, d].astype(int).tolist()), tmp_plain)
                                encoded_weights[a, b, c, d] = tmp_plain

        else:
            encoded_weights = None

        # add encoded value to the new weight dictionary under the same key
        weights_dic[key] = encoded_weights

    return weights_dic


# load encrypted inputs from directory
# return: loaded encrypted inputs
def load_inputs(context, path, np_shape):

    # auxiliary value for loaded inputs
    loaded_inputs = np.zeros(shape=np_shape, dtype=object)

    # load ciphertexts from files
    for file in os.listdir(path):
        filename = os.fsdecode(file)
        indices = [int(s) for s in filename.split('.')[0].split('_') if s.isdigit()]
        tmp_load_cp = seal.Ciphertext()
        tmp_load_cp.load(context, '%s/%s' % (path, filename))
        loaded_inputs[indices[0], indices[1], indices[2]] = tmp_load_cp

    return loaded_inputs


# save encrypted outputs to zip file
def save_outputs(pid, outputs):
    # create tmp dir to store values before zip
    dirname = 'encr_outputs_%s' % pid
    os.mkdir(dirname)

    # save outputs to tmp dir
    for node_i in range(outputs.shape[0]):
        outputs[node_i].save('%s/outsave_%d.cp' % (dirname, node_i))

    # add files to zip
    with ZipFile('%s.zip' % dirname, 'w') as zipObj:
        for foldername, subfolders, filenames in os.walk(dirname):
            for filename in filenames:
                filePath = os.path.join(foldername, filename)
                zipObj.write(filePath, basename(filePath))

    # remove tmp dir
    shutil.rmtree(dirname)
    return


# run inference of the encrypted network
# (1) set up seal context for running process
# (2) encode weights with batch encoder
# (3) load inputs
# (4) run inference on encrypted network
# (5) save results to disk
def execute_network(pid, b_weights):
    # get process name (pid) and path to weights and inputs
    str_pid = 'p%s' % str(pid).zfill(2)
    process_name = 'process_%s: ' % str_pid
    weights_path = '_batched_weights_yale/w_%s.npy' % str_pid
    inp_path = 'yaleface_enc_%s.zip' % str_pid

    # SETUP SEAL CONTEXT
    params = seal.EncryptionParameters(seal.scheme_type.BFV)

    poly_mod_degree = 16384
    params.set_poly_modulus_degree(poly_mod_degree)
    params.set_coeff_modulus(seal.CoeffModulus.BFVDefault(poly_mod_degree))

    # plain moduli t_i
    t_i = [2424833, 2654209, 2752513, 3604481, 3735553, 4423681, 4620289, 4816897, 4882433, 5308417, 5767169]
    # choose t_i according to process number
    params.set_plain_modulus(t_i[pid])

    context = seal.SEALContext.Create(params)

    # check that parameters are suitable for batch encoding
    qualifiers = context.first_context_data().qualifiers()
    print(process_name, "Batching enabled: " + str(qualifiers.using_batching))

    # load existing keys
    keygen = seal.KeyGenerator(context)
    public_key = keygen.public_key()
    relin_keys = keygen.relin_keys()

    pk_file = '_keys_yale/' + str(pid) + '_pubkey'
    ev_file = '_keys_yale/' + str(pid) + '_evalkey'

    if Path(pk_file).exists() and Path(ev_file).exists():
        print(process_name, 'LOAD STORED KEYS')
        public_key.load(context, pk_file)
        relin_keys.load(context, ev_file)
    else:
        print(process_name, 'Error: required keys not available')
        exit(-1)

    # define batch encoder
    batch_encoder = seal.BatchEncoder(context)
    # slot_count = batch_encoder.slot_count()
    # row_size = int(slot_count / 2)
    # print('Slot-count: ' + str(slot_count) + ', Plaintext matrix row size: ' + str(row_size))

    # get instances of the encryptor and evaluator based on the
    # parameters defined above
    encryptor = seal.Encryptor(context, public_key)
    evaluator = seal.Evaluator(context)
    # END SETUP SEAL

    print(process_name, 'START TEST')

    # load and encode the batched weights
    start_weights = time.time()
    print(process_name, 'LOAD WEIGHTS')
    w_dic = enco_weights(weights_path, batched_weights=b_weights, enco=batch_encoder, name=process_name, pid=pid)
    print(process_name, 'Done loading weights.')
    end_weights = time.time()
    print(process_name, 'Time between Start and End of loading weights: ',
          str(datetime.timedelta(seconds=(end_weights - start_weights))))

    # load input here, if you do not want to load it inside the first convolution layer
    inp_shape = (192, 168, 1)
    # start_load_input = time.time()
    # print(process_name, 'LOAD INPUT')
    # encry_inp = load_inputs(context=context, path=inp_path, np_shape=inp_shape)
    # print(process_name, 'Done.')
    # end_load_input = time.time()
    # print(process_name, 'Time between Start and End of loading image: ',
    #       str(datetime.timedelta(seconds=(end_load_input - start_load_input))))
    encry_inp = None

    # conv0
    start_conv0 = time.time()
    print(process_name, 'Compute Layer: conv0')

    x = conv2d_enc(x=encry_inp,
                   num_filters=5, kernel_size=(9, 9), strides=(4, 4), padding='VALID',
                   input_shape=inp_shape,
                   weights=w_dic['conv0_weights'], bias=w_dic['conv0_bias'],
                   activation=None,
                   str_pid=str_pid,
                   path=inp_path,
                   evaluator=evaluator, relin_keys=relin_keys, context=context)
    end_conv0 = time.time()
    print(process_name, 'Time between Start and End of Conv0: ',
          str(datetime.timedelta(seconds=(end_conv0 - start_conv0))))

    # scaled avg-pool - pool0
    start_pool0 = time.time()
    print(process_name, 'Compute Layer: pool0')
    x = avgPool_enc(x, pool_size=(2, 2), strides=(2, 2), padding='VALID',
                    activation='squared',
                    evaluator=evaluator,
                    relin_keys=relin_keys)
    end_pool0 = time.time()
    print(process_name, 'Time between Start and End of Pool0: ',
          str(datetime.timedelta(seconds=(end_pool0 - start_pool0))))

    # conv1
    start_conv1 = time.time()
    print(process_name, 'Compute Layer: conv1')
    x = conv2d_enc(x, num_filters=50, kernel_size=(5, 5), strides=(2, 2), padding='VALID',
                   weights=w_dic['conv1_weights'], bias=w_dic['conv1_bias'],
                   activation=None, str_pid=str_pid,
                   evaluator=evaluator, relin_keys=relin_keys, context=context)
    end_conv1 = time.time()
    print(process_name, 'Time between Start and End of Conv1: ',
          str(datetime.timedelta(seconds=(end_conv1 - start_conv1))))

    # scaled avg-pool - pool1
    start_pool1 = time.time()
    print(process_name, 'Compute Layer: pool1')
    x = avgPool_enc(x, pool_size=(2, 2), strides=(2, 2), padding='VALID',
                    activation='squared',
                    evaluator=evaluator,
                    relin_keys=relin_keys)
    end_pool1 = time.time()
    print(process_name, 'Time between Start and End of Pool1: ',
          str(datetime.timedelta(seconds=(end_pool1 - start_pool1))))

    # flatten
    x = x.flatten()
    print(process_name, 'Flatten-Layer')
    print(process_name, 'output shape = ', x.shape)

    # fc0
    start_fc0 = time.time()
    print(process_name, 'Compute Layer: fc0')
    x = dense_enc(x, num_nodes=128, weights=w_dic['fc0_weights'],
                  bias=w_dic['fc0_bias'], evaluator=evaluator, relin_keys=relin_keys)
    end_fc0 = time.time()
    print(process_name, 'Time between Start and End of Fc0: ',
          str(datetime.timedelta(seconds=(end_fc0 - start_fc0))))

    # fc1
    start_fc1 = time.time()
    print(process_name, 'Compute Layer: fc1')
    out = dense_enc(x, num_nodes=38, weights=w_dic['fc1_weights'],
                    activation='squared',
                    bias=w_dic['fc1_bias'], evaluator=evaluator, relin_keys=relin_keys)
    end_fc1 = time.time()
    print(process_name, 'Time between Start and End of Fc1: ',
          str(datetime.timedelta(seconds=(end_fc1 - start_fc1))))

    print('OUTPUT-SHAPE: ', out.shape)

    # save outputs to disk
    start_save = time.time()
    print(process_name, "Saving results")
    save_outputs(pid=str_pid, outputs=out)
    end_save = time.time()
    print(process_name, 'Time between Start and End of Saving: ',
          str(datetime.timedelta(seconds=(end_save - start_save))))


# example script for inference of the encrypted network
if __name__ == '__main__':
    # start measuring computation time
    start_main = time.time()

    print('MAIN: START TEST ENCRYPTED NETWORK')

    # adapt paths according to the executing machine
    path_weights = '_weights/yale_scaled_Bit8.npy'
    N = 16384
    # maximum number of processes, here 11 as we have 11 t_i values
    num_processes = 1  # 2, 11

    try:
        # batch weights according to N
        start_batch = time.time()
        print('MAIN: Batch weights')
        batched_weights = batch_weights(path_weights, N)
        print('MAIN: Done')
        end_batch = time.time()
        print('MAIN: Time between Start and End of batching the weights: ',
              str(datetime.timedelta(seconds=(end_batch - start_batch))))

        for p_i in range(0, num_processes):

            # run processes
            # example for running one process at a time
            processes = []

            # initialize and start process(es)
            for i in range(0, 1):
                p = multiprocessing.Process(target=execute_network, args=(p_i, batched_weights))
                processes.append(p)
                p.start()

            # wait for all processes to finish
            for process in processes:
                process.join()

    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)

    end_main = time.time()
    print('MAIN: DONE TEST ENCRYPTED NETWORK')
    print('MAIN: Time between Start and End of main: ', str(datetime.timedelta(seconds=(end_main - start_main))))
