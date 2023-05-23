#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Introduction:   
    
    This ecode is cut-out from our workflow to show the neural networks generation and the usage of the relevance algorithm.
    It is (sadly) not intended to work out of the box, since we can't provide the dataset.
    You will have to bring your own data and then fit the network to your dataset. 
    You also need the toolbox 'keras' and 'tensorflow'. A GPU is not neccessary, the calculations run there only 20% faster
    
    If you have any questions about this ecode, I am willing to help. Please email me at
    touse_bias.de.taillez@uni-oldenburg.de
    or
    irazari@zae-ne.de
    
    Touse_bias de Taillez, Sept.2017
    Universität Oldenburg, Germany
'''
import os
import gc
import keras
from keras.models import Sequential, Model
# from keras.optimizers import SGD
from keras.regularizers import l2, l1  # , activity_l2,activity_l1
from keras.layers import Dense, Input, Dropout
from keras.layers import concatenate, Concatenate
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from ecode.fcn.Generator import DataGenerator, TestGenerator

import scipy.io as io
import numpy as np
import pandas as pd
import sys
import time
import random
import math
import eutils.util as util
from dotmap import DotMap
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
numHiddenLayer = 1
n_hidden = 2  # Num of hidden neurons
numSampleContext = 1  # training prediction window
loss = 2  # correlation based loss function
workingDir = util.makePath("../../result/fcn_test")
neuronWeightingFlag = True

data_document_path = "D:/eegdata/dataset_csv_Jon_202102B"
# data_document_path = "/document/data/eeg/dataset_csv_Jon_202102B"
data_meta = util.read_json(data_document_path + "/metadata.json")
args = DotMap()
args.numSampleContext = numSampleContext
args.people_number = data_meta.people_number
args.eeg_band = data_meta.eeg_band
args.eeg_channel_per_band = data_meta.eeg_channel_per_band
args.eeg_channel = args.eeg_band * args.eeg_channel_per_band
args.audio_band = data_meta.audio_band
args.audio_channel_per_band = data_meta.audio_channel_per_band
args.audio_channel = args.audio_band * args.audio_channel_per_band
args.channel_number = args.eeg_channel + args.audio_channel * 2
args.trail_number = data_meta.trail_number
args.cell_number = data_meta.cell_number
args.bands_number = data_meta.bands_number
args.fs = data_meta.fs
args.batch_size = 16
args.test_percent = 0.1
args.vali_percent = 0.1
args.delay = 11

args.netSize = 27  # 420 ms
dropout = 0.25
use_bias = True
l1Reg = 0.
l2Reg = 0.
earlyPatience = 10
GPU = "cpu"


###Functions 

def corr_loss(act, pred):  # Custom tailored loss function that values a high correlation. See our Paper for details
    cov = (K.mean((act - K.mean(act)) * (pred - K.mean(pred))))
    return 1 - (cov / (K.std(act) * K.std(pred) + K.epsilon()))


def create_base_network(net_input_dim, dropout, use_bias, kernel_init, bias_init, l1Reg, l2Reg):
    '''Base network to be shared (eq. to feature extraction).
    net_input_dim: size of subnet in TimeSamples
    '''
    seq = Sequential()  # Feed Forward Model
    if numHiddenLayer == 0:
        seq.add(Dense(1, activation='linear', input_dim=net_input_dim, kernel_regularizer=l1(l1Reg),
                      bias_regularizer=l1(l1Reg)
                      , use_bias=False, kernel_initializer=kernel_init, bias_initializer=bias_init))
    else:
        seq.add(Dense(n_hidden, activation='linear', input_dim=net_input_dim, bias_regularizer=l1(l1Reg),
                      kernel_regularizer=l1(l1Reg), use_bias=use_bias, kernel_initializer=kernel_init,
                      bias_initializer=bias_init))  # Input Layer
        seq.add(keras.layers.BatchNormalization())
        seq.add(keras.layers.Activation('tanh'))
        seq.add(Dropout(dropout))
        for layer in range(1, numHiddenLayer):
            seq.add(
                Dense(np.max([np.round(n_hidden / (layer + 1)), 1]), activation='linear', bias_regularizer=l2(l2Reg),
                      kernel_regularizer=l2(l2Reg), use_bias=use_bias, kernel_initializer=kernel_init,
                      bias_initializer=bias_init))
            seq.add(keras.layers.BatchNormalization())
            seq.add(keras.layers.Activation('tanh'))
            seq.add(Dropout(dropout))
        seq.add(Dense(1, activation='linear', bias_regularizer=l2(l2Reg), kernel_regularizer=l2(l2Reg),
                      use_bias=use_bias, kernel_initializer=kernel_init,
                      bias_initializer=bias_init))  # Output Layer. 1 Neuron
    return seq


def get_activations1(model, layer, X_batch):  # See get_interpret_new()
    if layer == -1:  # InputLayer returns data itself
        return [X_batch]
    else:  # Output of Layer X
        get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output])
        activations = get_activations([X_batch, 0])
        return activations


def get_interpret_new(model, X_batch, targetData, correctness, num_time_samples=0, use_bias=False, dropout=0.0):
    '''This is a self-tailored Version of Sturm et al 2016 Relevance algorithm
    Sadly there is NO easy-to-use Version for all possible network layouts. This is mainly due to the weights that play a role in 
    relevance calculation and the weights that DONT. These have to be deleted from the weight matrix. 
    Example:
        For a given Layer WITH Bias values for each neuron, these use_bias weight-layer have to be deleted since 
        in the mathematical equation only the activation of a neuron plays a role. We get this activation from get_activations1(model,layer,Data)
        and it already includes the effect of the use_bias layer.
        Sme for Dropout-layers. These appear in the activations-List but have to be deleted. In general you dont want any dropout occuring while testing/evaluation! 
        So generate your evaluation-network without dropout and then copy the weights from your trained network to this.
    '''
    weights = model.get_weights()

    # total number of layers (input + output + hidden layers)
    numLayers = len(weights)

    if dropout > 0.0:
        activations = list()
        if use_bias:
            for layer in range(-1, len(weights) - numHiddenLayer + numHiddenLayer - 1):
                activations.append(get_activations1(model, layer, X_batch)[0])
        else:
            for layer in range(-1, len(weights) + numHiddenLayer):
                activations.append(get_activations1(model, layer, X_batch)[0])

        activationLength = len(activations)
        for layer in reversed(range(2, activationLength, 2)):
            del activations[layer]

    #        If Bias, delete Bias-Layers from Weights
    if use_bias == True:
        for layer in reversed(range(1, numLayers, 2)):
            del weights[layer]

    numLayers = len(weights)

    if (num_time_samples == 0) or (num_time_samples == -1):
        num_time_samples = activations[0].shape[0]
    # calculated relevance, list of arrays:
    # one 2d-array for each layer, indexed as (time-sample, node) (same as activations)
    relevance = list()

    # relevance of the output nodes
    #        outputRelevance=np.ones((num_time_samples, num_output_nodes)) / num_output_nodes
    outputRelevance = correctness
    #        outputRelevance=1/(1+np.abs(activations[-1]-targetData))
    relevance.append(outputRelevance)

    # determine relevance per layer
    for layer in reversed(range(numLayers)):
        num_lower_nodes = activations[layer].shape[1]
        num_upper_nodes = activations[layer + 1].shape[1]
        # initialize relevance of current layer to zeros
        layer_relevance = np.zeros((num_time_samples, num_lower_nodes))

        # determine relevance of nodes in lower layer
        for upper_node in range(num_upper_nodes):
            upper_activation = activations[layer + 1][:num_time_samples, upper_node]
            upper_relevance = relevance[-1][:, upper_node]
            # calculate contribution of the current upper node for all time samples
            upper_contribution = upper_relevance / upper_activation
            # sum up for all the nodes in the lower layer for all time samples:
            # contribution of the upper node * weights from the current upper node to all lower nodes
            # this uses matrix multiplication:
            # upper_contribution as a column-vector (:,1) * weights as a row-vector (1,:)
            # see https://de.wikipedia.org/wiki/Matrizenmultiplikation#Spaltenvektor_mal_Zeilenvektor
            upper_contribution = upper_contribution.reshape(-1, 1)
            weight = weights[layer][:, upper_node].reshape(1, -1)
            layer_relevance += np.matmul(upper_contribution, weight)
        # lower activation can be factored out of all the terms within the sum
        layer_relevance *= activations[layer][:num_time_samples]
        # layer relevance complete
        relevance.append(layer_relevance)
    # reverse relevance to match the layer-order of activations, i.e. 0 is the input layer
    relevance.reverse()
    return relevance


def evaluation(modelA, test_data, baseNet, kernel_init, bias_init):
    gc.collect()  # collect garbage

    baseNetA = create_base_network(args.netSize * args.eeg_channel, 0, use_bias, kernel_init, bias_init, 0,
                                   0)  # Create BaseNet and load the weights from Training

    baseNetA.set_weights(modelA.get_weights())

    # calculate input neuron relevance for each of the input neurons
    envA, data, envU = util.get_split_data(test_data, args)
    envA = envA[:-args.netSize + 1]
    data = data[:-args.netSize + 1]
    envU = envU[:-args.netSize + 1]
    envA += np.random.randn(envA.shape[0], envA.shape[1]) * 0.000001
    test_generator = TestGenerator(test_data, args)

    # predA = baseNetA.predict_on_batch(data)  # Use the net to predict the test set
    predA = baseNetA.predict_generator(test_generator, verbose=1, max_queue_size=1, workers=3,
                                       use_multiprocessing=False)
    predA = np.squeeze(predA)
    # plt.plot(range(len(predA)), predA)
    # plt.plot(range(len(envA.flatten())), envA.flatten())
    # pd.DataFrame(predA).to_csv("try.csv")
    # plt.show()
    numSampleContext = 16
    if neuronWeightingFlag:  # calculate the input neuron relevance per sample for the test net
        corrMatrixCorrect = np.full((predA.shape[0], numSampleContext), np.nan)
        for samp in range(predA.shape[0] - numSampleContext):
            corrMatrixCorrect[samp:samp + numSampleContext, np.mod(samp, numSampleContext)] = \
                np.corrcoef(predA[samp:samp + numSampleContext].T, envA[samp:samp + numSampleContext, 0, None].T)[0][1]
        correctness = np.nanmean(corrMatrixCorrect[:predA.shape[0] - numSampleContext], axis=1).reshape(
            (predA.shape[0] - numSampleContext, 1))
        relevanceNeurons = get_interpret_new(baseNet, data[:-numSampleContext, :], envA[:-numSampleContext, 0, None],
                                             correctness, -1, use_bias, dropout)
        neuronWeighting = np.median(np.squeeze(relevanceNeurons[0][np.where(correctness[:, 0] > 0), :]), axis=0)
        neuronWeightingStd = np.std(np.squeeze(relevanceNeurons[0][np.where(correctness[:, 0] > 0), :]), axis=0)

    # Evaluate performance for different analysis window lengths
    blockRange = np.asarray([60, 30, 10, 5, 2, 1, 0.5])  # analysis window lengths

    corrResults = np.zeros((9, len(blockRange)))

    for blockLengthIterator in range(len(blockRange)):
        t = time.time()
        blockLength = int(args.fs * blockRange[blockLengthIterator])
        corrA = np.asarray(range(0, envA.shape[0] - blockLength)) * np.nan
        corrU = np.asarray(range(0, envU.shape[0] - blockLength)) * np.nan

        for block in range(corrA.shape[
                               0]):  # for a specific analysis window, run trough the test set prediction and correlate with attended and unattended envelope
            # print(envA[block:block + blockLength].T.shape)
            # print(predA[block:block + blockLength].T.shape)
            corrA[block] = np.corrcoef(envA[block:block + blockLength].T, predA[block:block + blockLength].T)[0][1]
            corrU[block] = np.corrcoef(envU[block:block + blockLength].T, predA[block:block + blockLength].T)[0][1]

        corrResults[0, blockLengthIterator] = np.nanmean(corrA)
        # corrResults[1, blockLengthIterator] = np.nanstd(corrA)
        corrResults[2, blockLengthIterator] = np.nanmean(corrU)
        # corrResults[3, blockLengthIterator] = np.nanstd(corrU)
        results = np.clip(corrA, -1, 1) > np.clip(corrU, -1, 1)
        # output = np.concatenate([corrA[None, :], corrU[None, :]], axis=0)
        # io.savemat('predict' + str(blockLength) + '.mat', {'results_' + str(blockLength): output})
        corrResults[4, blockLengthIterator] = np.nanmean(
            results)  # Values the networks decision. 1 denotes "correct" zero "wrong". Averages also over the complete test set. This result the networks accuracy!
        accuracy = corrResults[4, blockLengthIterator]
        if accuracy < 0.45:
            break
        corrResults[5, blockLengthIterator] = (np.log2(2) + accuracy * np.log2(accuracy) + (1 - accuracy) * np.log2(
            (1 - accuracy + 0.00000001) / 1)) * args.fs / blockLength * 60
        corrResults[6, blockLengthIterator] = blockRange[blockLengthIterator]
        # corrResults[7] = trainPat[0]  # Which participant is evaluated
        # corrResults[8] = startPointTest  # At which time point did the evaluation/test set started

    return corrResults, neuronWeighting, neuronWeightingStd


def train(data, kernel_init, bias_init):
    ##########Build Net########
    baseNet = create_base_network(args.netSize * args.eeg_channel, dropout, use_bias, kernel_init, bias_init, l1Reg,
                                  l2Reg)  # Create the base network that is to be used multiple times (numSampleContext denotes the number of usages)

    if args.numSampleContext == 1:
        inputTensor = Input(shape=(args.netSize * args.eeg_channel,))
        processed = baseNet(inputTensor)
        out = processed
        modelA = Model(inputs=inputTensor, outputs=out)
    else:
        inputTensor = [Input(shape=(args.netSize * args.eeg_channel,)) for i in
                       range(args.numSampleContext)]  # Generate list of input tensors
        processedTensor = [baseNet(inputTensor[i]) for i in range(
            args.numSampleContext)]  # Generate list of baseNet applications to their respective input tensor
        lay = concatenate(processedTensor)
        modelA = Model(inputs=inputTensor, outputs=lay)  # create Model

    # opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    opt = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    if loss == 1:  # Compilation
        modelA.compile(optimizer=opt, loss='mean_squared_error')  # mse Model
    elif loss == 2:
        modelA.compile(optimizer=opt, loss=corr_loss)  # CorrLoss Model

    checkLow = ModelCheckpoint(filepath=workingDir + '/weights_lowAcc' + str(GPU) + '.hdf5', verbose=0,
                               save_best_only=True,
                               mode='min', monitor='loss')  # Checkpoints for the lowest achieved evaluation loss
    early = EarlyStopping(monitor='val_loss', patience=earlyPatience,
                          mode='min')  # Daemon to stop before the maximum number of epochs is reached. It checks if the validation loss did not decrese for the last 'earlyPatience' trials

    modelA.fit_generator(data.train_generator, epochs=500, verbose=0, max_queue_size=1, workers=3,
                         use_multiprocessing=False,
                         validation_data=data.validation_generator, callbacks=[checkLow, early], shuffle=False)

    return modelA, baseNet


def prepare_data(name):
    # Generator
    train_data = []
    vali_data = []
    test_data = []
    for k in range(args.trail_number):
        # 读取数据
        filename = data_document_path + "/No/" + name + "Tra" + str(k + 1) + ".csv"
        data = pd.read_csv(filename, header=None)
        data = util.add_delay(data, args)
        envA, eeg, envU = util.get_split_data(data, args)

        # envA = util.moving_average(np.array(envA).flatten(), window_size=10)
        # envU = util.moving_average(np.array(envU).flatten(), window_size=10)

        envA = np.array(envA).flatten()
        envU = np.array(envU).flatten()

        envA = util.normalization(envA)
        envU = util.normalization(envU)

        envA = pd.DataFrame(envA[:, None])
        envU = pd.DataFrame(envU[:, None])

        eeg = util.normalization(eeg)

        data = pd.concat([envA, eeg, envU], axis=1, ignore_index=True)
        train_temp, test_temp = util.split_data(data, args.test_percent)
        train_temp, vali_temp = util.split_data(train_temp, args.vali_percent)
        train_data.append(train_temp)
        vali_data.append(vali_temp)
        test_data.append(test_temp)

    train_data = pd.concat(train_data, axis=0, ignore_index=True)
    vali_data = pd.concat(vali_data, axis=0, ignore_index=True)
    test_data = pd.concat(test_data, axis=0, ignore_index=True)

    data = DotMap()
    data.test_data = np.array(test_data)
    data.train_data = np.array(train_data)
    data.vali_data = np.array(vali_data)
    data.train_generator = DataGenerator(data.train_data, args)
    data.validation_generator = DataGenerator(data.vali_data, args)
    return data


def main(name, delay, log_path="./result/fcn"):
    print(name + " start")
    args.delay = delay
    print("delay:" + str(args.delay))
    corr_train = None
    corr_vali = None
    is_continue = True
    kernel_init = None
    bias_init = None
    Threshold = [0.7, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45]

    while is_continue:
        data = prepare_data(name)
        seed = random.randint(0, 2615731125)
        print(seed)
        kernel_init = keras.initializers.glorot_uniform(seed=seed)
        bias_init = keras.initializers.Zeros()
        modelA, baseNet = train(data, kernel_init, bias_init)
        modelA.load_weights(filepath=workingDir + '/weights_lowAcc' + str(
            GPU) + '.hdf5')  # Load the weights that worked best on the evaluation set
        corr_train, neuronWeighting, neuronWeightingStd = evaluation(modelA, data.train_data, baseNet, kernel_init, bias_init)
        corr_vali, neuronWeighting, neuronWeightingStd = evaluation(modelA, data.vali_data, baseNet, kernel_init, bias_init)
        is_continue = False
        for i in range(corr_train.shape[1]):
            ct = corr_train[4, i]
            cv = corr_vali[4, i]
            print(str(ct) + " " + str(cv))
            if ct < Threshold[i] or cv < Threshold[i] or (
                    i > 0 and (ct > corr_train[4, i - 1] or cv > corr_vali[4, i - 1])):
                is_continue = True
                break

    corr_test, neuronWeighting, neuronWeightingStd = evaluation(modelA, data.test_data, baseNet, kernel_init, bias_init)

    if neuronWeightingFlag:
        io.savemat(util.makePath(log_path) + '/CorrTrain' + name + '.mat', {'CorrTrain' + name: corr_train, 'neuronWeighting': neuronWeighting,
                                                    'neuronWeightingStd': neuronWeightingStd})
        io.savemat(util.makePath(log_path) + '/CorrVali' + name + '.mat', {'CorrVali' + name: corr_vali, 'neuronWeighting': neuronWeighting,
                                                    'neuronWeightingStd': neuronWeightingStd})
        io.savemat(util.makePath(log_path) + '/CorrTest' + name + '.mat', {'CorrTest' + name: corr_test, 'neuronWeighting': neuronWeighting,
                                                    'neuronWeightingStd': neuronWeightingStd})
    else:
        io.savemat(util.makePath(log_path) + '/CorrTrain' + name + '.mat', {'CorrTrain' + name: corr_train})
        io.savemat(util.makePath(log_path) + '/CorrVali' + name + '.mat', {'CorrVali' + name: corr_vali})
        io.savemat(util.makePath(log_path) + '/CorrTest' + name + '.mat', {'CorrTest' + name: corr_test})

    # # Save Results
    # if neuronWeightingFlag:
    #     io.savemat('backTell' + str(GPU) + name + '.mat', {'corrResults': corrResults, 'neuronWeighting': neuronWeighting,
    #                                                 'neuronWeightingStd': neuronWeightingStd})
    # else:
    #     io.savemat('backTell' + str(GPU) + name + '.mat', {'corrResults' + name: corrResults})


if __name__ == "__main__":
    if (len(sys.argv) > 1 and sys.argv[1].startswith("S")):
        main(sys.argv[1], sys.argv[2])
    else:
        main("S8", 11)
