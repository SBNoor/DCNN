#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:31:34 2018

@author: au560049
"""

from keras.models import Sequential, Model
import keras
from keras import optimizers
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping,ModelCheckpoint
import numpy as np
from keras.layers import Dense, Dropout, Flatten, concatenate,Input
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from os import getcwd, chdir
import keras
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import json
from generator import DataGenerator


# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)
LST = ['pi','thetaW','tajD','distVar','distSkew','distKurt','nDiplos','diplo_H1_win','diplo_H12_win',
       'diplo_H2/H1_win','diplo_ZnS_win','diplo_Omega']

Y_label = {'soft':0,'hard':1,'neutral':2,'linkedhard':3,'linkedSoft':4}

def integer_encoding(Y):
    
    """
    
    This function transforms categorical variables into integers. This helps me with one hot encoding later on. 
    
    Input: Y: List of labels
    Output: Integer encoded list (of labels)
    
    """
    
    int_label = []
    for label in Y:
        int_label.append(Y_label[label])
    
    return int_label

def encode_Y(Y):
    
    """
    
    One hot encoding of output labels i.e. creating a 2D matrix of size num of samples * num of labels. 
    
    Input: Y: List of labels
    Output: returns one hot encoded matrix
    
    """
    
    return to_categorical(Y)

def compile_model(network):
    #Compile a sequential model.
    #Args:
        #network (dict): the parameters of the network
    #Returns:
        #a compiled network.
    
    # Get network parameters.
    
    
    """
    'nb_conv' : [1,2,3,4,5,6],
    'conv_filters': [50,100,150,200,250],
    'filter_size' : [2,3,4,5],
    'nb_neurons': [64, 128, 256, 512, 768, 1024],
    'nb_layers': [1, 2, 3, 4],
    'activation': ['relu', 'elu', 'tanh', 'sigmoid'],
    'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',
                      'adadelta', 'adamax', 'nadam'],
    'dropout' : [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    
    """
    
    nb_conv = network['nb_conv']
    conv_filters = network['conv_filters']
    activation = network['activation']
    optimizer = network['optimizer']
    dropout_conv = network['dropout_conv']
    filter_size = network['filter_size']

    model = Sequential()
    
    for i in range(nb_conv):
        if i == 0:
            	model.add(Conv2D(conv_filters,kernel_size = filter_size,activation = activation,kernel_regularizer=keras.regularizers.l2(0.0001),padding='same',input_shape=(144,5000,1)))
        else:
            model.add(Conv2D(conv_filters,kernel_size = filter_size,activation = activation,padding='same',kernel_regularizer=keras.regularizers.l2(0.0001)))
            model.add(MaxPooling2D(pool_size=2))
            model.add(Dropout(dropout_conv))
    
    model.add(Flatten())


    # Output layer.
    model.add(Dense(5,activation = 'softmax')) 
    

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

    return model



def train_and_score(network, dataset):
    """Train the model, return test loss.
    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating
    """
    """
    numSubWins = 11
    
    hard = np.loadtxt("/home/noor/popGen/africa_single/data/trainingSets/hard.fvec",skiprows=1)
    nDims = int(hard.shape[1] / numSubWins)
    h1 = np.reshape(hard,(hard.shape[0],nDims,numSubWins))
    neut = np.loadtxt("/home/noor/popGen/africa_single/data/trainingSets/neut.fvec",skiprows=1)
    n1 = np.reshape(neut,(neut.shape[0],nDims,numSubWins))
    soft = np.loadtxt("/home/noor/popGen/africa_single/data/trainingSets/soft.fvec",skiprows=1)
    s1 = np.reshape(soft,(soft.shape[0],nDims,numSubWins))
    lsoft = np.loadtxt("/home/noor/popGen/africa_single/data/trainingSets/linkedSoft.fvec",skiprows=1)
    ls1 = np.reshape(lsoft,(lsoft.shape[0],nDims,numSubWins))
    lhard = np.loadtxt("/home/noor/popGen/africa_single/data/trainingSets/linkedHard.fvec",skiprows=1)
    lh1 = np.reshape(lhard,(lhard.shape[0],nDims,numSubWins))
    
    both=np.concatenate((h1,n1,s1,ls1,lh1))
    y=np.concatenate((np.repeat(0,len(h1)),np.repeat(1,len(n1)), np.repeat(2,len(s1)), np.repeat(3,len(ls1)), np.repeat(4,len(lh1))))
    
    both = both.reshape(both.shape[0],nDims,numSubWins,1)
    X_train, X_test, y_train, y_test = train_test_split(both, y, test_size=0.2)
    
    Y_train = np_utils.to_categorical(y_train, 5)
    Y_test = np_utils.to_categorical(y_test, 5)
    X_valid, X_test, Y_valid, Y_test = train_test_split(X_test, Y_test, test_size=0.5)

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        horizontal_flip=True)

    validation_gen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        horizontal_flip=False)
    test_gen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        horizontal_flip=False)


    #print(X_train.shape)
    print("training set has %d examples" % X_train.shape[0])
    print("validation set has %d examples" % X_valid.shape[0])
    print("test set has %d examples" % X_test.shape[0])
    
    #model = compile_model_wide(network, x_train)
    print('compiling model')
    model = compile_model(network, X_train)
    model.summary()
    
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=5, \
                              verbose=1, mode='auto')
    outputModel = 'outputModel'
    
    model_json = model.to_json()
    with open(outputModel+".json", "w") as json_file:
        json_file.write(model_json)
    modWeightsFilepath=outputModel+".weights.hdf5"
    checkpoint = ModelCheckpoint(modWeightsFilepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')

    callbacks_list = [earlystop,checkpoint]
    
    datagen.fit(X_train)
    validation_gen.fit(X_valid)
    test_gen.fit(X_test)
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32), \
                        steps_per_epoch=len(X_train) / 32, epochs=100,verbose=1, \
                        callbacks=callbacks_list, \
                        validation_data=validation_gen.flow(X_valid,Y_valid, batch_size=32), \
                        validation_steps=len(X_test)/32)

    score = model.evaluate_generator(test_gen.flow(X_test,Y_test, batch_size=32),len(Y_test)/32)
    
    print('returning the score')
    return score[1]  # 1 is accuracy. 0 is loss.
    """
    
    with open('/home/noor/faststorage/popGen/final_experiments/admixed/data/notTransposed/train/dictionary_5000/final_snp.json', 'r') as fp:
        partition_snp = json.load(fp)
    
    #with open('/home/noor/faststorage/popGen/final_experiments/admixed/data/optimization/dict_transposed/final_pos.json', 'r') as fp:
        #partition_pos = json.load(fp)
    
    
    with open('/home/noor/faststorage/popGen/final_experiments/admixed/data/notTransposed/train/dictionary_5000/final_label.json', 'r') as fp:
        labels = json.load(fp) 
    
    params = {'batch_size': 16,
          'n_classes': 5,
          'shuffle': True}

    inFilePath = '/home/noor/faststorage/popGen/final_experiments/admixed/scripts/models/GA/model_5000_144_2D'
    #training_generator = DataGenerator(partition_snp['train'], partition_pos['train'],labels, **params)
    #validation_generator = DataGenerator(partition_snp['valid'],partition_pos['valid'], labels, **params)
    training_generator = DataGenerator(partition_snp['train'],labels, **params)
    validation_generator = DataGenerator(partition_snp['valid'], labels, **params)
        
    print('compiling model')
    model = compile_model(network)
    model.summary()
    
    earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=3, verbose=1, mode='auto')

    model_json = model.to_json()
    with open(inFilePath+".json", "w") as json_file:
        json_file.write(model_json)
    modWeightsFilepath=inFilePath+".weights.hdf5"
    checkpoint = ModelCheckpoint(modWeightsFilepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')
    
    callbacks_list = [earlystop,checkpoint]
    
    model.fit_generator(generator=training_generator,epochs=10,validation_data=validation_generator,
                    use_multiprocessing=False,callbacks=callbacks_list)
    
    
    score = model.evaluate_generator(validation_generator)
    print('returning the score')
    return score[1]  # 1 is accuracy. 0 is loss.