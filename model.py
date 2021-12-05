#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:52:41 2019

@author: au560049
"""

import json
import numpy as np
from generator import DataGenerator
import tensorflow
KERAS_BACKEND=tensorflow
import keras
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv1D, AveragePooling1D, MaxPooling1D
from keras.layers.merge import concatenate
import os
import sys
from keras.callbacks import EarlyStopping,ModelCheckpoint

#'final_snp.json'

def base_model(kernel_size,lambda_rate,x,y,kernel_initializer):
	input = Input(shape=(x, y)) ## 4986x144

	cnn = Conv1D(320, kernel_size=kernel_size,activation='relu',kernel_regularizer=keras.regularizers.l2(lambda_rate),name='conv1')(input)
	cnn = Conv1D(320, kernel_size=kernel_size, activation='relu',kernel_regularizer=keras.regularizers.l2(lambda_rate),name='conv2')(cnn)
	cnn = MaxPooling1D(pool_size=2,name='maxPool_1')(cnn)
	cnn = Dropout(0.2,name='dropout_1')(cnn)

	cnn = Conv1D(320, kernel_size=kernel_size, activation='relu',kernel_regularizer=keras.regularizers.l2(lambda_rate),name='conv3')(cnn)
	cnn = MaxPooling1D(pool_size=2,name='maxPool_2')(cnn)
	cnn = Dropout(0.2,name='dropout_2')(cnn)

	cnn = Conv1D(320, kernel_size=kernel_size, activation='relu',kernel_regularizer=keras.regularizers.l2(lambda_rate),name='conv4')(cnn)
	cnn = MaxPooling1D(pool_size=2,name='maxPool_3')(cnn)
	cnn = Dropout(0.2,name='dropout_3')(c1)
	
	cnn = Conv1D(320, kernel_size=kernel_size, activation='relu',kernel_regularizer=keras.regularizers.l2(lambda_rate),name='conv5')(cnn)
	cnn = MaxPooling1D(pool_size=2,name='maxPool_4')(cnn)
	cnn = Dropout(0.2,name='dropout_4')(cnn)

	cnn = Flatten()(cnn)
	
	
	fnn = Dense(128, activation='relu',kernel_regularizer=keras.regularizers.l2(lambda_rate),name='fn')(cnn)
	fnn = Dropout(0.2,name='dropout_5')(fnn)
	output = Dense(5, activation='softmax',name='output_layer')(fnn)

	model = Model([i1], [output])

	model.summary()

	model.compile(loss=keras.losses.categorical_crossentropy,
					  optimizer=keras.optimizers.Adam(),
					  metrics=['accuracy'])
					  
	return model
	
	
if __name__ == "__main__":

	kernel_size, inFilePath_snp, inFilePath_labels, outFilePath = sys.argv[1:]

	with open(inFilePath_snp, 'r') as fp:
	partition_snp = json.load(fp)


	with open(inFilePath_labels, 'r') as fp:
	labels = json.load(fp) 


	params = {'batch_size': 64,
		  'n_classes': 5,
		  'shuffle': True}


	training_generator = DataGenerator(partition_snp['train'],labels, **params)
	validation_generator = DataGenerator(partition_snp['valid'], labels, **params)

	kernel_size = int(kernel_size)
	lambda_rate =  0.0001

	base_model(kernel_size,lambda_rate,4986,144)

	earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=5, verbose=1, mode='auto')

	model_json = model.to_json()
	with open(outFilePath+"model.json", "w") as json_file:
	json_file.write(model_json)
	modWeightsFilepath=outFilePath+"model.weights.hdf5"
	checkpoint = ModelCheckpoint(modWeightsFilepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='auto')

	callbacks_list = [earlystop,checkpoint]

	model.fit_generator(generator=training_generator,epochs=20,validation_data=validation_generator,
					use_multiprocessing=True,
					workers=18,callbacks=callbacks_list)

