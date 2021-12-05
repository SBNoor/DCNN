#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:46:44 2019

@author: au560049
"""

import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    def __init__(self,list_IDs_snp,labels,batch_size=32,n_classes=5,shuffle=True):
        self.batch_size = batch_size
        self.list_IDs_snp = list_IDs_snp
        #self.list_IDs_pos = list_IDs_pos
        #self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.labels = labels
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        #print('In len function')
        #print('length of list ID shit : ',len(self.list_IDs_snp))
        #print('batch size : ',self.batch_size)
        #print('returning : ',int(np.floor(len(self.list_IDs_snp) / self.batch_size)))
        return int(np.floor(len(self.list_IDs_snp) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #print('in get item function')
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        
        #print('indexes : ',indexes)

        # Find list of IDs
        list_IDs_temp_snp = [self.list_IDs_snp[k] for k in indexes]
        #list_IDs_temp_pos = [self.list_IDs_pos[k] for k in indexes]
        #print('list_IDs_temp : ',list_IDs_temp_snp,'   ',list_IDs_temp_pos)

        # Generate data
        #snp,pos,y = self.__data_generation(list_IDs_temp_snp,list_IDs_temp_pos)
        #print('calling data generation function')
        snp,y = self.__data_generation(list_IDs_temp_snp)
        
        #print('shape of snp matrix : ',snp.shape,'   type : ',type(snp))
        #print('shape of pos vector : ',pos.shape, '    type : ',type(pos))
        #print('shape of y vector : ',y.shape, '    type : ',type(y))
        
        #print('back in get item function and returning')

        return [snp],y
        #yield ({'input_1': snp, 'input_2': pos}, {'output': y})
        
    def on_epoch_end(self):
        #print('In on_epoch_end function')
        self.indexes = np.arange(len(self.list_IDs_snp))
        #print('self_indexes : ',self.indexes)
        if self.shuffle==True:
            np.random.shuffle(self.indexes)
        #print('end on_epoch_function and therefore, teh shuffeled indexes : ',self.indexes)
        #print('shape of snp matrix : ',snp.shape)
        #print('shape of pos vector : ',pos.shape)
            
    def __data_generation(self,list_IDs_temp_snp):
        snp = np.empty((self.batch_size,144,5000))
        #pos = np.empty((self.batch_size,1000))
        y = np.empty((self.batch_size),dtype=int)
        #snp = []
        #pos = []
        #y = []
        
        #print('In data generation function. Magic happens here')
        for ID in range(len(list_IDs_temp_snp)):
            #print('ID : ',ID)
            #snp.append(np.load(list_IDs_temp_snp[ID]))
            #pos.append(np.load(list_IDs_temp_pos[ID]))
            snp[ID] = np.load(list_IDs_temp_snp[ID])
            #pos[ID] = np.load(list_IDs_temp_pos[ID])
            y[ID] = self.labels[list_IDs_temp_snp[ID]]
            
            #y.append(self.labels[ID])
        snp = snp.reshape(self.batch_size,144,5000,1)
        #return np.asarray(snp),np.asarray(pos),keras.utils.to_categorical(y, num_classes=self.n_classes)
        #print('returning snp and pos numpy arrays')
        return snp,keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    

    
    
            
            
    
    
    
    
    
    
