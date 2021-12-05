#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    def __init__(self,list_IDs_snp,list_IDs_pos,labels,batch_size=32,n_classes=5,shuffle=True):
        self.batch_size = batch_size
        self.list_IDs_snp = list_IDs_snp
        self.list_IDs_pos = list_IDs_pos
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.labels = labels
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs_snp) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp_snp = [self.list_IDs_snp[k] for k in indexes]
        list_IDs_temp_pos = [self.list_IDs_pos[k] for k in indexes]

        # Generate data
        snp,pos,y = self.__data_generation(list_IDs_temp_snp,list_IDs_temp_pos)
        

        return [snp,pos],y
        
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs_snp))
        if self.shuffle==True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self,list_IDs_temp_snp,list_IDs_temp_pos):
        snp = np.empty((self.batch_size,4986,144))
        pos = np.empty((self.batch_size,4986))
        y = np.empty((self.batch_size),dtype=int)

        for ID in range(len(list_IDs_temp_snp)):
            snp[ID] = np.load(list_IDs_temp_snp[ID])
            pos[ID] = np.load(list_IDs_temp_pos[ID])
            y[ID] = self.labels[list_IDs_temp_snp[ID]]
               
        return snp,pos,keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    

    
    
            
            
    
    
    
    
    
    
