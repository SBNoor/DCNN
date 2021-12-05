#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:31:44 2018

@author: Noor
"""

import numpy as np
from glob import glob
from os import getcwd, chdir
import random

###Setting up global variables
LST = ['pi','thetaW','tajD','distVar','distSkew','distKurt','nDiplos','diplo_H1_win','diplo_H12_win',
       'diplo_H2/H1_win','diplo_ZnS_win','diplo_Omega']

Y_label = {'soft':0,'hard':1,'neutral':2,'linkedhard':3,'linkedSoft':4}


def files_in_a_directory(directory,extension):
    """Extracting file names from directory of interest.
    Args: 
        directory (string): Path to directory
        extension (string): Extension of files of interest      
    Returns:
        files (list): filenames
    """
    
    files = []
    
    ###Saving the current working directory
    saved = getcwd()
    print('saved : ',saved)
    
    ###Switching to directory of interest i.e. the one with input files
    #chdir(directory)
    files = glob('*.' + extension)
    
    ###Switching back to original directory
    chdir(saved)
    
    print('files : ',files)
    
    return files


def shuffling_dataset(images,Y):
    
    """Shuffling numpy arrays aka. images and corresponding 'Y' lables randomly.
    Args:
        images (numpy array): List of numpy arrays (input data), and 
         Y (numpy array): list of output labels    
    Returns: 
        Shuffled images and Y
    """
    
    ###Zipping both of numpy arrays so that they are shuffled in unison
    combined = list(zip(images, Y))
    random.shuffle(combined)
    
    ###Unzipping the arrays
    images[:], Y[:] = zip(*combined)
    
    return images,Y

def reading_file(files,directory):
    
    """Reading file (hard, soft hard-linked, soft-linked, neutral) and creating an image representation.
    Args: 
        filename (NOTE TO SELF: perhaps change it with directory later on. I don't know yet. Will check later.)
    Returns: 
        list of numpy arrays where by each numpy array will the image representation of some sweep-type.
    
    """
        
    images = []
    Y = []
    
    for filename in files:
        #path_to_file = os.path.join(directory,filename)
        #print('path to file : ',path_to_file)
        file = open(filename, 'r')
        lines = file.readlines()
        
        ###Extracting first row from the file
        print('line : ',lines[0])
        statistics_names = lines[0].strip('\n').split('\t')
        
        ###Creating images.        
        #print('filename : ', filename)
        sweep_type = filename.split('.')
        
        
        l = [sweep_type[0]] * (len(lines)-1)
        Y.extend(l)
    
        for line in lines[1:]: ###started from 1 because I've already extracted the first row
            line = line.strip('\n').split('\t') ###Step 1 done
    
            ###Creating a numpy array which will be appended to a list ('images.'). Each array represents an image
            ###Size of each image is 12 summary statistics * 11 subwindows. 
            array = np.zeros((len(LST),11)) 
            
            for k in range(len(LST)):
                indices = [i for i, elem in enumerate(statistics_names) if LST[k] in elem]
                index = 0
                for num in indices:
                    array[k][index] = float(line[num])
                    index += 1

            images.append(array)
        
    ###shuffling the numpy arrays in images and corresponding labels in Y.
    images, Y = shuffling_dataset(images,Y)
    
    return images,Y


def summary_statistics_reader(directory,extension):
    
    ###retrieving filenames in the directory(of interest)
    files = files_in_a_directory(directory,extension)
    
    ###reading the files
    images,Y = reading_file(files)
    
    return images, Y, files

"""
if __name__ == "__main__":
    
    extension = 'txt'
    directory = '/Users/au560049/Documents/CNN-LSTM'
    
    ###retrieving filenames in the directory(of interest)
    files = files_in_a_directory(directory,extension)
    
    ###reading the files
    images,Y = reading_file(files)
    
    
    
    Y1 = Y
    ###Integer encoding of Y
    Y = integer_encoding(Y)
    
    ###One-hot encoding of Y
    Y = encode_Y(Y)
    
    images = np.asarray(images)
    Y = np.asarray(Y)
    
    x_train = images.reshape(10000,12,11,1)
    #x_test = x_test.reshape(10000,28,28,1)
    
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(12,11,1)))
    
"""

    
    
    
    
    
    
    