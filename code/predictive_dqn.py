#!/usr/bin/env python
####################################################################
# Implementation of Oh, et al. action-conditoinal video prediction #
#	in keras.													   #
#																   #
# UW CSE 571 Project											   #
# Thomas Crosley and Karan Singh								   #
####################################################################

# Keras Imports
from keras.models import *
from keras.layers import *
from keras.utils.visualize_util import plot

# Other Imports
import scipy.io as sio
import numpy as np
import time, datetime

###############################
###        Load data        ###
###############################

# Load frame data as numpy arrays
#	Input has to be arranged such that, even if the input for a single training
#	example has multiple concatenated frames, all frames are a grouped in the
#	training data.
#	input_size = frames * height * width * channels
#	size(frames) = [input_size, input_size]
print('Loading data...\n')
data_file = '../data/sprites/single_input_sprites_training.npz'
data = np.load(data_file)
input_frames = data['frames']
labels = data['labels']

# Save the trained model to a file
ts = time.time()
training_timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

arch_str = 'ff'
if recurrent:
	arch_str = 'rec'

data_name = data_file.split('/')[-1].split('.')[0]
model_output_file = '../data/trained_models/' + data_name + '_' + arch_str + '_' + training_timestamp

# Load action data
#	There are N timesteps, t = 0 is the initial step, t = i is the system at time i
#	range: actions[0]...actions[N-1], timesteps from [0...N-1]
#	actions[t] is a one-hot binary vector of actions
#	size(actions) = [timesteps, num_actions]
actions = data['actions']


