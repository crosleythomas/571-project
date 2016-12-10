#!/usr/bin/env python
####################################################################
# Implementation of Oh, et al. action-conditoinal video prediction #
#	in keras.  Testing how changing the hidden unit size and the   #
#	bottleneck affects predictions.								   #
# UW CSE 571 Project											   #
# Thomas Crosley and Karan Singh								   #
####################################################################

# Keras Imports
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Merge, Reshape, Flatten, Deconvolution2D, Input, LSTM

# Other Imports
import scipy.io as sio
import numpy as np
import time, datetime
import sys
import gc
sys.path.insert(0, '../code')
import matplotlib.pyplot as plt
plt.switch_backend('Qt4Agg')
plt.ion()

# Local imports
from ff_acvp import ffnn

hidden_unit_min_size = 1
hidden_unit_max_size = 10
hidden_unit_step_size = 1

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
actions = data['actions']
model_training_input = [input_frames, actions]

histories = []
for hidden_size in range(hidden_unit_min_size, hidden_unit_max_size, hidden_unit_step_size):
	print('Model with hidden size of : ' + str(hidden_size))
	# Create new network with new hidden size
	net_obj = ffnn(hidden_size)
	model = net_obj.model
	# Train on the same data with a split for validation data
	history = model.fit(model_training_input, labels, verbose=1, nb_epoch=10, batch_size=1, validation_split=0.3)
	histories.append(history)
	gc.collect()



plt.plot(history.history['mean_squared_error'])
plt.show()