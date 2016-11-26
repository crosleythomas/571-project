#!/usr/bin/env python
####################################################################
# Implementation of Oh, et al. action-conditoinal video prediction #
#	in keras.													   #
#																   #
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

# Switch for using the feed-forward vs recurrent architecture
# The only difference for the ff vs recurrent architectures is
#	the use of an LSTM on the high-level encoded feature layer
#	(right before the last dense layer before action transformation)
action_conditional = 0
recurrent = 0
display_network_sizes = 1

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
data_file = '../data/sprites/sprites_training.npz'
data = np.load(data_file)
input_frames = data['frames']
labels = data['labels']

# Save the trained model to a file
ts = time.time()
training_timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

arch_str = 'ff'
if recurrent:
	arch_str = 'rec'

ac_str = 'nac'
if action_conditional:
	ac_str = 'ac'

data_name = data_file.split('/')[-1].split('.')[0]
model_output_file = '../data/trained_models/' + data_name + '_' + ac_str + '_' + arch_str + '_' + training_timestamp

# Load action data
#	There are N timesteps, t = 0 is the initial step, t = i is the system at time i
#	range: actions[0]...actions[N-1], timesteps from [0...N-1]
#	actions[t] is a one-hot binary vector of actions
#	size(actions) = [timesteps, num_actions]
actions = data['actions']

#####################################
### Setup arguments based on data ###
#####################################
action_size = 1
num_frames = input_frames.shape[0]
num_input_channels = input_frames.shape[1]
num_output_channels = labels.shape[1]
input_height = input_frames.shape[2]
input_width = input_frames.shape[3]
num_input_frames = 3
input_size = num_input_frames * num_input_channels * input_height * input_width
hidden_size = 100

###################################################
### 				Build Model 				###
###################################################

####################################################
### 			Layer sizes 					 ###
### The convolutional and deconvolutional layer  ###
### sizes are mirrored around the merge with actions
####################################################
# conv_sizes[i][0] = num_filters in ith layer
# convsizes[i][0] = filter height in ith layer
# conv_sizes[i][0] = filter width in ith layer
print('Building model...\n')
conv_sizes = [[3,3,3],[3,3,3]]
deconv_sizes = conv_sizes
deconv_sizes.reverse()
final_conv_layer_index = 1

# Dense layer sizes mirrored around the point-wise product between 
# actions and high-level feature representation
dense_output_sizes = [100, hidden_size]

# Frames Branch
fmodel = Sequential()

###############################################
### 			Step 1: Encoding 			###
###############################################

# Add convolutional layer(s)
# Convolution2D(num_filters, conv_width)
fmodel.add(Convolution2D(conv_sizes[0][0], conv_sizes[0][1], conv_sizes[0][2], init='glorot_uniform', activation='relu', border_mode='same', input_shape=(num_input_channels, input_height, input_width)))
fmodel.add(Convolution2D(conv_sizes[1][0], conv_sizes[1][1], conv_sizes[1][2], init='glorot_uniform', activation='relu', border_mode='same'))

dedense_output_sizes = [dense_output_sizes[0], np.prod(fmodel.layers[-1].output_shape[1:])]

# Vectorize the output of the convolutional layers to be used in FC layers
fmodel.add(Flatten())
if not recurrent:
	# Add fully-connected layer(s) WITH non-linearity
	fmodel.add(Dense(dense_output_sizes[0], activation='relu'))
else:
	fmodel.add(LSTM(dense_output_sizes[0]))

# Add fully-connected layer(s) WITHOUT non-linearity
fmodel.add(Dense(dense_output_sizes[1], init='glorot_uniform'))

###########################################
### 	Step 2: Action transformation 	###
###########################################
if action_conditional:
	# Action Branch
	amodel = Sequential()

	# Action multiplication
	amodel.add(Dense(hidden_size, input_dim=action_size, init='glorot_uniform'))

	# Point-wise multiplicative combination
	encoded = Merge([fmodel, amodel], mode='mul')
	dmodel = Sequential()
	dmodel.add(encoded)
	model_training_input = [input_frames, actions]
else:
	# Input to deconvolutional branch is just the encoded state vector
	dmodel = Sequential()
	dmodel.add(fmodel)
	model_training_input = input_frames

###########################################
### 		Step 3: Deconvolution 		###
###########################################

# Fully-connected reshaping to form a 3D feature map
dmodel.add(Dense(dedense_output_sizes[0], init='glorot_uniform'))
dmodel.add(Dense(dedense_output_sizes[1], init='glorot_uniform', activation='relu'))
dmodel.add(Reshape(fmodel.layers[final_conv_layer_index].output_shape[1:]))

# Add deconvolutional layer(s)
dmodel.add(Deconvolution2D(deconv_sizes[0][0], deconv_sizes[0][1], deconv_sizes[0][2], output_shape=(fmodel.layers[final_conv_layer_index].input_shape), border_mode='same'))
dmodel.add(Deconvolution2D(num_output_channels, deconv_sizes[1][1], deconv_sizes[1][2], output_shape=(fmodel.layers[final_conv_layer_index-1].input_shape), border_mode='same'))

###########################################
### 		Step 4: Train! 				###
###########################################
print('Compiling model...\n')
dmodel.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

if display_network_sizes:
	for l in fmodel.layers:
		print(l)
		print(l.input_shape)
		print(l.output_shape)
		print('')

	if action_conditional:
		for l in amodel.layers:
			print(l)
			print(l.input_shape)
			print(l.output_shape)
			print('')

	for l in dmodel.layers:
		print(l)
		print(l.input_shape)
		print(l.output_shape)
		print('')

print('Training...\n')

dmodel.fit(model_training_input, labels, verbose=1, nb_epoch=5, batch_size=1)
print('Training completed...\n')

# Save the model in HD5 format
dmodel.save(model_output_file + '.h5')