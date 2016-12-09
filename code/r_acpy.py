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

# Other Imports
import scipy.io as sio
import numpy as np
import time, datetime

action_conditional = 1
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
input_frames = input_frames[0:30*300,:,:,:]
input_frames = np.reshape(input_frames, [30, 300, 3, 40, 40])
labels = data['labels']
labels = labels[0:30*300,:,:,:]
labels = np.reshape(labels, [30, 300, 1, 40, 40])

# Save the trained model to a file
ts = time.time()
training_timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

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
actions = actions[0:30*300]
actions = np.reshape(actions, [30, 300, 1])

print('Actions input size:')
print(actions.shape)
print('')
print('Frames input size:')
print(input_frames.shape)
print('')
print('Labels size:')
print(labels.shape)
print('')

#####################################
### Setup arguments based on data ###
#####################################
action_size = 1
num_seqs = input_frames.shape[0]
seq_length = input_frames.shape[1]
num_input_channels = input_frames.shape[2]
num_output_channels = labels.shape[2]
input_height = input_frames.shape[3]
input_width = input_frames.shape[4]
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
conv_sizes = [[2,3,3],[1,3,3]]
deconv_sizes = conv_sizes[:]
deconv_sizes.reverse()
final_conv_layer_index = 1

# Dense layer sizes mirrored around the point-wise product between 
# actions and high-level feature representation
dense_output_sizes = [100, hidden_size]

# Frames Branch
encoding = Sequential()

###############################################
### 			Step 1: Encoding 			###
###############################################

# The overall model
model = Sequential()

# Add convolutional layer(s)
# Convolution2D(num_filters, conv_width)
encoding.add(Convolution2D(conv_sizes[0][0], conv_sizes[0][1], conv_sizes[0][2], init='glorot_uniform', activation='relu', border_mode='same', input_shape=(num_input_channels, input_height, input_width)))
encoding.add(Convolution2D(conv_sizes[1][0], conv_sizes[1][1], conv_sizes[1][2], init='glorot_uniform', activation='relu', border_mode='same'))

dedense_output_sizes = [dense_output_sizes[0], np.prod(encoding.layers[-1].output_shape[1:])]

# Vectorize the output of the convolutional layers to be used in FC layers
encoding.add(Flatten())

# Add fully-connected layer(s) WITH non-linearity
encoding.add(Dense(dense_output_sizes[0], activation='relu'))

# Add the encoding part to the overall model
model.add(TimeDistributed(encoding, input_shape=encoding.layers[0].input_shape))

# Add recurrent part (LSTM)
model.add(LSTM(128, return_sequences=True))

# Add fully-connected layer(s) WITHOUT non-linearity
model.add(TimeDistributed(Dense(dense_output_sizes[1], init='glorot_uniform')))

###########################################
### 	Step 2: Action transformation 	###
###########################################
if action_conditional:
	# Action Branch
	amodel = Sequential()
	# Action multiplication
	amodel.add(TimeDistributed(Dense(hidden_size, init='glorot_uniform'), input_shape=(None,action_size)))
	# Point-wise multiplicative combination
	encoded = Merge([model, amodel], mode='mul')
	dmodel = Sequential()
	dmodel.add(encoded)
	model_training_input = [input_frames, actions]
else:
	# Input to deconvolutional branch is just the encoded state vector
	dmodel = Sequential()
	dmodel.add(model)
	model_training_input = input_frames

###########################################
### 		Step 3: Deconvolution 		###
###########################################

# Fully-connected reshaping to form a 3D feature map
deconv = Sequential()
deconv.add(Dense(dedense_output_sizes[0], init='glorot_uniform', input_dim=dmodel.output_shape[-1]))
deconv.add(Dense(dedense_output_sizes[1], init='glorot_uniform', activation='relu'))
deconv.add(Reshape(encoding.layers[final_conv_layer_index].output_shape[1:]))

# Add deconvolutional layer(s)
deconv.add(Deconvolution2D(deconv_sizes[0][0], deconv_sizes[0][1], deconv_sizes[0][2], output_shape=(encoding.layers[final_conv_layer_index].input_shape), border_mode='same'))
deconv.add(Deconvolution2D(num_output_channels, deconv_sizes[1][1], deconv_sizes[1][2], output_shape=(encoding.layers[final_conv_layer_index-1].input_shape), border_mode='same'))

dmodel.add(TimeDistributed(deconv))

###########################################
### 		Step 4: Train! 				###
###########################################
print('Compiling model...\n')
dmodel.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])


### Display some information about the network shape

print('First we set up the encoding branch')
for l in encoding.layers:
	print(l)
	print(l.input_shape)
	print(l.output_shape)

print('These are all wrapped in a TimeDistributed layer with overall input shape: ' + str(model.layers[0].input_shape))
print('\tand output shape ' + str(model.layers[0].output_shape))

print(model.layers[1])
print(model.layers[1].input_shape)
print(model.layers[1].output_shape)

print('Then another TimeDistributed dense following the LSTM')
print(model.layers[2])
print(model.layers[2].input_shape)
print(model.layers[2].output_shape)

print('Then we set up the action conditioning branch')
print('First a TimeDistributed dense')
print(amodel.layers[0])
print(amodel.layers[0].input_shape)
print(amodel.layers[0].output_shape)

print('Then we merge the action and encoding branches together')
print(dmodel.layers[0])
print(dmodel.layers[0].input_shape)
print(dmodel.layers[0].output_shape)

print('Then we set up the deconvolutional part')
for l in deconv.layers:
	print(l)
	print(l.input_shape)
	print(l.output_shape)

print('These are all wrapped in a TimeDistributed layer with overall input shape: ' + str(dmodel.layers[1].input_shape))
print('and output shape ' + str(dmodel.layers[1].output_shape))


"""
if display_network_sizes:
	print('Encoding layers...')
	for l in encoding.layers:
		print('\t' + str(l))
		print('\tInput Shape: ' + str(l.input_shape))
		print('\tOutput Shape: ' + str(l.output_shape))
		print('')

	if action_conditional:
		print('Action layers...')
		for l in amodel.layers:
			print('\t' + str(l))
			print('\tInput Shape: ' + str(l.input_shape))
			print('\tOutput Shape: ' + str(l.output_shape))
			print('')

	print('Deconvolution layers...')
	for l in dmodel.layers:
		print('\t' + str(l))
		print('\tInput Shape: ' + str(l.input_shape))
		print('\tOutput Shape: ' + str(l.output_shape))
		print('')
"""

print('Training...\n')

dmodel.fit(model_training_input, labels, verbose=1, nb_epoch=5, batch_size=1)
print('Training completed...\n')

# Save the model in HD5 format
dmodel.save(model_output_file + '.h5')