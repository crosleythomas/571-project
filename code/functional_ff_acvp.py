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
import sys
import scipy.io as sio
import numpy as np
import time, datetime

# Switch for using the feed-forward vs recurrent architecture
# The only difference for the ff vs recurrent architectures is
#	the use of an LSTM on the high-level encoded feature layer
#	(right before the last dense layer before action transformation)
action_conditional = 1
recurrent = 0
display_network_sizes = 1
save_model = 1

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

data_file = '../data/sprites/sprites_walled_training.npz'
data_file = sys.argv[1] if len(sys.argv) > 1 else data_file

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
hidden_size = 10

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
deconv_sizes = conv_sizes[:]
deconv_sizes.reverse()

pixels = 40 

# Dense layer sizes mirrored around the point-wise product between 
# actions and high-level feature representation
dense_output_sizes = [100, hidden_size]
dedense_output_sizes = [dense_output_sizes[0], 4800]
deconv_input_size = (3, pixels, pixels)
deconv_output_sizes = [(-1, 3, pixels, pixels), (None, 1, pixels, pixels)]

# Frames Branch
frames_input = Input(shape=(num_input_channels, input_height, input_width))

###############################################
### 			Step 1: Encoding 			###
###############################################

# Add convolutional layer(s)
# Convolution2D(num_filters, conv_width)
conv1 = Convolution2D(conv_sizes[0][0], conv_sizes[0][1], conv_sizes[0][2], init='glorot_uniform', activation='relu', border_mode='same')(frames_input)
conv2 = Convolution2D(conv_sizes[1][0], conv_sizes[1][1], conv_sizes[1][2], init='glorot_uniform', activation='relu', border_mode='same')(conv1)

# Vectorize the output of the convolutional layers to be used in FC layers
features_flatten = Flatten()(conv2)

# Add fully-connected layer(s) WITH non-linearity
features_d1 = Dense(dense_output_sizes[0], activation='relu')(features_flatten)

# Add fully-connected layer(s) WITHOUT non-linearity
features_d2 = Dense(dense_output_sizes[1], init='glorot_uniform')(features_d1)

###########################################
### 	Step 2: Action transformation 	###
###########################################
if action_conditional:
	# Action Branch
	action_input = Input(shape=(action_size,))
	# Action-feature mapping
	action_projection = Dense(hidden_size, input_dim=action_size, init='glorot_uniform')(action_input)
	# Point-wise multiplicative combination
	encoded = merge([features_d2, action_projection], mode='mul')
else:
	encoded = features_d2

###########################################
### 		Step 3: Deconvolution 		###
###########################################

# Fully-connected reshaping to form a 3D feature map
dedense1 = Dense(dedense_output_sizes[0], init='glorot_uniform')(encoded)
dedense2 = Dense(dedense_output_sizes[1], init='glorot_uniform', activation='relu')(dedense1)
dedense2_reshaped = Reshape(deconv_input_size)(dedense2)

# Add deconvolutional layer(s)
deconv1 = Deconvolution2D(deconv_sizes[0][0], deconv_sizes[0][1], deconv_sizes[0][2], output_shape=(deconv_output_sizes[0]), border_mode='same')(dedense2_reshaped)
deconv2 = Deconvolution2D(num_output_channels, deconv_sizes[1][1], deconv_sizes[1][2], output_shape=(None, num_output_channels, input_height, input_width), border_mode='same')(deconv1)

# Make the model
if action_conditional:
	model_training_input = [input_frames, actions]
	model = Model(input=[frames_input, action_input], output=deconv2)
else:
	model_training_input = input_frames
	model = Model(input=frames_input, output=deconv2)

###########################################
### 		Step 4: Train! 				###
###########################################
print('Compiling model...\n')
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

model.summary()
plot(model, to_file='../figures/functional_ff_model.png')

print('Training...\n')

# TODO: check if we need to disable shuffling for the recurrent model
model.fit(model_training_input, labels, verbose=1, nb_epoch=10, batch_size=1, validation_split=0.3)
print('Training completed...\n')

# Save the model in HD5 format
if save_model:
	print('Find saved model at: ' + model_output_file + '.h5')
	model.save(model_output_file + '.h5')