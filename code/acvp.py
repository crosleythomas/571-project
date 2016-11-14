####################################################################
# Implementation of Oh, et al. action-conditoinal video prediction #
#	in keras.													   #
#																   #
# UW CSE 571 Project											   #
# Thomas Crosley and Karan Singh								   #
####################################################################

# Keras Imports
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Merge, Reshape, Flatten, Deconvolution2D

# Other Imports
import scipy.io as sio
import numpy as np

# Switch for using the feed-forward vs recurrent architecture
# The only difference for the ff vs recurrent architectures is
#	the use of an LSTM on the high-level encoded feature layer
#	(right before the last dense layer before action transformation)
recurrent = 0
display_network_sizes = 0

###############################
###        Load data        ###
###############################

# Load frame data as numpy arrays
#	Input has to be arranged such that, even if the input for a single training
#	example has multiple concatenated frames, all frames are a grouped in the
#	training data.
#	input_size = frames * height * width * channels
#	size(frames) = [input_size, input_size]
frames = np.array([sio.loadmat('../data/8x8_sprite.mat')['sprite']])

# Load action data
#	There are N timesteps, t = 0 is the initial step, t = i is the system at time i
#	range: actions[0]...actions[N-1], timesteps from [0...N-1]
#	actions[t] is a one-hot binary vector of actions
#	size(actions) = [timesteps, num_actions]
actions = np.array([0]*32)
actions[3] = 1

#####################################
### Setup arguments based on data ###
#####################################
action_size = (32)
labels = np.random.random((1, 32))
num_frames = frames.shape[0]
input_height = frames[0].shape[0]
input_width = frames[0].shape[1]
num_input_channels = 1
num_input_frames = 3
input_size = num_input_frames * input_height * input_width * num_input_channels
hidden_size = 2048

frames = frames.reshape((num_frames, num_input_channels, input_width, input_height))
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
conv_sizes = [[64,3,3],[64,3,3]]
deconv_sizes = conv_sizes
deconv_sizes.reverse()
final_conv_layer_index = 1

# Dense layer sizes mirrored around the point-wise product between 
# actions and high-level feature representation
dense_output_sizes = [2048, hidden_size]

# Action Branch
amodel = Sequential()

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
	fmodel.add(LSTM())

# Add fully-connected layer(s) WITHOUT non-linearity
fmodel.add(Dense(dense_output_sizes[1], init='glorot_uniform'))

###########################################
### 	Step 2: Action transformation 	###
###########################################

# Action multiplication
amodel.add(Dense(hidden_size, input_dim=action_size, init='glorot_uniform'))

# Point-wise multiplicative combination
encoded = Merge([fmodel, amodel], mode='mul')

###########################################
### 		Step 3: Deconvolution 		###
###########################################
# Deconvolutional Branch takes the merge branches as input
dbranch = Sequential()
dbranch.add(encoded)

# Fully-connected reshaping to form a 3D feature map
dbranch.add(Dense(dedense_output_sizes[0], init='glorot_uniform'))
dbranch.add(Dense(dedense_output_sizes[1], init='glorot_uniform', activation='relu'))
dbranch.add(Reshape(fmodel.layers[final_conv_layer_index].output_shape[1:]))

# Add deconvolutional layer(s)
dbranch.add(Deconvolution2D(deconv_sizes[0][0], deconv_sizes[0][1], deconv_sizes[0][2], output_shape=(fmodel.layers[final_conv_layer_index].input_shape), border_mode='same'))
dbranch.add(Deconvolution2D(num_input_channels, deconv_sizes[1][1], deconv_sizes[1][2], output_shape=(fmodel.layers[final_conv_layer_index-1].input_shape), border_mode='same'))

###########################################
### 		Step 4: Train! 				###
###########################################
dbranch.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

if display_network_sizes:
	for l in fmodel.layers:
		l
		l.input_shape
		l.output_shape
		print('')

	for l in amodel.layers:
		l
		l.input_shape
		l.output_shape
		print('')

	for l in dbranch.layers:
		l
		l.input_shape
		l.output_shape
		print('')


dbranch.fit([frames, actions], labels)
